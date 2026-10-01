# Unseen-species detection: evaluate q(x) + species-head confidence + their
# fusion, phase by phase, on top of the same closed-form pipeline as
# main_analytic_cnn14mv2.py. At each phase, the species about to be
# introduced are scored as "novel" against everything seen so far, using
# only the statistics accumulated BEFORE this phase is trained on.
import warnings
warnings.filterwarnings('ignore')
import os
import time
import logging
import argparse

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from models.audio_model import Audio_Frontend, AudioModel
from models.cnn14_mobilev2 import Cnn14_mobilev2
from core.incremental_eval import load_ckpt
from core.analytic_trainer import _extract_embeddings
from core.analytic_head import compute_batch_stats
from core.open_set_detection import (
    build_species_index, update_species_head, score_novelty,
    evaluate_phase_detection, youden_threshold, threshold_metrics,
    extract_embeddings_with_species,
)
from dataset.fish_incremental_dataset import (
    INCREMENTAL_PHASES, build_phase_data, build_all_phases_test_data, build_phase_data_with_species,
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/analytic_cnn14mv2.yaml')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    workspace = config['Workspace']
    exp_name = config['Exp_name'] + '_open_set'
    audio_features = config['Audio_features']
    Training = config['Training']
    base_path = config['Dataset']['base_path']
    opleg_base_path = config['Dataset'].get('opleg_base_path')
    seed = Training['seed']
    ridge_lambda = Training.get('ridge_lambda', 1.0)
    random_proj_dim = Training.get('random_proj_dim', 8192)
    embed_dim = 1024
    sample_rate = audio_features['sample_rate']

    log_dir = os.path.join(workspace, exp_name, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f'{exp_name}-{time.time():.0f}.log')),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    species_to_idx, n_species = build_species_index()
    logger.info(f'species order: {list(species_to_idx.keys())}')

    frontend = Audio_Frontend(**audio_features)
    backbone = Cnn14_mobilev2(classes_num=Training['classes_num'])
    model = AudioModel(frontend=frontend, backbone=backbone).to(device)
    load_ckpt(Training['phase0_ckpt'], model)
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.frontend.bn0.parameters():
        p.requires_grad = False
    model.eval()

    gen = torch.Generator().manual_seed(42)
    proj_w = torch.empty(embed_dim, random_proj_dim)
    torch.nn.init.kaiming_normal_(proj_w, generator=gen)
    proj_w = proj_w.to(device)
    proj_b = torch.zeros(random_proj_dim, device=device)

    def project(emb_raw):
        return F.relu(emb_raw.to(device) @ proj_w + proj_b).cpu()

    A = B_coral = A_sp = B_sp = beta_sp = None
    per_phase = []

    for phase_id in range(len(INCREMENTAL_PHASES)):
        logger.info(f'\n=== Phase {phase_id} ===')
        train_dict, _, test_dict_own = build_phase_data(
            phase_id, base_path=base_path, seed=seed, opleg_base_path=opleg_base_path)
        emb_raw, y_train = _extract_embeddings(model, train_dict, device, sample_rate)
        emb_train = project(emb_raw)

        if phase_id > 0:
            emb0_raw, _ = _extract_embeddings(model, test_dict_own, device, sample_rate)
            emb0 = project(emb0_raw)
            A_inv_prev = torch.linalg.inv(A + ridge_lambda * torch.eye(A.shape[0])).to(device)
            seen_pool = build_all_phases_test_data(
                phase_id - 1, base_path=base_path, seed=seed, opleg_base_path=opleg_base_path)
            emb_seen_raw, _ = _extract_embeddings(model, seen_pool, device, sample_rate)
            emb_seen = project(emb_seen_raw)

            q_novel, conf_novel = score_novelty(emb0, A_inv_prev, beta_sp, device)
            q_seen, conf_seen = score_novelty(emb_seen, A_inv_prev, beta_sp, device)
            result = evaluate_phase_detection(q_novel, conf_novel, q_seen, conf_seen)
            result['phase'] = phase_id
            per_phase.append(result)
            logger.info(f'  AUROC q={result["auroc_q"]:.4f} conf={result["auroc_conf"]:.4f} '
                        f'fused={result["auroc_fused"]:.4f}')

        A_new, B_new = compute_batch_stats(emb_train, y_train)
        A = A_new if A is None else A + A_new
        B_coral = B_new if B_coral is None else B_coral + B_new
        # (beta_coral itself isn't needed here -- this script only tracks
        # the detection signals; main_analytic_cnn14mv2.py tracks accuracy.)

        # species head uses its own per-sample tagged load (handles Phase 0's
        # two species correctly; build_phase_data's plain (wav, label) pairs
        # don't carry species identity beyond the phase they came from)
        train_dict_sp, _, _ = build_phase_data_with_species(
            phase_id, base_path=base_path, seed=seed, opleg_base_path=opleg_base_path)
        emb_sp_raw, _, species_names = extract_embeddings_with_species(
            model, train_dict_sp, device, sample_rate)
        emb_sp = project(emb_sp_raw)
        A_sp, B_sp, beta_sp = update_species_head(
            A_sp, B_sp, emb_sp, species_names, species_to_idx, n_species, ridge_lambda)

    mean_auroc_fused = sum(r['auroc_fused'] for r in per_phase) / len(per_phase)
    logger.info(f'\nMean fused AUROC: {mean_auroc_fused:.4f}')

    # ---- threshold-dependent metrics (pooled Youden's J, applied per phase) ----
    import numpy as np
    all_labels = np.concatenate([r['labels'] for r in per_phase])
    all_fused = np.concatenate([r['fused'] for r in per_phase])
    thresh, tpr, fpr = youden_threshold(all_labels, all_fused)
    logger.info(f"Youden's J threshold = {thresh:.4f} (pooled TPR={tpr:.4f}, FPR={fpr:.4f})")
    pooled_m = threshold_metrics(all_labels, all_fused, thresh)
    logger.info(f"Pooled @ threshold: Sens={pooled_m['sensitivity']*100:.2f}% "
                f"Spec={pooled_m['specificity']*100:.2f}% Prec={pooled_m['precision']*100:.2f}% "
                f"F1={pooled_m['f1']*100:.2f}%")
    for r in per_phase:
        m = threshold_metrics(r['labels'], r['fused'], thresh)
        logger.info(f"  Phase {r['phase']}: Sens={m['sensitivity']*100:.2f}% "
                    f"Spec={m['specificity']*100:.2f}% Prec={m['precision']*100:.2f}% F1={m['f1']*100:.2f}%")
