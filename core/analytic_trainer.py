"""Closed-form analytic class-incremental trainers.

Precondition (guaranteed by the entry-point scripts): Phase 0 has already
been trained with gradient descent (core/incremental_eval.py:train_phase0),
producing a phase0_best.pt checkpoint. From here on the backbone is frozen
and every later phase only needs a single ridge-regression solve on that
phase's own real data -- no replay buffer, no distillation, no epochs.

Why this is exact, not approximate: once the backbone (and therefore the
embedding function) is frozen, the per-phase statistics

    A_t = sum_{i in phase t} x_i x_i^T
    B_t = sum_{i in phase t} x_i y_i^T

can simply be summed across phases (A = sum_t A_t, B = sum_t B_t) and
re-solved (beta = (A + lambda I)^-1 B). This is mathematically identical to
solving the ridge regression once on the union of all phases' real data --
old phases' raw samples never need to be revisited or stored, only the
(A, B) they already contributed.
"""
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.fish_incremental_dataset import (
    INCREMENTAL_PHASES, build_phase_data, build_all_phases_test_data,
    FishTestDataset, collate_fn,
)
from core.incremental_eval import evaluate, evaluate_by_fish, compute_forgetting, coral_predict_intensity, load_ckpt
from core.analytic_head import (
    compute_batch_stats, solve_ridge, solve_ridge_ordinal_coupled,
    set_fc_audioset_from_beta, ordinal_violation_rate, predictive_uncertainty,
)

logger = logging.getLogger()

FROZEN_BACKBONE_BLOCKS = ['conv_block1', 'conv_block2', 'conv_block3', 'conv_block4', 'fc1']


def _load_matching_backbone(model, ckpt_path):
    """Key+shape-filtered load: skips fc_audioset/random_proj, which don't
    exist (or don't match shape) in the Phase-0 gradient checkpoint."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    pretrained_sd = ckpt['model_state_dict']
    own_sd = model.state_dict()
    loaded = {k: v for k, v in pretrained_sd.items()
              if k in own_sd and v.shape == own_sd[k].shape}
    skipped = [k for k in pretrained_sd if k not in loaded]
    own_sd.update(loaded)
    model.load_state_dict(own_sd)
    logger.info(f'  loaded: {ckpt_path} (acc={ckpt.get("acc", float("nan")):.4f}), '
                f'{len(loaded)} matched, {len(skipped)} skipped: {skipped}')
    return model


def _freeze_for_analytic(model):
    for name in FROZEN_BACKBONE_BLOCKS:
        for p in getattr(model.backbone, name).parameters():
            p.requires_grad = False
    for p in model.frontend.bn0.parameters():
        p.requires_grad = False
    logger.info(f'analytic mode: froze backbone.{FROZEN_BACKBONE_BLOCKS} + frontend.bn0; '
                f'only fc_audioset is set by the closed-form solve')


@torch.no_grad()
def _extract_embeddings(model, data_dict, device, sample_rate, batch_size=256):
    loader = DataLoader(FishTestDataset(data_dict, sample_rate), batch_size=batch_size,
                         shuffle=False, num_workers=8, collate_fn=collate_fn)
    model.eval()
    embs, labels = [], []
    for batch in loader:
        out = model(batch['waveform'].to(device))
        embs.append(out['embedding'].cpu())
        labels.append(batch['target'])
    return torch.cat(embs, dim=0), torch.cat(labels, dim=0)


def analytic_trainer(model, config, device, ckpt_dir, phase0_ckpt):
    """Main closed-form pipeline for the PANNs-Cnn10 backbone (with its
    built-in frozen random-projection buffer, see models/panns_cnn10.py)."""
    sample_rate = config['Audio_features']['sample_rate']
    batch_size = config['Training']['Batch_size']
    seed = config['Training']['seed']
    base_path = config['Dataset']['base_path']
    opleg_base_path = config['Dataset'].get('opleg_base_path')
    ridge_lambda = config['Training'].get('ridge_lambda', 1.0)
    coupling_mu = config['Training'].get('coupling_mu', 0.0)

    logger.info(f'loading Phase-0 checkpoint: {phase0_ckpt}')
    model = _load_matching_backbone(model, phase0_ckpt)
    _freeze_for_analytic(model)

    results = {}
    train_dict0, _, test_dict0 = build_phase_data(0, base_path=base_path, seed=seed,
                                                    opleg_base_path=opleg_base_path)

    # ---- Phase 0: analytic refit, replacing the gradient-trained head ----
    emb0, y0 = _extract_embeddings(model, train_dict0, device, sample_rate)
    A, B = compute_batch_stats(emb0, y0)
    beta = solve_ridge_ordinal_coupled(A, B, ridge_lambda, coupling_mu)
    A_inv = torch.linalg.inv(A + ridge_lambda * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)).to(device)
    set_fc_audioset_from_beta(model.backbone.fc_audioset, beta)

    test_loader0 = DataLoader(FishTestDataset(test_dict0, sample_rate), batch_size=batch_size,
                               shuffle=False, num_workers=8, collate_fn=collate_fn)
    test_stats0 = evaluate(model, test_loader0, device, 0, split='test')
    logger.info(f'Phase 0 (analytic refit) Test Acc: {test_stats0["overall_acc"]:.4f}')
    viol_rate0 = ordinal_violation_rate(model, test_loader0, device)
    test_stats0['ordinal_violation_rate'] = viol_rate0
    results[0] = {'test': test_stats0, 'forgetting': 0.0}

    # ---- Phase 1-N: accumulate this phase's real data into (A, B), re-solve ----
    for phase_id in range(1, len(INCREMENTAL_PHASES)):
        logger.info(f'\n{"=" * 60}\nPHASE {phase_id} START (analytic)\n{"=" * 60}')

        train_dict, _, _ = build_phase_data(phase_id, base_path=base_path, seed=seed,
                                             opleg_base_path=opleg_base_path)
        emb, y = _extract_embeddings(model, train_dict, device, sample_rate)

        # Free novel-species signal: Bayesian posterior variance q(x) using
        # the statistics from BEFORE this phase was seen (zero extra cost --
        # reuses the same A the intensity head already accumulated).
        seen_test = build_all_phases_test_data(phase_id - 1, base_path=base_path, seed=seed,
                                                opleg_base_path=opleg_base_path)
        emb_seen, _ = _extract_embeddings(model, seen_test, device, sample_rate)
        q_seen = predictive_uncertainty(emb_seen.to(device), A_inv).mean().item()
        q_novel = predictive_uncertainty(emb.to(device), A_inv).mean().item()
        logger.info(f'Phase {phase_id} novelty check (pre-update stats): '
                    f'seen q={q_seen:.6f}, novel(untrained) q={q_novel:.6f}, ratio={q_novel / q_seen:.2f}x')

        A_new, B_new = compute_batch_stats(emb, y)
        A = A + A_new
        B = B + B_new
        beta = solve_ridge_ordinal_coupled(A, B, ridge_lambda, coupling_mu)
        A_inv = torch.linalg.inv(A + ridge_lambda * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)).to(device)
        set_fc_audioset_from_beta(model.backbone.fc_audioset, beta)

        all_test = build_all_phases_test_data(phase_id, base_path=base_path, seed=seed,
                                               opleg_base_path=opleg_base_path)
        test_loader = DataLoader(FishTestDataset(all_test, sample_rate), batch_size=batch_size,
                                  shuffle=False, num_workers=8, collate_fn=collate_fn)
        test_stats = evaluate(model, test_loader, device, phase_id, split='test')
        test_stats['ordinal_violation_rate'] = ordinal_violation_rate(model, test_loader, device)

        test_stats['phase0_acc'] = evaluate_by_fish(model, phase_id, config, device)
        results[phase_id] = {'test': test_stats}
        results[phase_id]['forgetting'] = compute_forgetting(results, phase_id)
        logger.info(f'Phase {phase_id} Forgetting: {results[phase_id]["forgetting"]:.4f}')

        torch.save({'model_state_dict': model.state_dict(), 'acc': float(test_stats['overall_acc'])},
                   f'{ckpt_dir}/phase{phase_id}_analytic.pt')

    logger.info('\n' + '=' * 60)
    logger.info('Training complete:')
    for pid, result in results.items():
        logger.info(f'Phase {pid} | Acc:{result["test"]["overall_acc"]:.4f} | '
                    f'Forgetting:{result["forgetting"]:.4f}')

    return model, results


# ============================================================
# Backbone-matched variant: Cnn14-MobileV2 (no built-in projection layer,
# so the random projection is applied explicitly outside the model).
# ============================================================

@torch.no_grad()
def _evaluate_projected(model, data_dict, device, sample_rate, beta, proj_w, proj_b, batch_size=256):
    loader = DataLoader(FishTestDataset(data_dict, sample_rate), batch_size=batch_size,
                         shuffle=False, num_workers=8, collate_fn=collate_fn)
    model.eval()
    all_preds, all_targets = [], []
    for batch in loader:
        emb_raw = model(batch['waveform'].to(device))['embedding']
        emb_proj = F.relu(emb_raw @ proj_w + proj_b)
        ones = torch.ones(emb_proj.shape[0], 1, device=emb_proj.device, dtype=emb_proj.dtype)
        x = torch.cat([emb_proj, ones], dim=1)
        pred = coral_predict_intensity(x @ beta.to(emb_proj.device))
        all_preds.append(pred.cpu().numpy())
        all_targets.append(batch['target'].numpy())
    import numpy as np
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return {'overall_acc': float((all_preds == all_targets).mean()),
            'preds': all_preds, 'targets': all_targets}


def _evaluate_by_fish_projected(model, phase_id, config, device, beta, proj_w, proj_b):
    sample_rate = config['Audio_features']['sample_rate']
    base_path = config['Dataset']['base_path']
    seed = config['Training']['seed']
    opleg_base_path = config['Dataset'].get('opleg_base_path')
    phase0_acc = None
    for pid in range(phase_id + 1):
        _, _, test_dict = build_phase_data(pid, base_path=base_path, seed=seed,
                                            opleg_base_path=opleg_base_path)
        stats = _evaluate_projected(model, test_dict, device, sample_rate, beta, proj_w, proj_b)
        logger.info(f'  Phase {pid} {INCREMENTAL_PHASES[pid]}: acc={stats["overall_acc"]:.4f}')
        if pid == 0:
            phase0_acc = stats['overall_acc']
    return phase0_acc


def analytic_trainer_cnn14mv2(model, config, device, ckpt_dir, phase0_ckpt):
    """Same closed-form mechanism as analytic_trainer, on the Cnn14-MobileV2
    backbone -- used to confirm the result isn't an artifact of PANNs'
    AudioSet pretraining, and to match AquaRecall's own backbone for a
    fair head-to-head comparison."""
    sample_rate = config['Audio_features']['sample_rate']
    seed = config['Training']['seed']
    base_path = config['Dataset']['base_path']
    opleg_base_path = config['Dataset'].get('opleg_base_path')
    ridge_lambda = config['Training'].get('ridge_lambda', 1.0)
    random_proj_dim = config['Training'].get('random_proj_dim', 8192)
    embed_dim = 1024  # Cnn14_mobilev2's fc1 output dimension

    logger.info(f'loading Cnn14-MobileV2 Phase-0 checkpoint: {phase0_ckpt}')
    model = load_ckpt(phase0_ckpt, model)
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.frontend.bn0.parameters():
        p.requires_grad = False

    gen = torch.Generator().manual_seed(42)
    proj_w = torch.empty(embed_dim, random_proj_dim)
    torch.nn.init.kaiming_normal_(proj_w, generator=gen)
    proj_w = proj_w.to(device)
    proj_b = torch.zeros(random_proj_dim, device=device)

    A = B = None
    results = {}

    for phase_id in range(len(INCREMENTAL_PHASES)):
        logger.info(f'\n{"=" * 60}\nPHASE {phase_id} START (Cnn14-MobileV2)\n{"=" * 60}')

        train_dict, _, _ = build_phase_data(phase_id, base_path=base_path, seed=seed,
                                             opleg_base_path=opleg_base_path)
        emb_raw, y = _extract_embeddings(model, train_dict, device, sample_rate)
        emb_proj = F.relu(emb_raw.to(device) @ proj_w + proj_b).cpu()

        A_new, B_new = compute_batch_stats(emb_proj, y)
        if A is None:
            A, B = A_new, B_new
        else:
            A = A + A_new
            B = B + B_new
        beta = solve_ridge(A, B, ridge_lambda)

        if phase_id == 0:
            _, _, test_dict0 = build_phase_data(0, base_path=base_path, seed=seed,
                                                 opleg_base_path=opleg_base_path)
            test_stats = _evaluate_projected(model, test_dict0, device, sample_rate, beta, proj_w, proj_b)
            results[0] = {'test': test_stats, 'forgetting': 0.0}
        else:
            all_test = build_all_phases_test_data(phase_id, base_path=base_path, seed=seed,
                                                   opleg_base_path=opleg_base_path)
            test_stats = _evaluate_projected(model, all_test, device, sample_rate, beta, proj_w, proj_b)
            test_stats['phase0_acc'] = _evaluate_by_fish_projected(model, phase_id, config, device, beta, proj_w, proj_b)
            results[phase_id] = {'test': test_stats}
            results[phase_id]['forgetting'] = compute_forgetting(results, phase_id)
            logger.info(f'Phase {phase_id} Forgetting: {results[phase_id]["forgetting"]:.4f}')

    logger.info('\n' + '=' * 60)
    logger.info('Training complete (Cnn14-MobileV2):')
    for pid, result in results.items():
        logger.info(f'Phase {pid} | Acc:{result["test"]["overall_acc"]:.4f} | '
                    f'Forgetting:{result["forgetting"]:.4f}')

    return model, results
