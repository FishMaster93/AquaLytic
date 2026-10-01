"""Unseen-species (open-set) detection: two free uncertainty signals derived
from the closed-form ridge solve, fused to flag species the model has never
been trained on.

Signal 1 -- q(x): the ridge posterior predictive variance x^T A^-1 x, reusing
the exact same A matrix the intensity head already accumulates (zero extra
cost). Small q(x) means x sits in a region covered by lots of real training
data; large q(x) means it's far from everything seen so far.

Signal 2 -- species-head confidence: a second, independent closed-form solve
on the same embeddings against one-hot species targets (not the intensity
head's A -- a separate A_sp/B_sp pair), giving max_k sigmoid(x^T beta_sp_k).

Fusion: z-score each signal (they're on different scales) and sum:
    score(x) = z(q(x)) + z(1 - confidence(x))
Empirically the two signals fail on different phases (see the paper's
complementarity analysis), so the fused score is rarely the worse of the two
on any given phase.
"""
import logging

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from torch.utils.data import DataLoader

from dataset.fish_incremental_dataset import (
    INCREMENTAL_PHASES, SpeciesTaggedDataset, collate_fn_with_species,
)
from core.analytic_head import ridge_stats, solve_ridge, predictive_uncertainty

logger = logging.getLogger()


@torch.no_grad()
def extract_embeddings_with_species(model, species_tagged_items, device, sample_rate, batch_size=256):
    """Like core.analytic_trainer._extract_embeddings, but for
    build_phase_data_with_species's (wav, label, species) triples -- returns
    (embeddings, labels, species_names) with species_names aligned row-for-row."""
    loader = DataLoader(SpeciesTaggedDataset(species_tagged_items, sample_rate), batch_size=batch_size,
                         shuffle=False, num_workers=8, collate_fn=collate_fn_with_species)
    model.eval()
    embs, labels, species = [], [], []
    for batch in loader:
        out = model(batch['waveform'].to(device))
        embs.append(out['embedding'].cpu())
        labels.append(batch['target'])
        species.extend(batch['species'])
    return torch.cat(embs, dim=0), torch.cat(labels, dim=0), species


def build_species_index(incremental_phases=INCREMENTAL_PHASES):
    """Species order follows phase order; Phase 0 contributes as many
    entries as it has species (e.g. 2 for ['Red_tilapia_2', 'Red_tilapia_3']),
    so the species head has >=2 real classes to discriminate from the very
    first phase."""
    all_species = []
    for pid in sorted(incremental_phases.keys()):
        all_species.extend(incremental_phases[pid])
    return {s: i for i, s in enumerate(all_species)}, len(all_species)


def update_species_head(A_sp, B_sp, embeddings, species_names, species_to_idx, n_species, ridge_lambda=1.0):
    """Accumulate this phase's contribution to the species head and re-solve.
    `embeddings` and `species_names` must be aligned (same order, one species
    name per embedding row)."""
    sp_idx = torch.tensor([species_to_idx[s] for s in species_names], dtype=torch.long)
    n = embeddings.shape[0]
    ones = torch.ones(n, 1)
    x_aug = torch.cat([embeddings, ones], dim=1)
    y_sp = F.one_hot(sp_idx, num_classes=n_species).float()
    A_new, B_new = ridge_stats(x_aug, y_sp)
    A_sp = A_new if A_sp is None else A_sp + A_new
    B_sp = B_new if B_sp is None else B_sp + B_new
    beta_sp = solve_ridge(A_sp, B_sp, ridge_lambda)
    return A_sp, B_sp, beta_sp


def score_novelty(embeddings, A_inv, beta_species, device):
    """Returns (q, confidence) for a batch of embeddings, using the
    pre-update statistics (i.e. computed BEFORE training on the phase being
    scored, so the "novel" class is genuinely unseen by both heads)."""
    n = embeddings.shape[0]
    ones = torch.ones(n, 1)
    x = torch.cat([embeddings, ones], dim=1).to(device)
    q = predictive_uncertainty(x[:, :-1], A_inv).cpu().numpy()
    conf = torch.sigmoid(x @ beta_species.to(device)).max(dim=1).values.cpu().numpy()
    return q, conf


def fuse_scores(q_all, conf_all):
    """z-score sum; returns a score where HIGHER = more likely novel."""
    z_q = (q_all - q_all.mean()) / (q_all.std() + 1e-8)
    neg_conf = -conf_all
    z_negconf = (neg_conf - neg_conf.mean()) / (neg_conf.std() + 1e-8)
    return z_q + z_negconf


def evaluate_phase_detection(q_novel, conf_novel, q_seen, conf_seen):
    """AUROC/AUPR for q(x), species-confidence, and the fused score on one
    phase's novel-vs-seen split. Returns a dict plus the raw labels/scores
    (needed for ROC curves or threshold calibration downstream)."""
    labels = np.concatenate([np.ones(len(q_novel)), np.zeros(len(q_seen))])
    q_all = np.concatenate([q_novel, q_seen])
    conf_all = np.concatenate([conf_novel, conf_seen])
    fused = fuse_scores(q_all, conf_all)

    return {
        'labels': labels, 'q_all': q_all, 'conf_all': conf_all, 'fused': fused,
        'auroc_q': roc_auc_score(labels, q_all),
        'aupr_q': average_precision_score(labels, q_all),
        'auroc_conf': roc_auc_score(labels, -conf_all),
        'aupr_conf': average_precision_score(labels, -conf_all),
        'auroc_fused': roc_auc_score(labels, fused),
        'aupr_fused': average_precision_score(labels, fused),
    }


def youden_threshold(labels, scores):
    """Youden's J statistic (argmax TPR - FPR) optimal operating point --
    use this to turn the AUROC-level evaluation above into a single
    deployable decision threshold, then report sensitivity/specificity/
    precision/F1 at that point (see threshold_metrics)."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j = tpr - fpr
    best = int(np.argmax(j))
    return float(thresholds[best]), float(tpr[best]), float(fpr[best])


def threshold_metrics(labels, scores, threshold):
    """Confusion-matrix-derived sensitivity/specificity/precision/F1 at a
    fixed, already-chosen threshold (positive class = novel/unseen, label=1).
    Apply the SAME threshold across phases to see how a single deployed
    cutoff behaves as the seen/novel class balance shifts over time."""
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    spec = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
    prec = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else float('nan')
    return {'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp,
            'sensitivity': sens, 'specificity': spec, 'precision': prec, 'f1': f1}
