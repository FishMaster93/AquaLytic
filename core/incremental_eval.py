"""CORAL ordinal loss/decoding and shared evaluation utilities for Phase 0
(gradient pretraining) and all subsequent closed-form phases."""
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.fish_incremental_dataset import (
    INCREMENTAL_PHASES, build_phase_data, FishIncrementalDataset,
    FishTestDataset, collate_fn,
)

logger = logging.getLogger()


def coral_loss_intensity(logits, targets, num_classes=4):
    """Standard CORAL rank-consistent logistic loss (Cao et al. 2020): each
    of the num_classes-1 thresholds is an independent binary classifier
    ("is intensity > k?"), trained with BCE. Used only for Phase 0's
    gradient-trained temporary head -- the closed-form phases use a
    different margin-regression reformulation (see core/analytic_head.py)
    so the ridge solve stays linear in the targets."""
    batch_size = logits.size(0)
    ordinal_targets = torch.zeros(batch_size, num_classes - 1).to(logits.device)
    intensity_targets = targets % num_classes
    for i in range(num_classes - 1):
        ordinal_targets[:, i] = (intensity_targets > i).float()
    return F.binary_cross_entropy_with_logits(logits, ordinal_targets)


def coral_predict_intensity(logits):
    """CORAL decoding: sum the sigmoid threshold probabilities and round."""
    probs = torch.sigmoid(logits)
    pred = probs.sum(dim=-1).round().long()
    return pred.clamp(0, 3)


def save_ckpt(path, model, optimizer, acc, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'acc': float(acc),
        'epoch': int(epoch),
    }, path)
    logger.info(f'  saved: {path} (acc={acc:.4f})')


def load_ckpt(path, model):
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    logger.info(f'  loaded: {path} (acc={ckpt["acc"]:.4f})')
    return model


def evaluate(model, data_loader, device, phase_id, split='val'):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in data_loader:
            waveform = batch['waveform'].to(device)
            target = batch['target'].to(device)
            output = model(waveform)
            logits = output['clipwise_output']
            pred = coral_predict_intensity(logits)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    intensity_acc = ((all_preds % 4) == (all_targets % 4)).mean()

    class_names = {0: 'None', 1: 'Weak', 2: 'Medium', 3: 'Strong'}
    class_acc = {}
    for label, name in class_names.items():
        mask = (all_targets % 4) == label
        class_acc[name] = float(((all_preds % 4)[mask] == (all_targets % 4)[mask]).mean()) \
            if mask.sum() > 0 else 0.0

    logger.info(f'  [{split}] Phase {phase_id} Overall Acc: {intensity_acc:.4f}')
    for name, acc in class_acc.items():
        logger.info(f'    {name}: {acc:.4f}')

    return {'overall_acc': intensity_acc, 'class_acc': class_acc,
            'preds': all_preds, 'targets': all_targets}


def evaluate_by_fish(model, phase_id, config, device):
    """Per-species breakdown; also returns Phase 0's own accuracy, used by
    compute_forgetting."""
    sample_rate = config['Audio_features']['sample_rate']
    batch_size = config['Training']['Batch_size']
    seed = config['Training']['seed']
    base_path = config['Dataset']['base_path']
    opleg_base_path = config['Dataset'].get('opleg_base_path')

    logger.info('per-species evaluation:')
    phase0_acc = None

    for pid in range(phase_id + 1):
        fish_list = INCREMENTAL_PHASES[pid]
        _, _, test_dict = build_phase_data(pid, base_path=base_path, seed=seed,
                                            opleg_base_path=opleg_base_path)
        test_loader = DataLoader(FishTestDataset(test_dict, sample_rate), batch_size=batch_size,
                                  shuffle=False, num_workers=8, collate_fn=collate_fn)
        stats = evaluate(model, test_loader, device, phase_id, split='test')
        logger.info(f'  Phase {pid} {fish_list}: acc={stats["overall_acc"]:.4f}')
        if pid == 0:
            phase0_acc = stats['overall_acc']

    return phase0_acc


def compute_forgetting(results, current_phase):
    """Forgetting = Phase 0's own accuracy at the moment it was trained,
    minus Phase 0's accuracy re-measured after the current phase."""
    if current_phase == 0:
        return 0.0
    phase0_acc = results[0]['test']['overall_acc']
    current_acc = results[current_phase]['test'].get(
        'phase0_acc', results[current_phase]['test']['overall_acc'])
    return phase0_acc - current_acc


def train_phase0(model, optimizer, sample_rate, batch_size, max_epoch,
                  seed, base_path, device, ckpt_dir, opleg_base_path=None):
    """Gradient pretraining of Phase 0: the backbone is fine-tuned on the
    first species together with a temporary CORAL ordinal head (standard
    logistic rank-consistency loss). After this, the backbone is frozen and
    the temporary head discarded -- every later phase uses the closed-form
    ridge update instead (see core/analytic_trainer.py)."""
    logger.info('=' * 60)
    logger.info('Phase 0: Initial training')
    logger.info('=' * 60)

    train_dict, val_dict, test_dict = build_phase_data(
        0, base_path=base_path, seed=seed, opleg_base_path=opleg_base_path)
    train_dataset = FishIncrementalDataset(train_dict, None, sample_rate)
    val_dataset = FishTestDataset(val_dict, sample_rate)
    test_dataset = FishTestDataset(test_dict, sample_rate)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=8, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=8, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=8, collate_fn=collate_fn)

    best_acc = 0.0

    for epoch in range(max_epoch):
        model.train()
        mean_loss = 0.0

        for batch in tqdm(train_loader, desc=f'Phase0 Epoch {epoch}/{max_epoch}'):
            waveform = batch['waveform'].to(device)
            target = batch['target'].to(device)
            output = model(waveform)
            logits = output['clipwise_output']
            loss = coral_loss_intensity(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mean_loss += loss.item()

        logger.info(f'Phase0 Epoch {epoch} loss:{mean_loss / len(train_loader):.4f}')

        if epoch % 10 == 0:
            val_stats = evaluate(model, val_loader, device, 0, split='val')
            if val_stats['overall_acc'] > best_acc:
                best_acc = val_stats['overall_acc']
                save_ckpt(os.path.join(ckpt_dir, 'phase0_best.pt'), model, optimizer, best_acc, epoch)
            model.train()

    model = load_ckpt(os.path.join(ckpt_dir, 'phase0_best.pt'), model)
    test_stats = evaluate(model, test_loader, device, 0, split='test')
    logger.info(f'Phase 0 Test Acc: {test_stats["overall_acc"]:.4f}')
    return model, train_dict, test_stats
