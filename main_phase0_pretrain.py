# Phase 0 pretraining: gradient fine-tune the backbone + a temporary CORAL
# ordinal head on the first species (Red_tilapia). This is the one and only
# gradient-trained step in the whole pipeline; every later phase in
# main_analytic_*.py uses the closed-form ridge update instead.
import warnings
warnings.filterwarnings('ignore')
import os
import time
import logging
import argparse

import torch
import torch.optim as optim
from omegaconf import OmegaConf

from models.audio_model import Audio_Frontend, AudioModel
from models.panns_cnn10 import PANNS_Cnn10, load_pretrained_cnn10
from models.cnn14_mobilev2 import Cnn14_mobilev2
from core.incremental_eval import train_phase0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/phase0_pretrain.yaml')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    workspace = config['Workspace']
    exp_name = config['Exp_name']
    audio_features = config['Audio_features']
    Training = config['Training']

    ckpt_dir = os.path.join(workspace, exp_name, 'save_models')
    log_dir = os.path.join(workspace, exp_name, 'logs')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f'{exp_name}-{time.time():.0f}.log')),
            logging.StreamHandler(),
        ],
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    frontend = Audio_Frontend(**audio_features)
    backbone_name = config.get('Model', 'PANNS_Cnn10')
    if backbone_name == 'PANNS_Cnn10':
        backbone = PANNS_Cnn10(classes_num=Training['classes_num'])
        model = AudioModel(frontend=frontend, backbone=backbone).to(device)
        if Training.get('pretrained_cnn10_ckpt'):
            loaded_keys = load_pretrained_cnn10(model.backbone, Training['pretrained_cnn10_ckpt'])
            logging.info(f'loaded {len(loaded_keys)} AudioSet-pretrained tensors into the backbone')
    elif backbone_name == 'Cnn14_mobilev2':
        backbone = Cnn14_mobilev2(classes_num=Training['classes_num'])
        model = AudioModel(frontend=frontend, backbone=backbone).to(device)
    else:
        raise ValueError(f'unknown Model: {backbone_name}')

    optimizer = optim.Adam(model.parameters(), lr=Training.get('lr', 1e-4))

    model, train_dict, test_stats = train_phase0(
        model, optimizer,
        sample_rate=audio_features['sample_rate'],
        batch_size=Training['Batch_size'],
        max_epoch=Training['max_epoch'],
        seed=Training['seed'],
        base_path=config['Dataset']['base_path'],
        device=device,
        ckpt_dir=ckpt_dir,
        opleg_base_path=config['Dataset'].get('opleg_base_path'),
    )
