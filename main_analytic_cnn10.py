# Closed-form analytic CIL, PANNs-Cnn10 backbone (AudioSet-pretrained).
# Reuses the phase0_best.pt produced by main_phase0_pretrain.py; from Phase 1
# onward the backbone is frozen and every update is a single ridge solve.
import warnings
warnings.filterwarnings('ignore')
import os
import time
import logging
import argparse

import torch
from omegaconf import OmegaConf

from models.audio_model import Audio_Frontend, AudioModel
from models.panns_cnn10 import PANNS_Cnn10
from core.analytic_trainer import analytic_trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/analytic_cnn10.yaml')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    workspace = config['Workspace']
    exp_name = config['Exp_name']
    audio_features = config['Audio_features']
    Training = config['Training']
    phase0_ckpt = Training['phase0_ckpt']

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

    random_proj_dim = Training.get('random_proj_dim', None)
    frontend = Audio_Frontend(**audio_features)
    backbone = PANNS_Cnn10(classes_num=Training['classes_num'], random_proj_dim=random_proj_dim)
    model = AudioModel(frontend=frontend, backbone=backbone).to(device)

    model, results = analytic_trainer(model, config, device, ckpt_dir, phase0_ckpt)
