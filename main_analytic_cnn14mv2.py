# Closed-form analytic CIL, Cnn14-MobileV2 backbone (from scratch, no
# AudioSet pretraining). This is the backbone-matched configuration used in
# the paper's main results table, to compare head-to-head against
# AquaRecall and the gradient baselines on identical architecture.
import warnings
warnings.filterwarnings('ignore')
import os
import time
import logging
import argparse

import torch
from omegaconf import OmegaConf

from models.audio_model import Audio_Frontend, AudioModel
from models.cnn14_mobilev2 import Cnn14_mobilev2
from core.analytic_trainer import analytic_trainer_cnn14mv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/analytic_cnn14mv2.yaml')
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

    frontend = Audio_Frontend(**audio_features)
    backbone = Cnn14_mobilev2(classes_num=Training['classes_num'])
    model = AudioModel(frontend=frontend, backbone=backbone).to(device)

    model, results = analytic_trainer_cnn14mv2(model, config, device, ckpt_dir, phase0_ckpt)
