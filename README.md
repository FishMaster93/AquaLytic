# AquaLytic

**AquaLytic: closed-form species-incremental learning for acoustic fish
feeding intensity recognition in aquaculture**

Ying Xiao, Tan Wang, Liang Wang, Tomas Norton, Meng Cui&dagger;
(&dagger; corresponding author)

*Information Processing in Agriculture*

A frozen AudioSet-pretrained (or from-scratch) CNN backbone feeds a fixed,
high-dimensional random-projection buffer, on top of which a CORAL ordinal
head is updated with a **closed-form ridge-regression solve** every time a
new species is introduced -- no replay buffer, no distillation, no
gradient steps after Phase 0. Because the per-phase statistics
`A = sum x x^T`, `B = sum x y^T` are simply additive across phases, the
incrementally-accumulated solution is mathematically identical to solving
the regression once on the union of all phases' real data: the method is
exact, not approximate, and **provably order-invariant** (matrix addition
commutes).

On top of that mechanism, the ridge solve's own posterior variance gives a
second capability for free: detecting when a sample belongs to a species the
model has never seen, by fusing two signals that turn out to fail on
different phases (so the fusion is rarely worse than either alone).

## Repository layout

```
models/            backbones (PANNs-Cnn10, Cnn14-MobileV2) + shared audio frontend
dataset/           species-incremental data loading
core/
  analytic_head.py       the ridge-regression math (A/B accumulation, closed-form solve)
  analytic_trainer.py    the closed-form CIL loop, for both backbones
  incremental_eval.py    CORAL loss/decoding, evaluation, Phase-0 gradient pretraining
  open_set_detection.py  unseen-species detection (q(x) + species-head confidence + fusion)
config/            example YAML configs
main_phase0_pretrain.py        Phase 0: gradient fine-tune backbone + temporary CORAL head
main_analytic_cnn10.py         Phases 1-N, PANNs-Cnn10 backbone
main_analytic_cnn14mv2.py      Phases 1-N, Cnn14-MobileV2 backbone (backbone-matched comparison)
main_open_set_detection.py     unseen-species detection on top of the Cnn14-MobileV2 pipeline
```

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.10+, PyTorch 2.x, CUDA 12.x. A GPU is recommended for
Phase 0 pretraining and embedding extraction; the closed-form solves
themselves are cheap enough to run on CPU (a single 8193x8193 ridge solve
takes well under a second).

## Dataset

Acoustic recordings of 7 fish species, each labeled at 4 feeding-intensity
levels (None / Weak / Medium / Strong), introduced incrementally one (or two,
for Phase 0) species at a time:

Phase 0: Red_tilapia_2, Red_tilapia_3 &nbsp;&middot;&nbsp;
Phase 1: Tilapia &nbsp;&middot;&nbsp; Phase 2: Jade_perch &nbsp;&middot;&nbsp;
Phase 3: Largemouth_bass &nbsp;&middot;&nbsp; Phase 4: Lotus_carp &nbsp;&middot;&nbsp;
Phase 5: Sunfish &nbsp;&middot;&nbsp; Phase 6: Oplegnathus_punctatus

Dataset: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21888149.svg)](https://doi.org/10.5281/zenodo.21888149)

Expected layout under `base_path` (Phase 6 uses a separate directory --
see `opleg_base_path` in the config files and `build_opleg_phase_data` in
`dataset/fish_incremental_dataset.py` for its exact structure):

```
audio_dataset/
├── <recording_session>/
│   ├── <species_name>/
│   │   ├── None/*.wav
│   │   ├── Weak/*.wav
│   │   ├── Medium/*.wav
│   │   └── Strong/*.wav
│   └── ...
└── ...
```

## Usage

**1. Phase 0 pretraining** (the only gradient-trained step):

```bash
python main_phase0_pretrain.py --config config/phase0_pretrain.yaml
```

Produces `workspace/phase0_pretrain/save_models/phase0_best.pt`. Point
`phase0_ckpt` in the analytic configs at this file (or at your own Phase-0
checkpoint, as long as it was trained with the same frontend/backbone and a
CORAL head on the first species).

**2. Closed-form incremental learning**, either backbone:

```bash
python main_analytic_cnn10.py --config config/analytic_cnn10.yaml
python main_analytic_cnn14mv2.py --config config/analytic_cnn14mv2.yaml
```

Each phase prints overall accuracy and forgetting (accuracy drift on
Phase 0's own species, re-measured after every later phase).

**3. Unseen-species detection**:

```bash
python main_open_set_detection.py --config config/analytic_cnn14mv2.yaml
```

At each phase, scores the about-to-be-introduced species as "novel" against
everything accumulated so far, reports AUROC/AUPR for each of the two
signals and their fusion, then calibrates a single deployment threshold
(Youden's J on the pooled ROC curve) and reports sensitivity/specificity/
precision/F1 at that fixed threshold, phase by phase.

## Method summary

| Component | What it is |
|---|---|
| Backbone | Frozen after Phase 0 (conv blocks + fc1 + frontend BN) |
| Random-projection buffer | `Linear(d, D)`, `W ~ N(0, 2/d)` (Kaiming, fan_in), frozen at init |
| Intensity head | CORAL ordinal thresholds, closed-form ridge: `beta = (A + lambda I)^-1 B` |
| Species head | A second, independent closed-form solve (own A_sp/B_sp) on one-hot species targets |
| q(x) | Ridge posterior variance `x^T A^-1 x` -- reuses the intensity head's A for free |
| Novelty score | `z(q(x)) + z(1 - species confidence)` |

## Citation

If you use this code, please cite:

```bibtex
@article{xiao2026aqualytic,
  title   = {AquaLytic: closed-form species-incremental learning for acoustic
             fish feeding intensity recognition in aquaculture},
  author  = {Xiao, Ying and Wang, Tan and Wang, Liang and Norton, Tomas and Cui, Meng},
  journal = {Information Processing in Agriculture},
  year    = {2026}
}
```

(Update with volume/pages/DOI once the paper is published.)

## License

MIT -- see [LICENSE](LICENSE).
