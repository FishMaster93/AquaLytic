"""Species-incremental data loading for the 7-phase fish feeding-intensity
dataset. Each phase introduces one or more new species; the intensity label
space (None/Weak/Medium/Strong) is shared and fixed across all phases.

Expected directory layout under ``base_path`` (see README for the dataset
link):

    audio_dataset/
    |-- <recording_session>/
    |   |-- <species_name>/
    |   |   |-- None/*.wav
    |   |   |-- Weak/*.wav
    |   |   |-- Medium/*.wav
    |   |   `-- Strong/*.wav
    `-- ...

Phase 6 (Oplegnathus_punctatus) ships as a separately-structured dataset
(see ``build_opleg_phase_data``) -- pass its root via ``opleg_base_path``.
"""
import glob
import os

import librosa
import numpy as np
import torch
from scipy.signal import resample
from torch.utils.data import Dataset

INCREMENTAL_PHASES = {
    0: ['Red_tilapia_2', 'Red_tilapia_3'],
    1: ['Tilapia'],
    2: ['Jade_perch'],
    3: ['Largemouth_bass'],
    4: ['Lotus_carp'],
    5: ['Sunfish'],
    6: ['Oplegnathus_punctatus'],
}

LABEL_MAP = {
    'None': 0,
    'Weak': 1,
    'Medium': 2,
    'Strong': 3,
}

# Phase 6 uses a separately-structured dataset -- see build_opleg_phase_data.
OPLEG_EXCLUDED = {'AM_5', 'PM_5', 'AM_15', 'PM_15'}
OPLEG_LABEL_MAP = {'none': 0, 'weak': 1, 'medium': 2, 'strong': 3}


def load_audio(path, sr=64000):
    try:
        y, _ = librosa.load(path, sr=None)
        y = resample(y, num=sr * 2)
        return y.astype(np.float32)
    except Exception as e:
        print(f'[Warning] failed to read: {path}, {e}')
        return np.zeros(sr * 2, dtype=np.float32)


def get_wav_name_for_fish(fish_name, split, base_path):
    audio = []
    if not os.path.exists(base_path):
        raise FileNotFoundError(f'path does not exist: {base_path}')
    for date_dir in sorted(os.listdir(base_path)):
        date_path = os.path.join(base_path, date_dir)
        if not os.path.isdir(date_path):
            continue
        wav_pattern = os.path.join(date_path, fish_name, split, '*.wav')
        audio.extend(glob.glob(wav_pattern))
    return audio


def build_opleg_phase_data(opleg_base_path, seed=42):
    """Phase 6 (Oplegnathus_punctatus) loader: layout is
    {date}/{session}/{none|weak|medium|strong}/*.wav, excluding a few
    low-sample sessions (OPLEG_EXCLUDED)."""
    random_state = np.random.RandomState(seed)
    by_label = {0: [], 1: [], 2: [], 3: []}

    for date_dir in sorted(os.listdir(opleg_base_path)):
        date_path = os.path.join(opleg_base_path, date_dir)
        if not os.path.isdir(date_path):
            continue
        for subfolder in sorted(os.listdir(date_path)):
            if subfolder in OPLEG_EXCLUDED:
                continue
            sub_path = os.path.join(date_path, subfolder)
            if not os.path.isdir(sub_path):
                continue
            for intensity_name, label in OPLEG_LABEL_MAP.items():
                intensity_path = os.path.join(sub_path, intensity_name)
                if not os.path.isdir(intensity_path):
                    continue
                wavs = glob.glob(os.path.join(intensity_path, '*.wav'))
                by_label[label].extend(wavs)

    train_dict, val_dict, test_dict = [], [], []
    for label, wav_list in sorted(by_label.items()):
        random_state.shuffle(wav_list)
        total = len(wav_list)
        train_end = int(total * 0.8)
        val_end = train_end + int(total * 0.1)
        for wav in wav_list[:train_end]:
            train_dict.append([wav, label])
        for wav in wav_list[train_end:val_end]:
            val_dict.append([wav, label])
        for wav in wav_list[val_end:]:
            test_dict.append([wav, label])

    random_state.shuffle(train_dict)
    return train_dict, val_dict, test_dict


def build_phase_data(phase_id, base_path, seed=42, opleg_base_path=None):
    """Load one phase's species, split 80/10/10 per (species, intensity)
    bucket. Phase 6 is dispatched to build_opleg_phase_data automatically
    if opleg_base_path is given."""
    if INCREMENTAL_PHASES[phase_id] == ['Oplegnathus_punctatus']:
        if opleg_base_path is None:
            raise ValueError('Phase 6 requires opleg_base_path')
        return build_opleg_phase_data(opleg_base_path, seed=seed)

    random_state = np.random.RandomState(seed)
    fish_list = INCREMENTAL_PHASES[phase_id]
    train_dict, val_dict, test_dict = [], [], []

    for fish in fish_list:
        for split_name, intensity in LABEL_MAP.items():
            wav_list = get_wav_name_for_fish(fish, split_name, base_path)
            if len(wav_list) == 0:
                print(f'[Warning] not found: {fish}/{split_name}')
                continue

            random_state.shuffle(wav_list)
            total = len(wav_list)
            train_end = int(total * 0.8)
            val_end = train_end + int(total * 0.1)

            for wav in wav_list[:train_end]:
                train_dict.append([wav, intensity])
            for wav in wav_list[train_end:val_end]:
                val_dict.append([wav, intensity])
            for wav in wav_list[val_end:]:
                test_dict.append([wav, intensity])

    random_state.shuffle(train_dict)
    return train_dict, val_dict, test_dict


def build_all_phases_test_data(up_to_phase, base_path, seed=42, opleg_base_path=None):
    all_test = []
    for phase_id in range(up_to_phase + 1):
        _, _, test_dict = build_phase_data(phase_id, base_path=base_path, seed=seed,
                                            opleg_base_path=opleg_base_path)
        all_test.extend(test_dict)
    return all_test


class FishIncrementalDataset(Dataset):
    def __init__(self, data_dict, exemplar_dict=None, sample_rate=64000):
        self.sample_rate = sample_rate
        self.n_new = len(data_dict)
        self.n_exemplar = len(exemplar_dict) if exemplar_dict else 0
        if exemplar_dict and len(exemplar_dict) > 0:
            self.data_dict = data_dict + exemplar_dict
        else:
            self.data_dict = data_dict

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index):
        wav_path, label = self.data_dict[index]
        wav = load_audio(wav_path, sr=self.sample_rate)
        return {
            'audio_name': wav_path,
            'waveform': wav,
            'target': label,
            'is_exemplar': int(index >= self.n_new),
        }


class FishTestDataset(Dataset):
    def __init__(self, data_dict, sample_rate=64000):
        self.data_dict = data_dict
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index):
        wav_path, label = self.data_dict[index]
        wav = load_audio(wav_path, sr=self.sample_rate)
        return {
            'audio_name': wav_path,
            'waveform': wav,
            'target': label,
            'is_exemplar': 0,
        }


def collate_fn(batch):
    return {
        'audio_name': [d['audio_name'] for d in batch],
        'waveform': torch.FloatTensor(np.array([d['waveform'] for d in batch])),
        'target': torch.LongTensor([d['target'] for d in batch]),
        'is_exemplar': torch.LongTensor([d['is_exemplar'] for d in batch]),
    }


def build_phase_data_with_species(phase_id, base_path, seed=42, opleg_base_path=None):
    """Same split as build_phase_data (identical seed/order), but each item
    is tagged (wav_path, label, species_name) instead of (wav_path, label) --
    needed when a phase introduces more than one species at once (Phase 0:
    ['Red_tilapia_2', 'Red_tilapia_3']) and a downstream consumer (e.g. the
    open-set detector's species head) needs per-sample species identity."""
    fish_list = INCREMENTAL_PHASES[phase_id]
    if fish_list == ['Oplegnathus_punctatus']:
        if opleg_base_path is None:
            raise ValueError('Phase 6 requires opleg_base_path')
        train, val, test = build_opleg_phase_data(opleg_base_path, seed=seed)
        name = 'Oplegnathus_punctatus'
        tag = lambda lst: [(wav, label, name) for wav, label in lst]
        return tag(train), tag(val), tag(test)

    random_state = np.random.RandomState(seed)
    train, val, test = [], [], []
    for fish in fish_list:
        for split_name, intensity in LABEL_MAP.items():
            wav_list = get_wav_name_for_fish(fish, split_name, base_path)
            if len(wav_list) == 0:
                continue
            random_state.shuffle(wav_list)
            total = len(wav_list)
            train_end = int(total * 0.8)
            val_end = train_end + int(total * 0.1)
            for wav in wav_list[:train_end]:
                train.append((wav, intensity, fish))
            for wav in wav_list[train_end:val_end]:
                val.append((wav, intensity, fish))
            for wav in wav_list[val_end:]:
                test.append((wav, intensity, fish))
    random_state.shuffle(train)
    return train, val, test


class SpeciesTaggedDataset(Dataset):
    """Like FishTestDataset, but items are (wav_path, label, species_name)
    triples -- see build_phase_data_with_species."""

    def __init__(self, items, sample_rate=64000):
        self.items = items
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        wav_path, label, species = self.items[i]
        wav = load_audio(wav_path, sr=self.sample_rate)
        return {'waveform': wav, 'target': label, 'species': species}


def collate_fn_with_species(batch):
    return {
        'waveform': torch.FloatTensor(np.array([b['waveform'] for b in batch])),
        'target': torch.LongTensor([b['target'] for b in batch]),
        'species': [b['species'] for b in batch],
    }
