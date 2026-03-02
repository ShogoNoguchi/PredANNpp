


"""
[Overall Premise]
- split_seed is fixed to 42, and self.train_test_splitting is set to "random_split_30s"
  for reproducible train/test splitting in 30-second chunks.
※Maintaining and managing this code is critically important, as information leakage between
  train/test sets directly constitutes research misconduct.

- EEG data covers 30s × 8 chunks (= 240 seconds) or more, sufficient for SW_valid mode.

[Source Data Structure]
🎵 Audio Data
- Storage path: <NMEDT_BASE_DIR>/audio/
- Filenames: 21.wav ~ 30.wav (10 songs total)
- Format: WAV, monaural
- Sampling rate: **44,100Hz** (no resampling performed)
- Length: approximately 270 ~ 297 seconds
- Amplitude: maximum value is typically 1.000 (watch for clipping)


🧠 EEG Data
- Storage path: <NMEDT_BASE_DIR>/DS_EEG_pkl/
- Filename format: {subject}_{song}_{trial=1}.pkl (e.g., 10_21_1.pkl)
- trial: **fixed to 1**
- Content: shape = [128, T], T varies by song (approx. 271 ~ 298 seconds at 125Hz). Fixed at 128 channels
- Sampling rate: **125Hz**
- Time series length: approximately 33874 ~ 37249 samples (= approx. 271 ~ 298 seconds)
- Format: NumPy array (.pkl format)
- Note: Perfect 1:1 correspondence with audio (song_id correspondence guaranteed by get_file_list())

Directory structure (example):
<NMEDT_BASE_DIR>/
├── audio
│   ├── 21.wav
│   ├── 22.wav
│   ├── 23.wav
│   ├── 24.wav
│   ├── 25.wav
│   ├── 26.wav
│   ├── 27.wav
│   ├── 28.wav
│   ├── 29.wav
│   └── 30.wav
└── DS_EEG_pkl
    ├── 2_21_1.pkl ~ 2_30_1.pkl
    ├── 3_21_1.pkl ~ 3_30_1.pkl
    ├── 4_21_1.pkl ~ 4_30_1.pkl
    ├── 5_21_1.pkl ~ 5_30_1.pkl
    ├── 6_21_1.pkl ~ 6_30_1.pkl
    ├── 7_21_1.pkl ~ 7_30_1.pkl
    ├── 8_21_1.pkl ~ 8_30_1.pkl
    ├── 9_21_1.pkl ~ 9_30_1.pkl
    ├── 10_21_1.pkl ~ 10_30_1.pkl
    ├── 11_21_1.pkl ~ 11_30_1.pkl
    ├── 12_21_1.pkl ~ 12_30_1.pkl
    ├── 13_21_1.pkl ~ 13_30_1.pkl
    ├── 14_21_1.pkl ~ 14_30_1.pkl
    ├── 15_21_1.pkl ~ 15_30_1.pkl
    ├── 16_21_1.pkl ~ 16_30_1.pkl
    ├── 17_21_1.pkl ~ 17_30_1.pkl
    ├── 19_21_1.pkl ~ 19_30_1.pkl
    ├── 20_21_1.pkl ~ 20_30_1.pkl
    ├── 21_21_1.pkl ~ 21_30_1.pkl
    └── 23_21_1.pkl ~ 23_30_1.pkl

😲Surprisal (discretized using equal-frequency binning without clipping edges to avoid information loss)
Storage path: <NMEDT_BASE_DIR>/NoClip_Discreat_K1Surprisal

[Filename Convention]
├── {song_id}_chunk{chunk_id}.npy
    └─ song_id: 21 ~ 30 (10 songs total)
    └─ chunk_id: 0 ~ 7 (8 chunks of 30s each per song)
[File Count]
- Total: 10 songs × 8 chunks = **80 files**
[Data Specification]
- Content      : Surprisal sequence for one chunk (30 seconds), quantized
- Data type    : `np.ndarray(dtype=np.uint8)`
- shape        : `(1500,)`
- Time resolution: **20ms** (= 50 samples per second → 30s × 50 = 1500 points)
- Value range  : [0, 127] (Q=128 quantization levels)

Directory structure:
NoClip_Discreat_K1Surprisal/
├── 21_chunk0.npy ~ 21_chunk7.npy
├── 22_chunk0.npy ~ 22_chunk7.npy
├── 23_chunk0.npy ~ 23_chunk7.npy
├── 24_chunk0.npy ~ 24_chunk7.npy
├── 25_chunk0.npy ~ 25_chunk7.npy
├── 26_chunk0.npy ~ 26_chunk7.npy
├── 27_chunk0.npy ~ 27_chunk7.npy
├── 28_chunk0.npy ~ 28_chunk7.npy
├── 29_chunk0.npy ~ 29_chunk7.npy
├── 30_chunk0.npy ~ 30_chunk7.npy
├── datasheet/
│   ├── bin_table.xlsx
│   └── histograms.png
├── edges.json
└── edges.pkl

[Note]
- The segmentation and naming conventions of these Surprisal files are consistent with the
  song segmentation rules defined in the get_30s_file() function in this code, making
  integration easier later.

🎧MuQ Discretized Embeddings
Storage path: <NMEDT_BASE_DIR>/MuQ_Discreat_K128

[Filename Convention]
├── {song_id}_chunk{chunk_id}.npy
    └─ song_id: 21 ~ 30 (10 songs total)
    └─ chunk_id: 0 ~ 7 (8 chunks of 30s each per song)
[File Count]
- Total: 10 songs × 8 chunks = **80 files**
[Data Specification]
- Content      : MuQ embedding sequence for one chunk (30 seconds), discretized with K-means
- Data type    : `np.ndarray(dtype=np.uint8)`
- shape        : `(750,)`
- Time resolution: **40ms** (= 25 samples per second → 30s × 25 = 750 points)
- Value range  : [0, 127] (K=128)

Directory structure:
MuQ_Discreat_K128/
├── 21_chunk0.npy ~ 21_chunk7.npy
├── 22_chunk0.npy ~ 22_chunk7.npy
├── 23_chunk0.npy ~ 23_chunk7.npy
├── 24_chunk0.npy ~ 24_chunk7.npy
├── 25_chunk0.npy ~ 25_chunk7.npy
├── 26_chunk0.npy ~ 26_chunk7.npy
├── 27_chunk0.npy ~ 27_chunk7.npy
├── 28_chunk0.npy ~ 28_chunk7.npy
├── 29_chunk0.npy ~ 29_chunk7.npy
├── 30_chunk0.npy ~ 30_chunk7.npy

[Note]
- The segmentation and naming conventions of these MuQ discretized embeddings are consistent
  with the song segmentation rules defined in the get_30s_file() function in this code,
  similar to Surprisal, making integration easier later.

🤔Entropy (discretized using equal-frequency binning without clipping edges to avoid information loss)
Storage path: <NMEDT_BASE_DIR>/Entropy_k1_Q128

[Filename Convention]
├── {song_id}_chunk{chunk_id}.npy
    └─ song_id: 21 ~ 30 (10 songs total)
    └─ chunk_id: 0 ~ 7 (8 chunks of 30s each per song)
[File Count]
- Total: 10 songs × 8 chunks = **80 files**
[Data Specification]
- Content      : Entropy sequence for one chunk (30 seconds), quantized
- Data type    : `np.ndarray(dtype=np.uint8)`
- shape        : `(1500,)`
- Time resolution: **20ms** (= 50 samples per second → 30s × 50 = 1500 points)
- Value range  : [0, 127] (Q=128 quantization levels)

Directory structure:
Entropy_k1_Q128/
├── 21_chunk0.npy ~ 21_chunk7.npy
├── 22_chunk0.npy ~ 22_chunk7.npy
├── 23_chunk0.npy ~ 23_chunk7.npy
├── 24_chunk0.npy ~ 24_chunk7.npy
├── 25_chunk0.npy ~ 25_chunk7.npy
├── 26_chunk0.npy ~ 26_chunk7.npy
├── 27_chunk0.npy ~ 27_chunk7.npy
├── 28_chunk0.npy ~ 28_chunk7.npy
├── 29_chunk0.npy ~ 29_chunk7.npy
├── 30_chunk0.npy ~ 30_chunk7.npy
├── datasheet/
│   ├── bin_table.xlsx
│   └── histograms.png
├── edges.json
└── edges.pkl

[Note]
- The segmentation and naming conventions of these Entropy files are consistent with the
  song segmentation rules defined in the get_30s_file() function in this code, making
  integration easier later.

"""



import os
import glob
import pickle
import numpy as np
import torch
import torchaudio
import pandas as pd
import random
import logging
from scipy.signal import butter, filtfilt
from torch import Tensor
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, KFold
from audiomentations import Compose
from predann.datasets import Dataset
from typing import Any, Tuple, Optional, List
import warnings

from pathlib import Path

def setup_logger():
    # Change log output destination to the same directory as this file
    log_path = Path(__file__).resolve().parent / "dataloader_debug.log"
    if log_path.exists():
        os.remove(log_path)

    logger = logging.getLogger('dataloader_debug')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(str(log_path))  # Absolute path specification
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

debug_logger = setup_logger()

# ============================================================================
# Dataset path configuration (PUBLIC REPOSITORY SAFE DEFAULTS)
# ============================================================================
# IMPORTANT:
# - This public repository does not ship the NMED-T dataset.
# - Configure the dataset location by either:
#   (A) Passing --dataset_dir <NMEDT_BASE_DIR> on the CLI (recommended), or
#   (B) Editing Preprocessing_EEGMusic_dataset._base_dir in this file.
#
# Expected directory structure under <NMEDT_BASE_DIR>:
#   <NMEDT_BASE_DIR>/
#     ├── audio/
#     ├── DS_EEG_pkl/
#     ├── NoClip_Discreat_K1Surprisal/
#     ├── Entropy_k1_Q128/
#     ├── MuQ_Discreat_K128/
#     ├── surprisal_k1/               (optional; raw)
#     ├── entropy_k1/                 (optional; raw)
#     ├── MuQ_Continuous_embedding/   (optional; raw)
#     └── SurpEnt0.1stride*/          (optional; experimental newMF)
#
# The default path is a placeholder to avoid machine-dependent absolute paths.
DEFAULT_NMEDT_BASE_DIR = "/path/to/NMED-T_dataset"


def _as_path(p) -> Path:
    if isinstance(p, Path):
        return p
    return Path(str(p)).expanduser()


def build_new_surpent_roots(base_dir: Path) -> dict:
    base_dir = _as_path(base_dir)
    return {
        8: base_dir / "SurpEnt0.1stride",
        16: base_dir / "SurpEnt0.1stride_ctx16",
        32: base_dir / "SurpEnt0.1stride_ctx32",
    }


_new_surpent_cache = {}


def resolve_new_surpent_root(context_win: int, roots: dict) -> Path:
    cw = int(context_win)
    if cw not in roots:
        raise ValueError(
            f"[newMF] unsupported new_mf_context_win={cw}. supported={sorted(list(roots.keys()))}"
        )
    return roots[cw]



def _load_new_surpent_for_song(song_id: int, new_surpent_root: Path):
    """Load SurpEnt segments with metadata for context window-based retrieval."""
    sid = int(song_id)
    root_str = str(new_surpent_root)
    cache_key = (root_str, sid)
    if cache_key in _new_surpent_cache:
        return _new_surpent_cache[cache_key]

    base_dir = Path(new_surpent_root) / f"{sid:02d}"
    if not base_dir.exists():
        raise FileNotFoundError(f"[newMF] directory not found: {base_dir}")

    surp_fp = base_dir / "surp.npy"
    ent_fp = base_dir / "ent.npy"
    surp_q_fp = base_dir / "surp_Q128.npy"
    ent_q_fp = base_dir / "ent_Q128.npy"
    meta_fp = base_dir / "meta.csv"

    if not surp_fp.exists():
        raise FileNotFoundError(f"[newMF] surp.npy not found: {surp_fp}")
    if not meta_fp.exists():
        raise FileNotFoundError(f"[newMF] meta.csv not found: {meta_fp}")

    surp_raw_np = np.load(str(surp_fp)).astype(np.float32)   # (N,150)
    if surp_raw_np.ndim != 2 or surp_raw_np.shape[1] != 150:
        raise ValueError(f"[newMF] {surp_fp} shape {surp_raw_np.shape}, expected (N, 150)")
    surp_raw = torch.from_numpy(surp_raw_np)  # (N,150)

    ent_raw = None
    ent_raw_np = None
    if ent_fp.exists():
        ent_raw_np = np.load(str(ent_fp)).astype(np.float32)
        if ent_raw_np.shape != surp_raw_np.shape:
            raise ValueError(f"[newMF] ent.npy shape {ent_raw_np.shape} != surp.npy shape {surp_raw_np.shape}")
        ent_raw = torch.from_numpy(ent_raw_np)

    surp_id = None
    if surp_q_fp.exists():
        surp_id_np = np.load(str(surp_q_fp))
        if surp_id_np.shape != surp_raw_np.shape or surp_id_np.dtype != np.uint8:
            raise ValueError(f"[newMF] {surp_q_fp} shape/dtype mismatch: {surp_id_np.shape}, {surp_id_np.dtype}")
        surp_id = torch.from_numpy(surp_id_np.astype(np.int64))  # LongTensor (N,150)

    ent_id = None
    if ent_q_fp.exists() and ent_raw_np is not None:
        ent_id_np = np.load(str(ent_q_fp))
        if ent_id_np.shape != ent_raw_np.shape or ent_id_np.dtype != np.uint8:
            raise ValueError(f"[newMF] {ent_q_fp} shape/dtype mismatch: {ent_id_np.shape}, {ent_id_np.dtype}")
        ent_id = torch.from_numpy(ent_id_np.astype(np.int64))

    meta_df = pd.read_csv(str(meta_fp))
    if "segment_start_s" not in meta_df.columns or "segment_end_s" not in meta_df.columns:
        raise KeyError(f"[newMF] meta.csv must contain 'segment_start_s' and 'segment_end_s': {meta_fp}")
    start_s = meta_df["segment_start_s"].astype(float).values  # (N,)
    end_s = meta_df["segment_end_s"].astype(float).values      # (N,)
    if surp_raw_np.shape[0] != start_s.shape[0]:
        raise ValueError(
            f"[newMF] rows mismatch: surp.npy has {surp_raw_np.shape[0]}, meta.csv has {start_s.shape[0]}"
        )

    cache = {
        "surp_raw": surp_raw,     # FloatTensor (N,150)
        "ent_raw": ent_raw,       # FloatTensor (N,150) or None
        "surp_id": surp_id,       # LongTensor (N,150) or None
        "ent_id": ent_id,         # LongTensor (N,150) or None
        "start_s": start_s,       # np.ndarray (N,)
        "end_s": end_s,           # np.ndarray (N,)
    }
    _new_surpent_cache[cache_key] = cache
    debug_logger.debug(
        f"[newMF] loaded SurpEnt for song_id={sid} from root={root_str}: "
        f"N={surp_raw_np.shape[0]}, surp_id={'ok' if surp_id is not None else 'none'}, "
        f"ent_raw={'ok' if ent_raw is not None else 'none'}, ent_id={'ok' if ent_id is not None else 'none'}"
    )
    return cache


def get_file_list(root, class_song_id):
    class_song_id = list(map(int, class_song_id.strip('[]').split(',')))
    dict_song_id= {song_id: idx for idx, song_id in enumerate(class_song_id)}

    BASE = os.path.join(root, "DS_EEG_pkl")
    if not os.path.exists(BASE):
        raise RuntimeError('BASE folder is not found') 
    EEG_path_list =[p for p in glob.glob(os.path.join(BASE, '*.pkl'), recursive=True) if os.path.isfile(p)]
    EEG_path_list = sorted(EEG_path_list)

    BASE = os.path.join(root, "audio")
    if not os.path.exists(BASE):
        raise RuntimeError('BASE folder is not found')
    Audio_path_list = [p for p in glob.glob(os.path.join(BASE, '*.wav'), recursive=True) if os.path.isfile(p)]
    Audio_path_list = sorted(Audio_path_list)

    df = pd.DataFrame(columns=['subject', 'song', 'trial', "eeg_path", "audio_path"])
    
    for idx, r_path in enumerate(EEG_path_list):

        r_name = os.path.splitext(os.path.basename(r_path))[0]
        r_song_id = int(r_name.split('_')[1])
        index_of_song = class_song_id.index(r_song_id)
        c_path = Audio_path_list[index_of_song]
        c_name = os.path.splitext(os.path.basename(c_path))[0]
        c_song_id = int(c_name.split('_')[0])
        assert r_song_id==c_song_id, "sort_error_1"

        if r_song_id in class_song_id:
            r_subject = r_name.split('_')[0]
            r_trial = r_name.split('_')[2]
            song = dict_song_id[r_song_id]
            c_subject = r_name.split('_')[0]
            c_trial = r_name.split('_')[2]
            assert r_subject==c_subject and r_trial==c_trial, "sort_error_2"

            df.loc[idx] = [r_subject, song, r_trial, r_path, c_path]

    return df

def get_5s_file(df):
    df.insert(5, "chunk", np.zeros(len(df.index)), True)
    newdf = pd.DataFrame(np.repeat(df.values, 48, axis=0))
    newdf.columns = df.columns
    print("check")
    
    for i in range(len(newdf.index)):
        newdf.at[i, 'chunk']=i%48
    return newdf

def get_30s_file(df):
    df.insert(5, "chunk", np.zeros(len(df.index)), True)
    newdf = pd.DataFrame(np.repeat(df.values, 8, axis=0))
    newdf.columns = df.columns
    
    for i in range(len(newdf.index)):
        newdf.at[i, 'chunk']=i%8

    return newdf

def get_window(df,chunk_length,window_size,stride):
    df.insert(6, "window", np.zeros(len(df.index)), True)
    newdf = pd.DataFrame(np.repeat(df.values,int((chunk_length - window_size)/stride + 1), axis=0))
    newdf.columns = df.columns
   
    for i in range(len(newdf.index)):
        newdf.at[i, 'window']=i%int((chunk_length - window_size)/stride + 1)

    return newdf

def K_split_valid(df,fold_num):
    print("fold_num:",fold_num)
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    splits = kf.split(df)
    for fold, (train_index, valid_index) in enumerate(splits):
        if fold == fold_num:
            train_df = df.iloc[train_index]
            test_df = df.iloc[valid_index]
    return train_df, test_df

def check_accessed_data(chunk_length,window_size,stride):
    accessed_data = np.zeros(((int((chunk_length-window_size)/stride)+1)*400, window_size))
    print("check")
    return accessed_data

from scipy.signal import butter, filtfilt

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def _ensure_exists(path: Path, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"[DatasetPath] Missing {kind} file: {path}\n"
            f"Please set --dataset_dir <NMEDT_BASE_DIR> (recommended) or edit _base_dir in preprocessing_eegmusic_dataset_3s.py."
        )

def get_surprisal_30s(song_id: int, chunk: int, base_dir: Path):
    base_dir = _as_path(base_dir)
    surprisal_path = base_dir / "NoClip_Discreat_K1Surprisal" / f"{song_id:02d}_chunk{chunk}.npy"
    _ensure_exists(surprisal_path, "Surprisal (discretized)")
    surprisal = np.load(str(surprisal_path))
    return torch.from_numpy(surprisal).long()


def get_muq_30s(song_id: int, chunk: int, base_dir: Path):
    base_dir = _as_path(base_dir)
    muq_path = base_dir / "MuQ_Discreat_K128" / f"{song_id:02d}_chunk{chunk}.npy"
    _ensure_exists(muq_path, "MuQ (discretized)")
    muq = np.load(str(muq_path))
    return torch.from_numpy(muq).long()


def get_entropy_30s(song_id: int, chunk: int, base_dir: Path):
    base_dir = _as_path(base_dir)
    entropy_path = base_dir / "Entropy_k1_Q128" / f"{song_id:02d}_chunk{chunk}.npy"
    _ensure_exists(entropy_path, "Entropy (discretized)")
    entropy = np.load(str(entropy_path))
    return torch.from_numpy(entropy).long()


def get_surprisal_raw_30s(song_id: int, chunk: int, base_dir: Path):
    base_dir = _as_path(base_dir)
    surprisal_path = base_dir / "surprisal_k1" / f"{song_id:02d}_chunk{chunk}.npy"
    _ensure_exists(surprisal_path, "Surprisal (raw)")
    surprisal = np.load(str(surprisal_path))
    return torch.from_numpy(surprisal.astype(np.float32))


def get_muq_raw_30s(song_id: int, chunk: int, base_dir: Path):
    base_dir = _as_path(base_dir)
    muq_path = base_dir / "MuQ_Continuous_embedding" / f"{song_id:02d}_chunk{chunk}.npy"
    _ensure_exists(muq_path, "MuQ (raw)")
    muq = np.load(str(muq_path))
    return torch.from_numpy(muq.astype(np.float32))


def get_entropy_raw_30s(song_id: int, chunk: int, base_dir: Path):
    base_dir = _as_path(base_dir)
    entropy_path = base_dir / "entropy_k1" / f"{song_id:02d}_chunk{chunk}.npy"
    _ensure_exists(entropy_path, "Entropy (raw)")
    entropy = np.load(str(entropy_path))
    return torch.from_numpy(entropy.astype(np.float32))


class Preprocessing_EEGMusic_dataset(Dataset):

    _base_dir = DEFAULT_NMEDT_BASE_DIR

    def __init__(
        self,
        root: str,
        base_dir: str = _base_dir,
        download: bool = False,
        subset: Optional[str] = None,
    ):

        self.root = root

        # Resolve dataset base directory:
        # Priority:
        #   1) root (passed from get_dataset(args.dataset_dir)) if it exists
        #   2) base_dir (class default or user override) if it exists
        # If none exists, keep the best-effort path and raise later when files are accessed.
        root_path = _as_path(root) if root is not None else None
        base_dir_path = _as_path(base_dir) if base_dir is not None else None

        resolved_base_dir = None
        if root_path is not None and root_path.exists():
            resolved_base_dir = root_path
        elif base_dir_path is not None and base_dir_path.exists():
            resolved_base_dir = base_dir_path
        else:
            resolved_base_dir = base_dir_path if base_dir_path is not None else _as_path(DEFAULT_NMEDT_BASE_DIR)
            debug_logger.warning(
                f"[DatasetPath] NMED-T base directory does not exist: {resolved_base_dir}. "
                "Please set --dataset_dir <NMEDT_BASE_DIR> (recommended) or edit _base_dir in this file."
            )

        self.base_dir = str(resolved_base_dir)
        self.base_dir_path = resolved_base_dir

        self.subset = subset
        self.eeg_normalization = None
        self.transform = None
        self.eeg_sample_rate = 125
        self.eeg_clip_length = 375
        self.audio_length = 0
        self.audio_sample_rate = 44100
        self.class_song_id = "[21,22,23,24,25,26,27,28,29,30]"
        self.train_test_splitting = "random_split_30s"
        self.random_numbers = []
        self.start_position = 0
        self.evaluation_length = 375

        # newMF flags must be initialized here to avoid AttributeError in evaluation pipelines
        self.use_new_surp = False
        self.use_new_ent = False

        assert subset is None or subset in [
            "train",
            "valid",
            "test",
            "CV",
            "SW_train",
            "SW_valid",
            "SW_test",
            "probe_train",
            "probe_test",
        ], (
            "When `subset` is not None, it must be one of: "
            "{'train','valid','test','CV','SW_train','SW_valid','SW_test','probe_train','probe_test'}."
        )

        self.window_size = None
        self.stride = None
        self.fold = None
        self.mode = None
        self.start = []
        self.start_value = 0

        self.fs = 125.0
        self.lowcut = 1.0
        self.highcut = 50.0

        # newMF (SurpEnt0.1stride) directory mapping depends on base_dir
        self.new_mf_context_win = 8
        self.new_surpent_roots = build_new_surpent_roots(self.base_dir_path)
        self.new_surpent_root = resolve_new_surpent_root(self.new_mf_context_win, self.new_surpent_roots)

        # Build main dataframe from resolved base_dir
        self.df = get_file_list(self.base_dir, self.class_song_id).reset_index(drop=True)

        if self.train_test_splitting == "random_split_5s":
            self.df = get_5s_file(self.df)
            self.chunk_length = 5 * 125
        if self.train_test_splitting == "random_split_30s":
            self.df = get_30s_file(self.df)
            self.chunk_length = 30 * 125

        # In __init__, only initialize self.df; splitting is done in set_other_parameters
        if self.subset == "CV":
            self.df_subset = self.df
        else:
            self.df_subset = None

 

    def file_path(self, n: int) -> str:
        pass

    def set_transform(self, transform):
        self.transform = Compose(transform)
        print(self.transform)

    def set_other_parameters(self, eeg_clip_length, audio_clip_length, split_seed, class_song_id, shifting_time, start_position):
        self.eeg_clip_length = eeg_clip_length
        self.audio_clip = audio_clip_length
        self.audio_length = audio_clip_length * self.audio_sample_rate
        self.class_song_id = class_song_id
        self.shifting_time = shifting_time
        self.start_position = start_position

        # Always rebuild df from the resolved base_dir to avoid accidental path drift
        self.df = get_file_list(self.base_dir, self.class_song_id).reset_index(drop=True)

        if self.train_test_splitting == "random_split_5s":
            self.df = get_5s_file(self.df)

        if self.train_test_splitting == "random_split_30s":
            self.df = get_30s_file(self.df)

        if self.subset == "CV":
            self.df_subset = self.df
        else:

            df_train, df_test = train_test_split(
                self.df,
                test_size=0.25,
                random_state=split_seed,
                stratify=self.df.song,
            )

            if self.subset == "train":
                self.df_subset = df_train
            elif self.subset == "test" or self.subset == "valid":
                self.df_subset = df_test
            elif self.subset == "SW_train":
                self.df_subset = df_train
            elif self.subset == "SW_test" or self.subset == "SW_valid":
                new_valid_df = get_window(df_test, self.chunk_length, self.window_size, self.stride)
                self.df_subset = new_valid_df
                self.df_subset = self.df_subset[self.df_subset.iloc[:, 6] >= (self.audio_clip - 3) / 2]
                self.accessed_data = check_accessed_data(self.chunk_length, self.window_size, self.stride)


    def set_random_numbers(self, random_numbers):
        self.random_numbers = random_numbers

    
    def set_sliding_window_parameters(self, window_size, stride):
        self.window_size = window_size
        self.stride = stride

    def set_mode(self, mode: Optional[str]) -> None:
        self.mode = mode

    def labels(self):
        num_label = len(self.df_subset.song.unique())
        return num_label

    def set_eeg_normalization(self, eeg_normalization, clamp_value=None):
        self.eeg_normalization = eeg_normalization
        self.clamp_value = clamp_value
    def set_new_mf_flags(self, use_new_surp: bool = False, use_new_ent: bool = False) -> None:
        self.use_new_surp = bool(use_new_surp)
        self.use_new_ent = bool(use_new_ent)

    def set_new_mf_context_win(self, context_win: int) -> None:
        self.new_mf_context_win = int(context_win)
        self.new_surpent_root = resolve_new_surpent_root(self.new_mf_context_win, self.new_surpent_roots)





    def _get_new_mf_segment(
        self,
        song_id: int,
        chunk: int,
        region_start_samples: int,
    ):
        """Retrieve 3s newMF segment matching EEG extraction position within chunk boundaries."""
        cache = _load_new_surpent_for_song(song_id, self.new_surpent_root)
        start_s_arr = cache["start_s"]
        end_s_arr = cache["end_s"]

        chunk = int(chunk)
        chunk_start_s = float(chunk) * 30.0
        chunk_end_s = chunk_start_s + 30.0

        t0 = chunk_start_s + float(region_start_samples) / float(self.eeg_sample_rate)

        inside_mask = (start_s_arr >= chunk_start_s) & (end_s_arr <= chunk_end_s)
        inside_idx = np.nonzero(inside_mask)[0]
        if inside_idx.size == 0:
            raise RuntimeError(
                f"[newMF] no SurpEnt segments fully inside chunk for song_id={song_id}, chunk={chunk}"
            )

        # Select the segment with segment_start_s closest to t0
        diffs = np.abs(start_s_arr[inside_idx] - t0)
        j_rel = int(diffs.argmin())
        j = int(inside_idx[j_rel])

        if diffs[j_rel] > 0.151:
            debug_logger.warning(
                f"[newMF] large time diff: target={t0:.3f}s vs segment_start={start_s_arr[j]:.3f}s "
                f"(Δ={diffs[j_rel]:.3f}s) for song_id={song_id}, chunk={chunk}"
            )

        surprisal_id_3s = None
        surprisal_raw_3s = None
        entropy_id_3s = None
        entropy_raw_3s = None

        if self.use_new_surp:
            surp_id_mat = cache.get("surp_id")
            surp_raw_mat = cache.get("surp_raw")
            if surp_id_mat is None or surp_raw_mat is None:
                raise RuntimeError(
                    "[newMF] Surprisal new MF is requested but surp_Q128.npy or surp.npy is missing."
                )
            surprisal_id_3s = surp_id_mat[j].clone().long()      # (150,)
            surprisal_raw_3s = surp_raw_mat[j].clone().float()   # (150,)

        if self.use_new_ent:
            ent_id_mat = cache.get("ent_id")
            ent_raw_mat = cache.get("ent_raw")
            if ent_id_mat is None or ent_raw_mat is None:
                raise RuntimeError(
                    "[newMF] Entropy new MF is requested but ent_Q128.npy or ent.npy is missing."
                )
            entropy_id_3s = ent_id_mat[j].clone().long()         # (150,)
            entropy_raw_3s = ent_raw_mat[j].clone().float()      # (150,)

        return surprisal_id_3s, surprisal_raw_3s, entropy_id_3s, entropy_raw_3s

    def K_split(self, k=4, random_state=42):
        train_list = [[] for _ in range(k)]
        valid_list = [[] for _ in range(k)]

        for song_id in self.df_subset.song.drop_duplicates().values:
            df_song = self.df_subset[self.df_subset.song == song_id]
            all_list_song = df_song.index.tolist()
            
            for i in range(k):
                valid_list_song_fold = df_song.subject.drop_duplicates().sample(6, random_state=random_state).index.tolist()
                train_list_song_fold = [x for x in all_list_song if x not in valid_list_song_fold]
                valid_list[i].extend(valid_list_song_fold)
                train_list[i].extend(train_list_song_fold)

                df_song = df_song.drop(valid_list_song_fold)

        Kfold_list = [[np.array(train_fold), np.array(valid_fold)] for train_fold, valid_fold in zip (train_list, valid_list)]
        
        return Kfold_list

    def K_split_random(self, k=4, random_state=42):
        kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
        return (kf.split(self.df_subset))
    

    def getitem(self, n, isClip=True, deterministic=False):
        p = self.start_position

        # Paper-scope modes (Pretrain is removed)
        mode = self.mode if self.mode is not None else "Finetune"
        valid_modes = {"Finetune", "MuQMultitask", "SurpMultitask", "EntropyMultitask"}
        if mode not in valid_modes:
            raise ValueError(
                f"[DatasetMode] Unsupported mode='{mode}'. "
                "Supported modes: Finetune, MuQMultitask, SurpMultitask, EntropyMultitask."
            )
        effective_mode = mode

        need_surprisal = effective_mode == "SurpMultitask"
        need_muq = effective_mode == "MuQMultitask"
        need_entropy = effective_mode == "EntropyMultitask"
        need_surprisal_raw = effective_mode == "SurpMultitask"
        need_muq_raw = effective_mode == "MuQMultitask"
        need_entropy_raw = effective_mode == "EntropyMultitask"

        chunk = None
        song_id = None
        song_idx = None
        window_idx = None

        eeg_path = self.df_subset.iloc[n, 3]
        with open(eeg_path, "rb") as f:
            eeg = pickle.load(f)

        if self.train_test_splitting == "random_split_30s":
            chunk = self.df_subset.iloc[n, 5]
            eeg = eeg[
                :,
                int(int(chunk) * (125 * 30) + (0.125 * self.shifting_time)) : int(int(chunk + 1) * (125 * 30) + (0.125 * self.shifting_time)),
            ]

        if self.train_test_splitting != "random_split_30s":
            warnings.warn("Surprisal/Entropy/MuQ features are currently prepared for 30s chunks.")
            if self.train_test_splitting == "random_split_5s":
                chunk = self.df_subset.iloc[n, 5]
                eeg = eeg[
                    :,
                    int(int(chunk) * (125 * 5) + (0.125 * self.shifting_time)) : int(int(chunk + 1) * (125 * 5) + (0.125 * self.shifting_time)),
                ]

        if self.subset == "SW_valid":
            window = self.df_subset.iloc[n, 6]
            eeg = eeg[:, int((window) * self.stride) + p : int((window) * self.stride + self.window_size) + p]
            window_idx = int(window)

        surprisal = None
        muq = None
        entropy = None
        surprisal_raw = None
        muq_raw = None
        entropy_raw = None

        if self.train_test_splitting == "random_split_30s":
            chunk = int(self.df_subset.iloc[n, 5])

            class_song_id = list(map(int, self.class_song_id.strip("[]").split(",")))
            song_idx = int(self.df_subset.iloc[n, 1])
            song_id = class_song_id[song_idx]

            # Old Surprisal/Entropy load only when not using newMF
            if need_surprisal and not self.use_new_surp:
                surprisal = get_surprisal_30s(song_id, chunk, self.base_dir_path)
            if need_muq:
                muq = get_muq_30s(song_id, chunk, self.base_dir_path)
            if need_entropy and not self.use_new_ent:
                entropy = get_entropy_30s(song_id, chunk, self.base_dir_path)

            if need_surprisal_raw and not self.use_new_surp:
                surprisal_raw = get_surprisal_raw_30s(song_id, chunk, self.base_dir_path)
            if need_muq_raw:
                muq_raw = get_muq_raw_30s(song_id, chunk, self.base_dir_path)
            if need_entropy_raw and not self.use_new_ent:
                entropy_raw = get_entropy_raw_30s(song_id, chunk, self.base_dir_path)

        if self.subset == "SW_valid":
            window = self.df_subset.iloc[n, 6]
            if surprisal is not None:
                surprisal_start = int(window * self.stride * 50 / 125)  # 50=1s/20ms
                surprisal = surprisal[surprisal_start : surprisal_start + int(self.window_size * 50 / 125)]
            if surprisal_raw is not None:
                surprisal_start = int(window * self.stride * 50 / 125)
                surprisal_raw = surprisal_raw[surprisal_start : surprisal_start + int(self.window_size * 50 / 125)]
            if muq is not None:
                muq_start = int(window * self.stride * 25 / 125)  # 25 = 1 s / 40 ms
                muq = muq[muq_start : muq_start + int(self.window_size * 25 / 125)]
            if muq_raw is not None:
                muq_start = int(window * self.stride * 25 / 125)
                muq_raw = muq_raw[muq_start : muq_start + int(self.window_size * 25 / 125)]
            if entropy is not None:
                entropy_start = int(window * self.stride * 50 / 125)
                entropy = entropy[entropy_start : entropy_start + int(self.window_size * 50 / 125)]
            if entropy_raw is not None:
                entropy_start = int(window * self.stride * 50 / 125)
                entropy_raw = entropy_raw[entropy_start : entropy_start + int(self.window_size * 50 / 125)]

        label = self.df_subset.iloc[n, 1]

        if isClip:
            eeg_length = eeg.size(1)

            if deterministic:
                # Always take the center 3s segment for deterministic evaluation
                eeg_start = int((eeg_length - self.eeg_clip_length) // 2)
            else:
                eeg_start = random.randint(
                    int((self.audio_clip - 3) / 2 * 125),
                    int(eeg_length - self.eeg_clip_length - (self.audio_clip - 3) / 2 * 125 - 1),
                )

            eeg = eeg[:, eeg_start : eeg_start + self.eeg_clip_length]

            region_start_in_chunk = eeg_start
            if self.subset == "SW_valid":
                if window_idx is None:
                    window_idx = int(self.df_subset.iloc[n, 6])
                region_start_in_chunk = int(window_idx * self.stride) + eeg_start

            if (
                self.train_test_splitting == "random_split_30s"
                and (self.use_new_surp or self.use_new_ent)
                and song_id is not None
                and chunk is not None
            ):
                new_surp_id, new_surp_raw, new_ent_id, new_ent_raw = self._get_new_mf_segment(
                    song_id=song_id,
                    chunk=chunk,
                    region_start_samples=region_start_in_chunk,
                )
                if self.use_new_surp and need_surprisal:
                    surprisal = new_surp_id
                    surprisal_raw = new_surp_raw
                if self.use_new_ent and need_entropy:
                    entropy = new_ent_id
                    entropy_raw = new_ent_raw

            if not self.use_new_surp:
                if surprisal is not None:
                    surprisal_start = int(eeg_start / 125 * 50)
                    surprisal = surprisal[surprisal_start : surprisal_start + int(self.eeg_clip_length / 125 * 50)]
                if surprisal_raw is not None:
                    surprisal_start = int(eeg_start / 125 * 50)
                    surprisal_raw = surprisal_raw[
                        surprisal_start : surprisal_start + int(self.eeg_clip_length / 125 * 50)
                    ]

            if muq is not None:
                muq_start = int(eeg_start / 125 * 25)
                muq = muq[muq_start : muq_start + int(self.eeg_clip_length / 125 * 25)]
            if muq_raw is not None:
                muq_start = int(eeg_start / 125 * 25)
                muq_raw = muq_raw[muq_start : muq_start + int(self.eeg_clip_length / 125 * 25)]

            if not self.use_new_ent:
                if entropy is not None:
                    entropy_start = int(eeg_start / 125 * 50)
                    entropy = entropy[entropy_start : entropy_start + int(self.eeg_clip_length / 125 * 50)]
                if entropy_raw is not None:
                    entropy_start = int(eeg_start / 125 * 50)
                    entropy_raw = entropy_raw[
                        entropy_start : entropy_start + int(self.eeg_clip_length / 125 * 50)
                    ]

        if self.eeg_normalization == "channel_mean":
            eeg = self.normalize_EEG(eeg)
        elif self.eeg_normalization == "all_mean":
            eeg = self.normalize_EEG_2(eeg)
        elif self.eeg_normalization == "constant_multiple":
            eeg = self.normalize_EEG_3(eeg)
        elif self.eeg_normalization == "MetaAI":
            eeg = self.normalize_EEG_4(eeg, self.clamp_value)

        if self.transform is not None:
            eeg = eeg.to("cpu").detach().numpy().copy()
            eeg = self.transform(eeg, sample_rate=self.eeg_sample_rate)
            eeg = torch.from_numpy(eeg.astype(np.float32)).clone()

        if surprisal_raw is not None:
            surprisal_raw = surprisal_raw.float()
        if muq_raw is not None:
            muq_raw = muq_raw.float()
        if entropy_raw is not None:
            entropy_raw = entropy_raw.float()

        if effective_mode == "Finetune":
            return eeg, label
        if effective_mode == "MuQMultitask":
            if muq is None or muq_raw is None:
                raise ValueError("MuQ features must be available in MuQMultitask mode")
            return eeg, label, muq, muq_raw
        if effective_mode == "SurpMultitask":
            if surprisal is None or surprisal_raw is None:
                raise ValueError("Surprisal features must be available in SurpMultitask mode")
            return eeg, label, surprisal, surprisal_raw
        if effective_mode == "EntropyMultitask":
            if entropy is None or entropy_raw is None:
                raise ValueError("Entropy features must be available in EntropyMultitask mode")
            return eeg, label, entropy, entropy_raw

        raise RuntimeError(f"Unreachable: effective_mode={effective_mode}")


    def __getitem__(self, n: int) -> Tuple[Tensor, ...]:
        deterministic = False
        if self.subset in ["test", "valid", "SW_valid", "probe_test"]:
            deterministic = True
        return self.getitem(n, deterministic=deterministic)

    def __len__(self) -> int:
        return len(self.df_subset)

    def normalize_EEG(self,eeg):
        eeg_mean=torch.mean(eeg,1)
        eeg=eeg-eeg_mean.unsqueeze(1)
        max_eeg=torch.max(abs(eeg),1)
        eeg=eeg/max_eeg.values.unsqueeze(1)
        return eeg
    
    def normalize_EEG_2(self,eeg):
        eeg_mean=torch.mean(eeg)*torch.ones(eeg.shape[0])
        eeg=eeg-eeg_mean.unsqueeze(1)
        max_eeg=torch.max(abs(eeg),1)
        eeg=eeg/max_eeg.values.unsqueeze(1)
        return eeg

    def normalize_EEG_3(self,eeg):
        eeg=100*eeg
        return eeg

    def normalize_EEG_4(self,eeg,clamp_value):
        for idx, ch_eeg in enumerate(eeg):
            transformer = RobustScaler().fit(ch_eeg.view(-1,1))
            ch_eeg = transformer.transform(ch_eeg.view(-1,1))
            ch_eeg = torch.from_numpy(ch_eeg.astype(np.float32)).clone()
            eeg[idx] = ch_eeg.view(1,-1)

        eeg = torch.clamp(eeg, min=int(-1*clamp_value), max=int(clamp_value))
        return eeg

    def get_last_iteration_value(self):
        if self.subset == "SW_valid": 
            return self.start_value

    def check_access(self,n):
        if self.subset == "SW_valid": 
            self.accessed_data[n][self.eeg_start:self.eeg_start + self.eeg_clip_length] = 1
            
            if np.all(self.accessed_data[n] == 1):
                debug_logger.debug(f"All data in item {n} has been accessed.")
            full_ones_rows = np.all(self.accessed_data == 1, axis=1).sum()
            percentage = (full_ones_rows / len(self.df_subset)) * 100
            if n==len(self.df_subset)-1:
                ones_ratio = np.mean(self.accessed_data[n] == 1)
                self.start.append(self.eeg_start)
                debug_logger.debug(f"The start array is:{self.start}")
                debug_logger.debug(f"The data is:{self.df_subset.iloc[n]}")
                debug_logger.debug(f"The proportion of 1 is:{ones_ratio}")
                debug_logger.debug(f"The percentage of rows in 'accessed_data' where all elements are 1 is: {percentage}%")

        return self.start