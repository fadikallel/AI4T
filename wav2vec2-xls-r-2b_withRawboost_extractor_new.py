import os
import torch
import librosa
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, Wav2Vec2Model
from RawBoost import (
    ISD_additive_noise,
    LnL_convolutive_noise,
    SSI_additive_noise,
    normWav,
)

# ---------------- RawBoost Augmentation ---------------- #

def process_Rawboost_feature(feature, sr, args, algo):
    if algo == 1:
        feature = LnL_convolutive_noise(
            feature,
            args.N_f, args.nBands,
            args.minF, args.maxF,
            args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff,
            args.minG, args.maxG,
            args.minBiasLinNonLin, args.maxBiasLinNonLin,
            sr,
        )
    elif algo == 2:
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
    elif algo == 3:
        feature = SSI_additive_noise(
            feature, args.SNRmin, args.SNRmax,
            args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff,
            args.maxCoeff, args.minG, args.maxG, sr,
        )
    elif algo == 4:
        feature = process_Rawboost_feature(feature, sr, args, 1)
        feature = process_Rawboost_feature(feature, sr, args, 2)
        feature = process_Rawboost_feature(feature, sr, args, 3)
    elif algo == 5:
        feature = process_Rawboost_feature(feature, sr, args, 1)
        feature = process_Rawboost_feature(feature, sr, args, 2)
    elif algo == 6:
        feature = process_Rawboost_feature(feature, sr, args, 1)
        feature = process_Rawboost_feature(feature, sr, args, 3)
    elif algo == 7:
        feature = process_Rawboost_feature(feature, sr, args, 2)
        feature = process_Rawboost_feature(feature, sr, args, 3)
    elif algo == 8:
        f1 = process_Rawboost_feature(feature, sr, args, 1)
        f2 = process_Rawboost_feature(feature, sr, args, 2)
        feature = normWav(f1 + f2, 0)
    return feature

# ---------------- Dataset Paths ---------------- #

BASE_PATHS = {
    "asv19": {
        "train": "/ds-slt/audio/ASVSpoof_LA_19/ASVspoof2019_LA_train/flac/",
        "dev": "/ds-slt/audio/ASVSpoof_LA_19/ASVspoof2019_LA_dev/flac/",
        "eval": "/ds-slt/audio/ASVSpoof_LA_19/ASVspoof2019_LA_eval/flac/",
    },
    "asv21": {
        "eval": "/netscratch/fkallel/DATA/ASVspoof2021_DF_eval/flac/",
    },
    "asv5": {
        "train": "/ds-slt/audio/ASVSpoof2024/flac_T/",
        "dev": "/ds-slt/audio/ASVSpoof2024/flac_D/",
    },
}

def read_metadata(file_path):
    entries = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue
            dataset = parts[0].lower()
            utt_id = parts[1]
            label = parts[2] if len(parts) > 2 else None

            if dataset == "asv19":
                if utt_id.startswith("LA_T"): split="train"
                elif utt_id.startswith("LA_D"): split="dev"
                elif utt_id.startswith("LA_E"): split="eval"
                else: continue
            elif dataset == "asv5":
                if utt_id.startswith("T_"): split="train"
                elif utt_id.startswith("D_"): split="dev"
                else: continue
            elif dataset == "asv21":
                split="eval"
            else: continue

            entries.append((dataset, split, utt_id, label))
    return entries

def get_audio_path(dataset, split, utt_id):
    return os.path.join(BASE_PATHS[dataset][split], f"{utt_id}.flac")

# ---------------- Wav2Vec2 Truncated ---------------- #

class Wav2Vec2Truncated(Wav2Vec2Model):
    def __init__(self, config):
        super().__init__(config)
        self.encoder.layers = self.encoder.layers[:9]

class HuggingFaceFeatureExtractor:
    def __init__(self, model_class, name=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(name)
        self.model = model_class.from_pretrained(name, output_hidden_states=True)
        self.model.eval().to(self.device)

    def __call__(self, audios, sr):
        inputs = self.feature_extractor(
            [a.numpy() for a in audios], sampling_rate=sr,
            return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k,v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.hidden_states[9]

FEATURE_EXTRACTORS = {
    "wav2vec2-xls-r-2b": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Truncated, "facebook/wav2vec2-xls-r-2b"
    )
}

# ---------------- Dataset & DataLoader ---------------- #

class RawBoostDataset(Dataset):
    def __init__(self, entries, args, algo):
        self.entries = entries
        self.args = args
        self.algo = algo

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        dataset, split, utt_id, label = self.entries[idx]
        path = get_audio_path(dataset, split, utt_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        audio, sr = librosa.load(path, sr=16000)
        audio = process_Rawboost_feature(audio, sr, self.args, self.algo)
        return torch.tensor(audio, dtype=torch.float32), sr, label

def collate_fn(batch):
    audios, srs, labels = zip(*batch)
    max_len = max(a.shape[0] for a in audios)
    padded = torch.zeros(len(audios), max_len)
    for i, a in enumerate(audios):
        padded[i, :a.shape[0]] = a
    return padded, srs[0], labels

# ---------------- Feature Extraction ---------------- #

def extract_features_rawboost(outdir, metadata_file, args, algo, batch_size=8):
    entries = read_metadata(metadata_file)
    print(f"Metadata contains {len(entries)} files.")

    dataset = RawBoostDataset(entries, args, algo)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=4, pin_memory=True)

    feature_extractor = FEATURE_EXTRACTORS["wav2vec2-xls-r-2b"]()
    layer_embeddings = []

    for audios, sr, labels in tqdm(dataloader):
        hidden_states = feature_extractor(audios, sr)
        mean_layer_output = torch.mean(hidden_states, dim=1).cpu().numpy()
        layer_embeddings.append(mean_layer_output)

    stacked_embeddings = np.vstack(layer_embeddings)
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, f"wav2vec2-xls-r-2b_augm_rb_Layer9.npy"), stacked_embeddings)

# ---------------- Main ---------------- #

if __name__ == "__main__":
    print("script running")
    metadata_file = "./processed_metadata/metadata_marginPruned_XLS_fromALL_margin_both_135.txt"
    outdir = "./feats/wav2vec2-xls-r-2b/"

    class Args:
        algo = 5
        nBands = 5
        minF = 20
        maxF = 8000
        minBW = 100
        maxBW = 1000
        minCoeff = 10
        maxCoeff = 100
        minG = 0
        maxG = 0
        minBiasLinNonLin = 5
        maxBiasLinNonLin = 20
        N_f = 5
        P = 10
        g_sd = 2
        SNRmin = 10
        SNRmax = 40
    args = Args()

    extract_features_rawboost(outdir, metadata_file, args, algo=5, batch_size=8)
