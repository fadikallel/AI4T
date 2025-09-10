import os
import random
import subprocess
import tempfile
from tqdm import tqdm
from pathlib import Path

import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, Wav2Vec2Model

# ---------------- Dataset Config ---------------- #
BASE_PATHS = {
    "asv19": {
        "train": "/ds-slt/audio/ASVSpoof_LA_19/ASVspoof2019_LA_train/flac/",
        "dev": "/ds-slt/audio/ASVSpoof_LA_19/ASVspoof2019_LA_dev/flac/",
        "eval": "/ds-slt/audio/ASVSpoof_LA_19/ASVspoof2019_LA_eval/flac/",
    },
    "asv21": {"eval": "/netscratch/fkallel/DATA/ASVspoof2021_DF_eval/flac/"},
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
            dataset, utt_id = parts[0].lower(), parts[1]
            # decide split
            if dataset == "asv19":
                split = (
                    "train"
                    if utt_id.startswith("LA_T")
                    else "dev"
                    if utt_id.startswith("LA_D")
                    else "eval"
                )
            elif dataset == "asv5":
                split = (
                    "train"
                    if utt_id.startswith("T_")
                    else "dev"
                    if utt_id.startswith("D_")
                    else None
                )
            elif dataset == "asv21":
                split = "eval"
            else:
                continue
            if split is None:
                continue
            entries.append((dataset, split, utt_id))
    return entries


def get_audio_path(dataset, split, utt_id):
    return os.path.join(BASE_PATHS[dataset][split], f"{utt_id}.flac")


# ---------------- Audio Augmentation ---------------- #
def augment_audio(input_file):
    codec = random.choice(["aac", "libopus"])
    bitrate = "128k" if codec == "aac" else "64k"
    suffix = ".m4a" if codec == "aac" else ".ogg"

    fd, tmp_out = tempfile.mkstemp(suffix=suffix, dir="/tmp")
    os.close(fd)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_file,
        "-c:a",
        codec,
        "-b:a",
        bitrate,
        "-loglevel",
        "quiet",
        tmp_out,
    ]
    subprocess.run(command, check=True)
    return tmp_out


# ---------------- Safe Loader ---------------- #
def safe_load_audio(path, target_sr=16000):
    # decode to wav with ffmpeg, then load with torchaudio
    fd, tmp_wav = tempfile.mkstemp(suffix=".wav", dir="/tmp")
    os.close(fd)
    command = ["ffmpeg", "-y", "-i", path, "-ar", str(target_sr), "-ac", "1", "-loglevel", "quiet",tmp_wav]
    subprocess.run(command, check=True)
    waveform, sr = torchaudio.load(tmp_wav)
    os.remove(tmp_wav)
    return waveform.squeeze(0), sr


# ---------------- Dataset ---------------- #
class AudioDataset(Dataset):
    def __init__(self, metadata_file):
        self.entries = read_metadata(metadata_file)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        dataset, split, utt_id = self.entries[idx]
        audio_path = get_audio_path(dataset, split, utt_id)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"{audio_path} not found")

        # augment + load
        aug_path = augment_audio(audio_path)
        waveform, sr = safe_load_audio(aug_path)
        return waveform, sr


# ---------------- Model ---------------- #
class Wav2Vec2Truncated(Wav2Vec2Model):
    def __init__(self, config):
        super().__init__(config)
        self.encoder.layers = self.encoder.layers[:9]


class HuggingFaceFeatureExtractor:
    def __init__(self, model_class, name="facebook/wav2vec2-xls-r-2b"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(name)
        self.model = model_class.from_pretrained(name, output_hidden_states=True)
        self.model.eval().to(self.device)

    def __call__(self, batch_audio, batch_sr):
        inputs = self.feature_extractor(
            batch_audio,
            sampling_rate=batch_sr[0],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.hidden_states[9]


# ---------------- Main Loop ---------------- #
def extract_features_codec(outdir, metadata_file, batch_size=8):
    dataset = AudioDataset(metadata_file)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: list(zip(*x))
    )

    feature_extractor = HuggingFaceFeatureExtractor(Wav2Vec2Truncated)
    all_embeddings = []

    for batch_audio, batch_sr in tqdm(dataloader):
        # batch_audio is list of tensors → convert to list of numpy arrays
        audio_list = [a.numpy() for a in batch_audio]
        sr_list = list(batch_sr)

        hidden_states = feature_extractor(audio_list, sr_list)
        mean_layer_output = torch.mean(hidden_states, dim=1).cpu().numpy()
        all_embeddings.append(mean_layer_output)

    stacked_embeddings = np.vstack(all_embeddings)
    os.makedirs(outdir, exist_ok=True)
    out_file = os.path.join(outdir, "wav2vec2-xls-r-2b_augm_codecs_Layer9.npy")
    np.save(out_file, stacked_embeddings)
    print("Saved:", out_file, stacked_embeddings.shape)


# ---------------- Entry ---------------- #
if __name__ == "__main__":
    metadata_file = "./processed_metadata/metadata_marginPruned_XLS_fromALL_margin_both_135.txt"
    outdir = "./feats/wav2vec2-xls-r-2b/"
    extract_features_codec(outdir, metadata_file, batch_size=8)
