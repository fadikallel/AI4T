import os
import torch
import librosa
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, Wav2Vec2Model


class HuggingFaceFeatureExtractor:
    def __init__(self, model_class, name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(name)
        self.model = model_class.from_pretrained(name, output_hidden_states=True)
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, audios, srs):
        inputs = self.feature_extractor(
            audios,
            sampling_rate=srs[0],  # assume consistent SR
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.hidden_states

import torchaudio

class AudioDataset(Dataset):
    def __init__(self, file_list, indir, sr=16000):
        self.file_list = file_list
        self.indir = indir
        self.sr = sr
        self.resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=sr)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fi = self.file_list[idx]
        path = os.path.join(self.indir, fi)
        audio, sr_orig = torchaudio.load(path)  # [channels, time], original SR
        # convert to mono if needed
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        audio = torchaudio.functional.resample(audio, orig_freq=sr_orig, new_freq=self.sr)
        return audio.squeeze(0).numpy(), self.sr, fi  # convert to 1D numpy array for HF extractor


def read_metadata(file_path):
    relevant_files = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) > 1:
                relevant_files.append(parts[0])
    return relevant_files


def collate_fn(batch):
    audios, srs, filenames = zip(*batch)
    return list(audios), list(srs), list(filenames)


def main(outdir, indir, metadata_file, batch_size=8, num_workers=2):
    relevant_files = read_metadata(metadata_file)
    print(f"Metadata contains {len(relevant_files)} files.")
    feature_extractor = HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-xls-r-2b"
    )

    dataset = AudioDataset(relevant_files, indir, sr=16000)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    layer_embeddings = []

    for audios, srs, filenames in tqdm(dataloader):
        hidden_states = feature_extractor(audios, srs)
        layer_output = hidden_states[9]  # [B, T, D]
        mean_layer_output = torch.mean(layer_output, dim=1).cpu().numpy()
        layer_embeddings.append(mean_layer_output)

    stacked_embeddings = np.vstack(layer_embeddings)
    os.makedirs(outdir, exist_ok=True)
    np.save(
        os.path.join(outdir, "wav2vec2-xls-r-2b_Layer9_timit.npy"),
        stacked_embeddings,
    )


if __name__ == "__main__":
    print("script running")
    indir = "/ds-slt/audio/TIMIT/TIMIT-TTS/CLEAN"
    outdir = "./feats/wav2vec2-xls-r-2b/"
    metadata_file = "./processed_metadata/timit_systems.csv"
    main(outdir, indir, metadata_file, batch_size=8)
