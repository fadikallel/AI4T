import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, Wav2Vec2Model
import argparse
from config import DATASETS

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


class AudioDataset(Dataset):
    def __init__(self, file_list, indir, sr=16000, flac=False, wav=False):
        self.file_list = file_list
        self.indir = indir
        self.sr = sr
        self.flac = flac
        self.wav = wav

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fi = self.file_list[idx]
        if self.flac:
            fi = fi + ".flac"
        if self.wav:
            fi = fi + ".wav"
        path = os.path.join(self.indir, fi)
        waveform, sr = torchaudio.load(path)  # [channels, time], sample rate
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            waveform = resampler(waveform)
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = waveform.squeeze().numpy()
        return waveform, self.sr, fi


def read_metadata(file_path, split="|"):
    relevant_files = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split(split)
            if len(parts) > 1:
                relevant_files.append(parts[0])
    return relevant_files


def collate_fn(batch):
    audios, srs, filenames = zip(*batch)
    return list(audios), list(srs), list(filenames)


def main(config, batch_size=8, num_workers=2):
    metadata_file = config["metadata"]
    wav = config.get("wav", False)   
    split = " " if wav else "|" 
    relevant_files = read_metadata(metadata_file, split=split)
    print(f"Metadata contains {len(relevant_files)} files.")
    
    feature_extractor = HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-xls-r-2b"
    )
    flac = config.get("flac", False)
    dataset = AudioDataset(relevant_files, config["indir"], sr=16000, flac=flac, wav=wav)
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
    os.makedirs(os.path.dirname(config["outfile"]), exist_ok=True)
    np.save(config["outfile"], stacked_embeddings)




if __name__ == "__main__":
    print("script running")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config = DATASETS[args.dataset]
    main(config, batch_size=args.batch_size, num_workers=0)
