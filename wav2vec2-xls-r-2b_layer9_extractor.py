import os
import torch
import librosa
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, Wav2Vec2Model
import argparse
from new_config import parquets, feats_dir, feats

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
    def __init__(self, file_list, sr=16000, flac=False, wav=False):
        self.file_list = file_list
        self.sr = sr
        self.flac = flac
        self.wav = wav

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        if self.flac:
            path = path + ".flac"
        if self.wav:
            path = path + ".wav"
        audio, _ = librosa.load(
            path,
            sr=self.sr,      # target sampling rate
            mono=True
        )


        return audio, self.sr, path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import random 
import pandas as pd
from torch.utils.data import DataLoader

def load_label(label_file):
    labels = {}
    wav_lists = []
    encode = {'spoof': 0, 'bonafide': 1}
    with open(label_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) > 1:
                wav_id = line[1]
                wav_lists.append(wav_id)
                try:
                    tmp_label = encode[line[4]]
                except:
                    tmp_label = encode[line[5]]
                labels[wav_id] = tmp_label

    return labels, wav_lists

def load_parquet_data(parquet_paths, dataset_name="dataset"):
    final_data = []
    final_label = []
    ids = []
    
    encode = {'spoof': 0, 'bonafide': 1}
    
    for parquet_path in parquet_paths:
        if not os.path.exists(parquet_path):
            print(f"Parquet file not found: {parquet_path}")
            continue
            
        df = pd.read_parquet(parquet_path)
        print(df)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {os.path.basename(parquet_path)}"):
            wav_id = row['ID']
            wav_path = row['path']
            label_str = row['label']
            
            # Flexibility: try both .wav and .flac
            if not os.path.exists(wav_path):
                base, ext = os.path.splitext(wav_path)
                alt_ext = ".flac" if ext.lower() == ".wav" else ".wav"
                alt_path = base + alt_ext
                if os.path.exists(alt_path):
                    wav_path = alt_path

            if os.path.exists(wav_path):
                ids.append(wav_id)
                final_data.append(wav_path)
                final_label.append(encode[label_str])
            else:
                #print("can not open {}".format(wav_path))
                pass
                
    return ids, final_data, final_label


def collate_fn(batch):
    audios, srs, filenames = zip(*batch)
    return list(audios), list(srs), list(filenames)


def main(parquet_path, feat_path, batch_size=8, num_workers=2):
    metadata_file = parquet_path
    ids, relevant_files, labels = load_parquet_data([parquet_path])
    print(f"Metadata contains {len(relevant_files)} files.")
    
    feature_extractor = HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-xls-r-2b"
    )
    dataset = AudioDataset(relevant_files)
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
    os.makedirs(os.path.dirname(feat_path), exist_ok=True)
    np.save(feat_path, stacked_embeddings)

if __name__ == "__main__":
    print("script running")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    parquet_path = parquets[args.dataset]
    feat_path =  os.path.join(feats_dir, feats[args.dataset])
    main(parquet_path, feat_path, batch_size=args.batch_size, num_workers=0)
