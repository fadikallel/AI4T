import os
from tqdm import tqdm
import numpy as np
import torch
import librosa
from transformers import AutoFeatureExtractor, Wav2Vec2Model


class HuggingFaceFeatureExtractor:
    def __init__(self, model_class, name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(name)
        self.model = model_class.from_pretrained(name, output_hidden_states=True)
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, audio, sr):
        inputs = self.feature_extractor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.hidden_states


FEATURE_EXTRACTORS = {
    "wav2vec2-xls-r-2b": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-xls-r-2b"
    ),
}


def read_metadata(file_path):
    relevant_files = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) > 1:
                relevant_files.append(parts[0])
    ## Be careful to the order in which the features are extracted !!
    return relevant_files


def main(outdir, indir, metadata_file):
    relevant_files = read_metadata(metadata_file)
    print(f"Metadata contains {len(relevant_files)} files.")
    feature_extractor = FEATURE_EXTRACTORS["wav2vec2-xls-r-2b"]()

    layer_embedding = []

    for fi in tqdm(relevant_files):
        audio, sr = librosa.load(os.path.join(indir, f'{fi}.flac'), sr=16000)
        hidden_states = feature_extractor(audio, sr)
        layer_output = hidden_states[9]
        ## average pooling on time frame
        mean_layer_output = torch.mean(layer_output, dim=1).cpu().numpy()
        layer_embedding.append(mean_layer_output)

    
    stacked_embeddings = np.vstack(layer_embedding)
    np.save(
        os.path.join(outdir, f"wav2vec2-xls-r-2b_Layer9_asv5_dev.npy"),
        stacked_embeddings,
    )


if __name__ == "__main__":
    print("script running")
    ## location of the wav files
    indir = "/ds-slt/audio/ASVSpoof2024/flac_D/"
    ## location for the saved features
    outdir = "./feats/wav2vec2-xls-r-2b/"
    ## location of the metadata coresponding to the extracted dataset
    metadata_file = "./processed_metadata/asv5_dev_systems.csv"
    main(outdir, indir, metadata_file)
