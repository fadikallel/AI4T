import os
from tqdm import tqdm
import numpy as np
import torch
import librosa
from transformers import AutoFeatureExtractor, Wav2Vec2Model, Wav2Vec2BertModel
import subprocess
import random


def augment_audio(input_file):
    codec = random.choice(["aac", "opus"])
    if codec == "aac":
        bitrate = "320k"
        temp_output = "/mnt/QNAP/comdav/temp/temp_output.aac"
    else:
        bitrate = "128k"
        temp_output = "/mnt/QNAP/comdav/temp/temp_output.ogg"
    # print('chosen codec: ',codec)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_file,
        "-acodec",
        codec,
        "-b:a",
        bitrate,
        "-strict",
        "-2",
        "-loglevel",
        "quiet",
        temp_output,
    ]
    subprocess.run(command, check=True)

    return temp_output


## truncate the SSL model starting from the best layer+1 for faster inference
class Wav2Vec2Truncated(Wav2Vec2Model):
    def __init__(self, config):
        super().__init__(config)
        self.encoder.layers = self.encoder.layers[:9]


model = Wav2Vec2Truncated.from_pretrained("facebook/wav2vec2-xls-r-2b")


class HuggingFaceFeatureExtractor:
    def __init__(self, model_class, name=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(name)
        self.model = model_class.from_pretrained(name, output_hidden_states=True)
        self.model.eval()
        self.model.to(self.device)
        print(self.model)

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
        return outputs.hidden_states[9]  ## the layer from which the feats are extracted


FEATURE_EXTRACTORS = {
    "wav2vec2-xls-r-1b": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-xls-r-1b"
    ),
    "wav2vec2-bert": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2BertModel, "facebook/w2v-bert-2.0"
    ),
    "wav2vec2-xls-r-2b": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Truncated, "facebook/wav2vec2-xls-r-2b"
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


def extract_features_codec(outdir, indir, metadata_file):
    relevant_files = read_metadata(metadata_file)
    print(f"Metadata contains {len(relevant_files)} files.")
    model_name = "wav2vec2-xls-r-2b"
    feature_extractor = FEATURE_EXTRACTORS[model_name]()

    layer_embeddings = []
    for fi in tqdm(relevant_files):
        fi = f"{os.path.join(indir,fi)}"
        augmented_audio_path = augment_audio(fi)
        augmented_audio, sr = librosa.load(augmented_audio_path, sr=16000)
        hidden_states = feature_extractor(augmented_audio, sr)
        mean_layer_output = torch.mean(hidden_states, dim=1).cpu().numpy()
        layer_embeddings.append(mean_layer_output)
    stacked_embeddings = np.vstack(layer_embeddings)
    np.save(
        os.path.join(outdir, f"wav2vec2-xls-r-2b_asv19_train_augm_codecs_Layer9.npy"), stacked_embeddings
    )


if __name__ == "__main__":
    print("script running")
    ## location of the wav files
    indir = "./DATA/asv19/"
    ## location for the saved features
    outdir = "./feats/wav2vec2-xls-r-2b/"
    ## location of the metadata coresponding to the extracted dataset
    metadata_file = "./processed_metadata/asv19_train_systems.csv"
    extract_features_codec(outdir, indir, metadata_file)
