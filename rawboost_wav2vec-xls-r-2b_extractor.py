import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset
from librosa import effects
import random
from RawBoost import ISD_additive_noise, LnL_convolutive_noise, SSI_additive_noise, normWav

import os
import sys
from tqdm import tqdm
import numpy as np
import torch
import librosa
from transformers import AutoFeatureExtractor, Wav2Vec2Model, Wav2Vec2BertConfig, Wav2Vec2BertModel




# --------------RawBoost data augmentation algorithms---------------------------##

def process_Rawboost_feature(feature, sr, args, algo):
    # Data process by Convolutive noise (1st algo)
    if algo == 1:

        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
                                        args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin,
                                        args.maxBiasLinNonLin, sr)

    # Data process by Impulsive noise (2nd algo)
    elif algo == 2:

        feature = ISD_additive_noise(feature, args.P, args.g_sd)

    # Data process by coloured additive noise (3rd algo)
    elif algo == 3:

        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF, args.minBW,
                                     args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)

    # Data process by all 3 algo. together in series (1+2+3)
    elif algo == 4:

        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
                                        args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin,
                                        args.maxBiasLinNonLin, sr)
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands, args.minF,
                                     args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG,
                                     args.maxG, sr)

        # Data process by 1st two algo. together in series (1+2)
    elif algo == 5:

        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
                                        args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin,
                                        args.maxBiasLinNonLin, sr)
        feature = ISD_additive_noise(feature, args.P, args.g_sd)

        # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo == 6:

        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
                                        args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin,
                                        args.maxBiasLinNonLin, sr)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF, args.minBW,
                                     args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)

        # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo == 7:

        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF, args.minBW,
                                     args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)

        # Data process by 1st two algo. together in Parallel (1||2)
    elif algo == 8:

        feature1 = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
                                         args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin,
                                         args.maxBiasLinNonLin, sr)
        feature2 = ISD_additive_noise(feature, args.P, args.g_sd)

        feature_para = feature1 + feature2
        feature = normWav(feature_para, 0)  # normalized resultant waveform

    # original data without Rawboost processing
    else:

        feature = feature

    return feature


def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
            _, key, _, _, label = line.strip().split(' ')
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list

    elif (is_eval):
        for line in l_meta:
            key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(' ')
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list


# def pad(x, max_len=64600):
#     x_len = x.shape[0]
#     if x_len >= max_len:
#         return x[:max_len]
#     # need to pad
#     num_repeats = int(max_len / x_len) + 1
#     padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
#     return padded_x


class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir, algo):
        '''self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)'''

        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.algo = algo
        self.args = args
        #self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir + 'flac/' + key + '.flac', sr=16000)
        Y = process_Rawboost_feature(X, fs, self.args, self.algo)
        # X_pad = pad(Y, self.cut)
        # x_inp = Tensor(X_pad)
        y = self.labels[key]

        return Y, y


class Dataset_ASVspoof2021_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
           '''

        self.list_IDs = list_IDs
        self.base_dir = base_dir
        #self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir + 'flac/' + key + '.flac', sr=16000)
        # X_pad = pad(X, self.cut)
        # x_inp = Tensor(X_pad)
        return X, key




class Wav2Vec2Truncated(Wav2Vec2Model):
    def __init__(self, config):
        super().__init__(config)
        self.encoder.layers = self.encoder.layers[:9]


model_name = "facebook/wav2vec2-xls-r-2b"
model = Wav2Vec2Truncated.from_pretrained(model_name)


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
        return outputs.hidden_states[9]


FEATURE_EXTRACTOR = {
    "wav2vec2-xls-r-2b": lambda: HuggingFaceFeatureExtractor(Wav2Vec2Truncated, "facebook/wav2vec2-xls-r-2b")}


def read_metadata(file_path):
    relevant_files = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) > 1:
                relevant_files.append(parts[0])
    ## Be careful to the order in which the features are extracted !!
    return relevant_files


def main(outdir,indir, metadata_file, args,algo):
    relevant_files = read_metadata(metadata_file)
    print(f"Metadata contains {len(relevant_files)} files.")
    model_name = 'wav2vec2-xls-r-2b'
    feature_extractor = FEATURE_EXTRACTOR[model_name]()

    layer_embeddings = []

    for fi in tqdm(relevant_files):
            fi = f'{os.path.join(indir,fi)}'
            audio, sr = librosa.load(fi, sr=16000)
            #algo = random.randint(1,24)
            audio = process_Rawboost_feature(audio, sr, args, algo)
            hidden_states = feature_extractor(audio, sr)
            layer_output = hidden_states[layer_idx]
            mean_layer_output = torch.mean(layer_output, dim=1).cpu().numpy()
            layer_embeddings[layer_idx].append(mean_layer_output)
    stacked_embeddings = np.vstack(layer_embeddings)
    np.save(os.path.join(outdir, f'{model_name}_feats_rawboost_for.npy'), stacked_embeddings)


if __name__ == '__main__':
    print('script running')
    ## location of the wav files    
    indir = './DATA/FoR/'
    ## location for the saved features
    outdir = './feats/wav2vec2-xls-r-2b/'
    ## location of the metadata coresponding to the extracted dataset
    metadata_file = './processed_metadata/for_systems.csv'


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

    main(outdir,indir, metadata_file,args, algo=5)
