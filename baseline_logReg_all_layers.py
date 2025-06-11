import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from joblib import dump
from config import *


def compute_eer(Ytest, Y_hat):
    fpr, tpr, thresholds = roc_curve(Ytest, Y_hat, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh


def load_dataset(indices, meta_dir, metadata, feats_dir, feats):
    X, Y, filenames, dbs = [], [], [], []
    for index in indices:
        with open(os.path.join(meta_dir, metadata[index])) as fin:
            lines = fin.readlines()
            labels = [1 if line.strip().split("|")[1] == "bonafide" else 0 for line in lines]
            files = [line.strip().split("|")[0] for line in lines]
            dbname = metadata[index].split("_")[0]

            Y.extend(labels)
            filenames.extend(files)
            dbs.extend([dbname] * len(labels))

        x = np.load(os.path.join(feats_dir, feats[index]))
        X.extend(x)

    return np.array(X), np.array(Y), filenames, dbs

## indices for baseline train and eval data
train_indices = train_groups['asv19_train'] + train_groups['asv19_dev']
all_eval_groups = {
    **{k: v for k, v in train_groups.items() if k not in ['asv19_train', 'asv19_dev']},
    **eval_groups,
}

## iterate thorugh all layers
for layer in range(49):
    print(f" Using layer {layer}...")

    ## all layers feats for all datasets
    feats = [
        f"wav2vec2-xls-r-2b_Layer{layer}_ai4trust.npy",
        f"wav2vec2-xls-r-2b_Layer{layer}_asv19_train.npy",
        f"wav2vec2-xls-r-2b_Layer{layer}_asv19_dev.npy",
        f"wav2vec2-xls-r-2b_Layer{layer}_asv19_eval.npy",
        f"wav2vec2-xls-r-2b_Layer{layer}_asv21.npy",
        f"wav2vec2-xls-r-2b_Layer{layer}_asv5_train.npy",
        f"wav2vec2-xls-r-2b_Layer{layer}_asv5_dev.npy",
        f"wav2vec2-xls-r-2b_Layer{layer}_for.npy",
        f"wav2vec2-xls-r-2b_Layer{layer}_itw.npy",
        f"wav2vec2-xls-r-2b_Layer{layer}_m-ailabs.npy",
        f"wav2vec2-xls-r-2b_Layer{layer}_mlaad.npy",
        f"wav2vec2-xls-r-2b_Layer{layer}_odss.npy",
        f"wav2vec2-xls-r-2b_Layer{layer}_timit.npy",
    ]


    X_train, Y_train, _, _ = load_dataset(train_indices, meta_dir, metadata, feats_dir, feats)
    model = LogisticRegression(random_state=42, C=1e6, max_iter=50000, verbose=False)
    model.fit(X_train, Y_train)
    #dump(model, f"logreg_baseline_Layer{layer}.joblib")

    for group_name, group_indices in all_eval_groups.items():
        X_eval, Y_eval, _, _ = load_dataset(group_indices, meta_dir, metadata, feats_dir, feats)
        Y_hat = model.predict_proba(X_eval)[:, 1]
        eer, threshold = compute_eer(Y_eval, Y_hat)
        print(f"[{group_name:<20}] EER: {eer * 100:.1f}")
