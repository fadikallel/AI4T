import numpy as np
import os
from joblib import load, dump
from config import *
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression



def compute_eer(Ytest, Y_hat):
    fpr, tpr, thresholds = roc_curve(Ytest, Y_hat[:, 1], pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = interp1d(fpr, thresholds)(eer)
    return np.round(eer * 100, 2), thresh

def load_dataset(indices, meta_dir, metadata, feats_dir, feats):
    Xtrain, Ytrain, filename, dbs = [], [], [], []
    for index in indices:
        with open(os.path.join(meta_dir, metadata[index])) as fin:
            for line in fin.readlines():
                label = 1 if line.strip().split("|")[1] == "bonafide" else 0
                Ytrain.append(label)
                filename.append(line.strip().split("|")[0])
                dbs.append(metadata[index].split("_")[0])
        x = np.load(os.path.join(feats_dir, feats[index]))
        Xtrain.extend(x)

    Ytrain = np.array(Ytrain)
    Xtrain = np.array(Xtrain)
    return Xtrain, Ytrain, filename, dbs


def load_labels(metadata):
    Ytrain = []
    with open(metadata) as fin:
        for line in fin.readlines():
            label = 1 if line.strip().split("|")[2] == "bonafide" else 0
            Ytrain.append(label)
    Ytrain = np.array(Ytrain)
    return  Ytrain

def prune_by_margin(
    train_groups,
    eval_groups,
    meta_dir,
    metadata,
    feats_dir,
    feats,
    margin_percentage,
    strategy,
    steps,
):
    results = []
    train_indices = [i for group in train_groups.values() for i in group]
    X, y, filename, dbs = load_dataset(
        train_indices, meta_dir, metadata, feats_dir, feats
    )
    X_itw, y_itw, _, _ = load_dataset(
        eval_groups["itw"], meta_dir, metadata, feats_dir, feats
    )
    X_ai4t, y_ai4t, _, _ = load_dataset(
        eval_groups["ai4trust"], meta_dir, metadata, feats_dir, feats
    )
    margin_total = 0
    model = LogisticRegression(max_iter=10_000, random_state=46, C=1e6)
    model.fit(X, y)
    ## train the logReg with all data before margin pruning
    print("### Fitting baseline logReg")
    Yhat = model.predict_proba(X_itw)
    eer1, thresh = compute_eer(y_itw, Yhat)
    print("Baseline ITW", eer1)

    Yhat = model.predict_proba(X_ai4t)
    eer2, thresh = compute_eer(y_ai4t, Yhat)
    print("Baseline AI4T", eer2)

    dump(model, "logreg_baseline.joblib")
    model_path = "logreg_baseline.joblib"
    model = load(model_path)
    print("loaded: ", model_path)
    print("using: ", strategy, "pruning")
    for x, margin_total in enumerate([10,21,33,47,63,82,105,135,178,252]):
        ## prune dataset
        X = np.load(f"selected_files_both_{x}.npy")
        y = load_labels(f"selected_files_both_{x}.txt")
        print(f"number of samples after pruning: {X.shape[0]}")
        clf = LogisticRegression(max_iter=10_000, random_state=46, C=1e6)
        clf.fit(X, y)

        Yhat = clf.predict_proba(X_itw)
        eer1, thresh = compute_eer(y_itw, Yhat)
        Yhat = clf.predict_proba(X_ai4t)
        eer2, thresh = compute_eer(y_ai4t, Yhat)

        results.append(
            {
                "step": x + 1,
                "margin": margin_total,
                "eer_itw": eer1,
                "eer_ai4t": eer2,
                "samples": X.shape[0],
            }
        )

    return results, clf


if "__main__" == __name__:
    ## config

    model_path = "logreg_allData.joblib"
    pruning_strategy = "both"  ## noisy or both
    margin_percentage = 10

    results, _ = prune_by_margin(
        train_groups=train_groups,
        eval_groups=eval_groups,
        meta_dir=meta_dir,
        metadata=metadata,
        feats_dir=feats_dir,
        feats=feats,
        margin_percentage=margin_percentage,
        strategy=pruning_strategy,
        steps=10,
    )
    for r in results:
        print(f"Step {r['step']}: EER ITW={r['eer_itw']}%, AI4T={r['eer_ai4t']}%")
