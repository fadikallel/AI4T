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

def load_add(meta_dir, metadata, feats_dir, feats):
    Xtrain, Ytrain = [], []
    with open(os.path.join(meta_dir, metadata)) as fin:
        for line in fin.readlines():
            label = 1 if line.strip().split(" ")[-1] == "bonafide" else 0
            Ytrain.append(label)
    Ytrain = np.array(Ytrain)
    x = np.load(os.path.join(feats_dir, feats))
    Xtrain.extend(x)
    Xtrain = np.array(Xtrain)
    return Xtrain, Ytrain

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
    #X_itw, y_itw, _, _ = load_dataset(
    #    eval_groups["itw"], meta_dir, metadata, feats_dir, feats
    #)
    #X_ai4t, y_ai4t, _, _ = load_dataset(
    #    eval_groups["ai4trust"], meta_dir, metadata, feats_dir, feats
    #)

    #X_ADD22_track1, y_ADD22_track1 = load_add(meta_dir, "ADD22_track1.txt", feats_dir, "wav2vec2-xls-r-2b_Layer9_ADD22_track1.npy")
    #X_ADD22_track3, y_ADD22_track3 = load_add(meta_dir, "ADD22_track3.txt", feats_dir, "wav2vec2-xls-r-2b_Layer9_ADD22_track3.npy")
    #X_ADD23_round1, y_ADD23_round1 = load_add(meta_dir, "ADD23_round1.txt", feats_dir, "wav2vec2-xls-r-2b_Layer9_ADD23_round1.npy")
    #X_ADD23_round2, y_ADD23_round2 = load_add(meta_dir, "ADD23_round2.txt", feats_dir, "wav2vec2-xls-r-2b_Layer9_ADD23_round2.npy")
    X_infer = np.load('inference.npy')
    margin_total = 0
    model = LogisticRegression(max_iter=10_000, random_state=46, C=1e6)
    model.fit(X, y)
    Yhat_labels = model.predict(X_infer)

    paths = [] 
    for root, _, files in os.walk("/netscratch/fkallel/datacentrictrain/Audiosamples/"):
        for file in sorted(files):
            paths.append(file)

    save_filepath = 'inference_LG.csv'
    with open(save_filepath, 'w') as f:
        for i in range(len(paths)):
            label = "bonafide" if Yhat_labels[i] == 1 else "spoof"
            f.write(f"{paths[i]},{label}\n")

    print(Yhat_labels)

    ## train the logReg with all data before margin pruning
    print("### Fitting baseline logReg")
    #Yhat = model.predict_proba(X_itw)
    #eer1, thresh = compute_eer(y_itw, Yhat)
    #print("Baseline ITW", eer1)

    #Yhat = model.predict_proba(X_ai4t)
    #eer2, thresh = compute_eer(y_ai4t, Yhat)
    #print("Baseline AI4T", eer2)

    #Yhat = model.predict_proba(X_ADD22_track1)
    #eer3, thresh = compute_eer(y_ADD22_track1, Yhat)
    #print("Baseline ADD22_track1", eer3)
    #Yhat = model.predict_proba(X_ADD22_track3)
    #eer4, thresh = compute_eer(y_ADD22_track3, Yhat)
    #print("Baseline ADD22_track3", eer4)
    #Yhat = model.predict_proba(X_ADD23_round1)
    #eer5, thresh = compute_eer(y_ADD23_round1, Yhat)
    #print("Baseline ADD23_round1", eer5)
    #Yhat = model.predict_proba(X_ADD23_round2)
    #eer6, thresh = compute_eer(y_ADD23_round2, Yhat)
    #print("Baseline ADD23_round2", eer6)

    #dump(model, "logreg_baseline.joblib")
    #model_path = "logreg_baseline.joblib"
    #model = load(model_path)
    #print("loaded: ", model_path)
    print("using: ", strategy, "pruning")
    for x, margin_total in enumerate([10,21,33,47,63,82,105,135,178,252]):
        ## prune dataset
        X = np.load(f"selected_files_both_{margin_total}.npy")
        y = load_labels(f"selected_files_both_{margin_total}.txt")
        print(f"number of samples after pruning: {X.shape[0]}")
        clf = LogisticRegression(max_iter=10_000, random_state=46, C=1e6)
        clf.fit(X, y)
        Yhat_labels = clf.predict(X_infer)
        save_filepath = f'inference_LG_{margin_total}.csv'
        with open(save_filepath, 'w') as f:
            for i in range(len(paths)):
                label = "bonafide" if Yhat_labels[i] == 1 else "spoof"
                f.write(f"{paths[i]},{label}\n")

        #Yhat = clf.predict_proba(X_itw)
        #eer1, thresh = compute_eer(y_itw, Yhat)
        #Yhat = clf.predict_proba(X_ai4t)
        #eer2, thresh = compute_eer(y_ai4t, Yhat)
        #Yhat = clf.predict_proba(X_ADD22_track1)
        #eer3, thresh = compute_eer(y_ADD22_track1, Yhat)    
        #Yhat = clf.predict_proba(X_ADD22_track3)
        #eer4, thresh = compute_eer(y_ADD22_track3, Yhat)
        #Yhat = clf.predict_proba(X_ADD23_round1)
        #eer5, thresh = compute_eer(y_ADD23_round1, Yhat)
        #Yhat = clf.predict_proba(X_ADD23_round2)
        #eer6, thresh = compute_eer(y_ADD23_round2, Yhat)
        #print(f"Step {x+1}: EER ITW={eer1}%, AI4T={eer2}%, ADD22_track1={eer3}%, ADD22_track3={eer4}%, ADD23_round1={eer5}%, ADD23_round2={eer6}%")
        #results.append(
        #    {
        #       "step": x + 1,
        #        "margin": margin_total,
        #        "eer_itw": eer1,
        #        "eer_ai4t": eer2,
        #        "samples": X.shape[0],
        #    }
        #)

    return results, clf


if "__main__" == __name__:
    ## config

    model_path = "logreg_allData.joblib"
    pruning_strategy = "both"  ## noisy or both
    margin_percentage = 10
    _, _ = prune_by_margin(
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
