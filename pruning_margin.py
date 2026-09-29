import numpy as np
import os
from joblib import load, dump
from new_config import *
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
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


def compute_eer(Ytest, Y_hat):
    fpr, tpr, thresholds = roc_curve(Ytest, Y_hat[:, 1], pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = interp1d(fpr, thresholds)(eer)
    return np.round(eer * 100, 2), thresh

def load_dataset(indices, parquets, feats_dir, feats):
    """
    Load features and labels using parquet metadata instead of old CSV format.
    """
    X_all = []
    Y_all = []
    filename_all = []
    dbs_all = []
    path_all = []

    encode = {'spoof': 0, 'bonafide': 1}

    for index in indices:
        parquet_path = parquets[index]
        feat_path = os.path.join(feats_dir, feats[index])

        # ----- Load metadata from parquet -----
        df = pd.read_parquet(parquet_path)
        # Ensure same order as during feature extraction
        ids = df["ID"].tolist()
        labels = df["label"].tolist()
        paths = df["path"].tolist()

        # ----- Load features -----
        X = np.load(feat_path)

        assert len(X) == len(df), \
            f"Feature/metadata mismatch in {feat_path}: {len(X)} vs {len(df)}"

        # ----- Collect -----
        X_all.append(X)

        Y_all.extend([encode[l] for l in labels])
        filename_all.extend(ids)

        # Dataset name from config name
        db_name = df["dataset_name"].tolist()
        dbs_all.extend(db_name)
        path_all.extend(paths)

    X_all = np.vstack(X_all)
    Y_all = np.array(Y_all)

    return X_all, Y_all, filename_all, dbs_all, path_all



def prune_by_margin(
    train_groups,
    eval_groups,
    parquets,
    feats_dir,
    feats,
    margin_percentage,
    strategy,
    steps,
):
    results = []
    train_indices = [i for group in train_groups.values() for i in group]
    X, y, filename, dbs, paths = load_dataset(
        train_indices, parquets, feats_dir, feats
    )
    X_itw, y_itw, _, _, _ = load_dataset(
        eval_groups["itw"], parquets, feats_dir, feats
    )

    margin_total = 0
    model = LogisticRegression(max_iter=10_000, random_state=46, C=1e6)
    model.fit(X, y)
    ## train the logReg with all data before margin pruning
    print("### Fitting baseline logReg")
    Yhat = model.predict_proba(X_itw)
    eer1, thresh = compute_eer(y_itw, Yhat)
    print("Baseline ITW", eer1)



    dump(model, "logreg_baseline.joblib")
    model_path = "logreg_baseline.joblib"
    model = load(model_path)
    print("loaded: ", model_path)
    print("computing margins")
    margins = np.abs(np.dot(X, model.coef_.T) + model.intercept_)  # Absolute margin
    print("starting to prune")
    percent = int(margin_percentage / 100 * X.shape[0])
    print(f"{margin_percentage} percent:", percent)
    print("using: ", pruning_strategy, "pruning")
    for x in range(steps):
        ## remove the closest samples with respect to the hyperplane
        if strategy == "noisy":
            lower_threshold = np.percentile(margins, margin_percentage)
            important_points = np.squeeze(margins >= lower_threshold)
        ## remove the closest and furthest samples with respect to the hyperplane
        elif strategy == "both":
            lower_threshold = np.percentile(
                margins, margin_percentage // 2
            )  ## close to boundary
            upper_threshold = np.percentile(
                margins, 100 - margin_percentage // 2
            )  ## far from boundary
            important_points = np.squeeze(
                (margins >= lower_threshold) & (margins <= upper_threshold)
            )
        else:
            raise ValueError(
                f"invalid pruning strategy: {strategy}, please choose between 'noisy' or 'both'"
            )

        important_index = [i for i, k in enumerate(important_points) if k]
        margin_total += margin_percentage
        parquet_out = f"selected_files_{strategy}_{margin_total}.parquet"

        df_out = pd.DataFrame({
            "ID": [filename[i] for i in important_index],
            "path": [paths[i] for i in important_index],
            "label": [
                "bonafide" if y[i] == 1 else "spoof"
                for i in important_index
            ],
            "dataset_name": [dbs[i] for i in important_index],
        })

        print(f"Writing parquet → {parquet_out}")
        df_out.to_parquet(parquet_out, index=False)
        ## prune dataset
        X_pruned, y_pruned = X[important_points], y[important_points]
        np.save(parquet_out.replace("parquet", "npy"), X_pruned, allow_pickle=True)
        X, y = X_pruned, y_pruned
        filename = [filename[j] for j in important_index]
        dbs = [dbs[j] for j in important_index]
        paths = [paths[j] for j in important_index]
        print(f"number of samples after pruning: {X_pruned.shape[0]}")
        clf = LogisticRegression(max_iter=10_000, random_state=46, C=1e6)
        clf.fit(X_pruned, y_pruned)

        Yhat = clf.predict_proba(X_itw)
        eer1, thresh = compute_eer(y_itw, Yhat)

        print(f"step {steps+1}: ITW, {eer1}, {X_pruned.shape[0]}")
        margins = np.abs(
            np.dot(X_pruned, model.coef_.T) + model.intercept_
        )  # Absolute margin
        margin_percentage = int(percent / X_pruned.shape[0] * 100)
        results.append(
            {
                "step": x + 1,
                "margin": margin_total,
                "eer_itw": eer1,
                "samples": X_pruned.shape[0],
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
        parquets=parquets,
        feats_dir=feats_dir,
        feats=feats,
        margin_percentage=margin_percentage,
        strategy=pruning_strategy,
        steps=10,
    )
    for r in results:
        print(f"Step {r['step']}: EER ITW={r['eer_itw']}%")
