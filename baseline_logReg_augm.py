import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from joblib import dump
from config import *
from sklearn.utils import shuffle

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


all_eval_groups = {
    **{k: v for k, v in train_groups.items() if k not in ['asv19_train', 'asv19_dev']},
    **eval_groups,
}


X_asv19_train, Y_asv19_train, _, _ = load_dataset(train_groups['asv19_train'], meta_dir, metadata, feats_dir, feats)
X_asv19_dev, Y_asv19_dev, _, _ = load_dataset(train_groups['asv19_dev'], meta_dir, metadata, feats_dir, feats)

asv19_train_length = len(Y_asv19_train)
asv19_dev_length = len(Y_asv19_dev)

## non-augmented train+dev
X_train = np.concatenate([X_asv19_train, X_asv19_dev])
Y_train = np.concatenate([Y_asv19_train, Y_asv19_dev])

## load the augmented features data
X_asv19_train_rb = np.load(os.path.join(feats_dir, asv19_augm[0]))
X_asv19_dev_rb = np.load(os.path.join(feats_dir, asv19_augm[1]))
X_asv19_train_codecs = np.load(os.path.join(feats_dir, asv19_augm[2]))
X_asv19_dev_codecs = np.load(os.path.join(feats_dir, asv19_augm[3]))

## get half codecs and half rb augmentation from each of train and dev partitions with their associated labels:
X_train_rb = X_asv19_train_rb[:asv19_train_length]
Y_train_rb = Y_asv19_train[:asv19_train_length]
X_train_codecs = X_asv19_train_codecs[:asv19_train_length]
Y_train_codecs = Y_asv19_train[:asv19_train_length]

X_dev_rb = X_asv19_dev_rb[:asv19_dev_length]
Y_dev_rb = Y_asv19_dev[:asv19_dev_length]
X_dev_codecs = X_asv19_dev_codecs[:asv19_dev_length]
Y_dev_codecs = Y_asv19_dev[:asv19_dev_length]

## combine the codecs and rb augmented data and labels
X_rb = np.concatenate([X_asv19_train_rb, X_asv19_dev_rb])
Y_rb = np.concatenate([Y_asv19_train, Y_asv19_dev])
X_codecs = np.concatenate([X_asv19_train_codecs, X_asv19_dev_codecs])
Y_codecs = np.concatenate([Y_asv19_train, Y_asv19_dev])
X_augm = np.concatenate([X_rb, X_codecs])
Y_augm = np.concatenate([Y_rb, Y_codecs])




## combine the non-augmented with the augmented features to form the training set
X = np.concatenate([X_train, X_augm])
Y = np.concatenate([Y_train, Y_augm])



model = LogisticRegression(random_state=42, C=1e6, max_iter=50000, verbose=False)
model.fit(X, Y)
#dump(model, f"logreg_baseline_augmented_Layer{layer}.joblib")

for group_name, group_indices in all_eval_groups.items():
    X_eval, Y_eval, _, _ = load_dataset(group_indices, meta_dir, metadata, feats_dir, feats)
    Y_hat = model.predict_proba(X_eval)[:, 1]
    eer, threshold = compute_eer(Y_eval, Y_hat)
    print(f"[{group_name:<20}] EER: {eer * 100:.1f}")