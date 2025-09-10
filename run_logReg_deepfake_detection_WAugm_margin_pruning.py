import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from config import *
from joblib import dump


def get_baseline_data():
    Ytrain = []
    ## after computing pruning, add the file saved in line 68 in pruning_margin.py
    fpath = os.path.join("selected_files_both_135.txt")
    with open(fpath) as fin:
        for line in fin.readlines():
            label = 1 if line.strip().split("|")[2] == "bonafide" else 0
            Ytrain.append(label)
    Ytrain = np.array(Ytrain)
    Xtrain = np.load("selected_files_both_135.npy")
    print("X pruned:", Xtrain.shape, Ytrain.shape)
    return Xtrain, Ytrain


def compute_eer(Ytest, Y_hat):
    fpr, tpr, thresholds = roc_curve(Ytest, Y_hat[:, 1], pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh


Xtrain, Ytrain = get_baseline_data()

## read augm data
N1 = 1
N2 = 2

## prepare training data
Y = []
for fi in metadata_augm[N1:N2]:
    with open(os.path.join(meta_dir, fi)) as fin:
        for line in fin:
            Y.append(1 if line.strip().split("|")[2] == "bonafide" else 0)

Y.extend(Ytrain)
Y = np.array(Y)
print("Y augmented: ", Y.shape)

X = []
for fi in feats_augm[N1:N2]:
    x = np.load(os.path.join(feats_dir, fi))
    print(fi, x.shape)
    X.extend(x)


l = Xtrain.shape[0]
# Sample half from original, half from rawboost
p1 = np.random.choice(range(l), size=l // 2, replace=False)
p2 = np.random.choice([l + x for x in range(l)], size=l // 2, replace=False)

p = np.hstack((p1, p2))


X.extend(Xtrain)
# np.save("selected_indices_random_augm_ALL.npy", p)
X = [X[i] for i in p]
Y = [Y[i] for i in p]
# X = Xtrain

X = np.array(X)
print("X augmented:", X.shape)
print("Fitting...")
print("Data shape for fitting: ", X.shape)
clf = LogisticRegression(max_iter=10_000, random_state=46, C=1e6)
print(clf)
clf.fit(X, Y)

Yhat_train = clf.decision_function(X)
# np.save("X_Y_best_David_waugm.npy", {"X":X, "Y":Y})
dump(clf, "logreg_margin_pruning_ALL.joblib")
## predict

print("Predicting...")
eval_groups = {
    "itw": [2],
    "ai4trust": [3]
}
for g in eval_groups:
    k = eval_groups[g]
    X, Y = [], []
    filenames = []

    for index in k:
        with open(os.path.join(meta_dir, metadata_augm[index])) as fin:
            for line in fin.readlines():
                Y.append(1 if line.strip().split("|")[1] == "bonafide" else 0)
                filenames.append(line.strip().split("|")[0])

        x = np.load(os.path.join(feats_dir, feats_augm[index]))
        X.extend(x)
    Ytest = np.array(Y)
    Xtest = np.array(X)
    print("\n", g.upper(), Xtest.shape[0])
    Yhat = clf.predict_proba(Xtest)
    eer, thresh = compute_eer(Ytest, Yhat)
    print(Yhat.shape)
    print(f"eer for {g}: {eer * 100:.2f}% | thresh: {thresh:.4f}")

    preds = (Yhat[:, 1] >= thresh).astype(int)
    with open("wrong_predictions_" + g + ".txt", "w") as fout:
        for fn, y_true, y_pred in zip(filenames, Ytest, preds):
            if y_true != y_pred:
                fout.write(f"{fn}\n")
        fout.close()
