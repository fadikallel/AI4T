import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from itertools import combinations
import os
from config import *
train_groups = {
  'asv19_all': [0, 1, 2],  ## 0,1,2 because asv19 has train+dev+eval
  'for': [3],
  'asv21': [4],
  'timit': [8],
  'odss': [7],
  'mlaad': [11, 12],  ## mlaad + m-ailabs
  'asv5': [5, 6]  ## asv5 train+dev
}
eval_groups = {
  'itw': [9],
  'ai4trust': [10],
}



def load_dataset(indices, meta_dir, metadata, feats_dir, feats):
  Xtrain, Ytrain, filename, dbs = [], [], [], []
  for index in indices:
    with open(os.path.join(meta_dir, metadata[index])) as fin:
      for line in sorted(fin.readlines()):
        label = 1 if line.strip().split('|')[1] == 'bonafide' else 0
        Ytrain.append(label)
        filename.append(line.strip().split('|')[0])
        dbs.append(metadata[index].split('_')[0])
    x = np.load(os.path.join(feats_dir, feats[index]))
    Xtrain.extend(x)

  Ytrain = np.array(Ytrain)
  Xtrain = np.array(Xtrain)
  print("#### X train shape:", Xtrain.shape)

  return Xtrain, Ytrain, filename, dbs



X_asv19, y_asv19, _, _ = load_dataset(train_groups['asv19_all'], meta_dir, metadata, feats_dir, feats)
X_for, y_for, _, _ = load_dataset(train_groups['for'], meta_dir, metadata, feats_dir, feats)
X_asv21, y_asv21 , _, _ = load_dataset(train_groups['asv21'], meta_dir, metadata, feats_dir, feats)
X_timit, y_timit , _, _ = load_dataset(train_groups['timit'], meta_dir, metadata, feats_dir, feats)
X_odss, y_odss, _, _ = load_dataset(train_groups['odss'], meta_dir, metadata, feats_dir, feats)
X_mlaad, y_mlaad , _, _ = load_dataset(train_groups['mlaad'], meta_dir, metadata, feats_dir, feats)
X_asv5, y_asv5, _, _ = load_dataset(train_groups['asv5'], meta_dir, metadata, feats_dir, feats)


X_itw, y_itw, _, _ = load_dataset(eval_groups['itw'], meta_dir, metadata, feats_dir, feats)
X_ai4t, y_ai4t, _, _ = load_dataset(eval_groups['ai4trust'], meta_dir, metadata, feats_dir, feats)




datasets = {
    "ASV19": (X_asv19, y_asv19),
    "ODSS": (X_odss, y_odss),
    "FoR": (X_for, y_for),
    "TIMIT": (X_timit, y_timit),
    "MLAAD": (X_mlaad, y_mlaad),
    "ASV5": (X_asv5, y_asv5),
    "ASV21": (X_asv21, y_asv21),
}

##permanent_element = {"ASV19": (np.concatenate((train1_layer, train2_layer)), np.concatenate((train_labels1, train_labels2)))}

dataset_names = list(datasets.keys())

all_combinations = []

for r in range(1, len(dataset_names) + 1):
    combinations_list = list(combinations(dataset_names, r))
    all_combinations.extend(combinations_list)

combined_data = []
for combination in all_combinations:
    combination_with_data = [(dataset, datasets[dataset]) for dataset in combination]
    ##combination_with_data.append(("ASV19", permanent_element["ASV19"]))
    combined_data.append(combination_with_data)

print(f'Total number of possible combinations is: {len(combined_data)}')

def classify_with_eer_threshold(probs, threshold):
    return np.array([1 if p > threshold else 0 for p in probs])

def write_metrics_to_file(metrics, filename="results.txt"):
    with open(filename, "a") as f:
        f.write(metrics + "\n")

for combination in combined_data:
    print(f"Training with datasets: {[comb[0] for comb in combination]}")

    train_combined = np.concatenate([comb[1][0] for comb in combination])
    train_labels_combined = np.concatenate([comb[1][1] for comb in combination])
    print(train_combined.shape)
    print(train_labels_combined.shape)

    model = LogisticRegression(random_state=42, C=1e6, max_iter=50000, verbose=False)
    model.fit(train_combined, train_labels_combined)

    print("*** Evaluating model...")
    y_val_prob_ITW = model.predict_proba(X_itw)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_itw, y_val_prob_ITW, pos_label=1)
    fnr = 1 - tpr
    eer_itw = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    eer_thresh = interp1d(fpr, thresholds)(eer_itw)
    y_val_pred_ITW = classify_with_eer_threshold(y_val_prob_ITW, eer_thresh)
    print(f"ITW validation EER: {eer_itw * 100:.2f}")
    print(f"threshold:{eer_thresh:.2f}")


    y_val_prob_AI4TRUST = model.predict_proba(X_ai4t)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_ai4t, y_val_prob_AI4TRUST, pos_label=1)
    fnr = 1 - tpr
    eer_ai4trust = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    eer_thresh = interp1d(fpr, thresholds)(eer_ai4trust)
    y_val_pred_AI4TRUST = classify_with_eer_threshold(y_val_prob_AI4TRUST, eer_thresh)
    print(f"AI4TRUST validation EER: {eer_ai4trust * 100:.2f}")
    print(f"threshold:{eer_thresh:.2f}")
    with  open('./results.txt',"a") as f:
        f.write(f"Datasets: {[comb[0] for comb in combination]}, ITW_eer:{eer_itw * 100:.2f}, AI4TRUST_eer:{eer_ai4trust * 100:.2f}" + "\n")



