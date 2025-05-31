import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from itertools import combinations
from sklearn.neighbors import NearestCentroid
import os
from config import *

def classify_with_eer_threshold(probs, threshold):
    return np.array([1 if p > threshold else 0 for p in probs])

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

## return the selected samples for each label of each dataset
def cluster_based_pruned(X, y, pruning_fraction, order='ascending'):
    clf = NearestCentroid()
    clf.fit(X, y)
    centroids = clf.centroids_
    distances = []
    for l, label in enumerate(y):
        class_centroid = centroids[int(label)]
        distance = np.linalg.norm(X[l] - class_centroid)  # euclidean distance
        distances.append((distance, l))

    distances = np.array(distances, dtype=[('distance', float), ('index', int)])

    selected_samples = []
    selected_labels = []
    for label in np.unique(y):
        class_distances = distances[y[distances['index']] == label]
        if order == 'ascending':
            sorted_distances = np.sort(class_distances, order='distance')  ## ascending order (close points)
        elif order == 'descending':
            sorted_distances = np.sort(class_distances, order='distance')[::-1]  # descending sorting (further point)
        else:
            raise ValueError(f"{order} is not a valid order, please choose from 'ascending' or 'descending'")

        num_points = int(len(class_distances) * pruning_fraction)
        top_points = sorted_distances[:num_points]['index']
        selected_samples.extend(X[top_points])
        selected_labels.extend(y[top_points])
        ## add the centroid to the already selected data
        selected_samples.append(centroids[int(label)])
        selected_labels.append(label)
    selected_samples = np.array(selected_samples)
    selected_labels = np.array(selected_labels)

    return selected_samples, selected_labels

X_itw, y_itw, _, _ = load_dataset(eval_groups['itw'], meta_dir, metadata, feats_dir, feats)
X_ai4t, y_ai4t, _, _ = load_dataset(eval_groups['ai4trust'], meta_dir, metadata, feats_dir, feats)


pruning_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for pruning_fraction in pruning_fractions:
    print('### PRUNING FRACTION = ',pruning_fraction)
    X_pruned_list = []
    y_pruned_list = []
    for ds_name, idxs in train_groups.items():
        X_ds, y_ds, fnames, dbs = load_dataset(
            indices=idxs,
            meta_dir=meta_dir,
            metadata=metadata,
            feats_dir=feats_dir,
            feats=feats
        )

        Xp, yp = cluster_based_pruned(
            X=X_ds,
            y=y_ds,
            pruning_fraction=pruning_fraction,
            order='descending',
        )
        X_pruned_list.append(Xp)
        y_pruned_list.append(yp)

    train_combined = np.concatenate(X_pruned_list, axis=0)
    train_labels_combined = np.concatenate(y_pruned_list, axis=0)

    print('### TRAIN SET SHAPE ###')
    print(train_combined.shape)
    print(train_labels_combined.shape)
        
    model = LogisticRegression(random_state=42, C=1e6, max_iter=50000, verbose=False)
    model.fit(train_combined, train_labels_combined)

    print("*** Evaluating model...")
    y_val_prob_ITW = model.predict_proba(X_itw)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_itw, y_val_prob_ITW, pos_label=1)
    eer_itw = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    eer_thresh = interp1d(fpr, thresholds)(eer_itw)
    y_val_pred_ITW = classify_with_eer_threshold(y_val_prob_ITW, eer_thresh)
    print(f"ITW validation EER: {eer_itw * 100:.2f}")
    print(f"threshold:{eer_thresh:.2f}")



    y_val_prob_AI4TRUST = model.predict_proba(X_ai4t)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_ai4t, y_val_prob_AI4TRUST, pos_label=1)
    eer_ai4trust = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    eer_thresh = interp1d(fpr, thresholds)(eer_ai4trust)
    y_val_pred_AI4TRUST = classify_with_eer_threshold(y_val_prob_AI4TRUST, eer_thresh)
    print(f"ai4trust validation EER: {eer_ai4trust * 100:.2f}")
    print(f"threshold:{eer_thresh:.2f}")
    with  open('./xls_results_ALL_pruning_cluster.txt',"a") as f:
        f.write(f"\tpruning fraction: {pruning_fraction}, itw_eer:{eer_itw * 100:.2f}, ai4trust_eer:{eer_ai4trust * 100:.2f}"+"\n")
