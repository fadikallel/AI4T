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



train_indices = [i for group in train_groups.values() for i in group]

Xtrain,ytrain, _, _ = load_dataset(train_indices, meta_dir=meta_dir, metadata=metadata, feats_dir=feats_dir, feats=feats)

X_itw, y_itw, _, _ = load_dataset(eval_groups['itw'], meta_dir, metadata, feats_dir, feats)
X_ai4t, y_ai4t, _, _ = load_dataset(eval_groups['ai4trust'], meta_dir, metadata, feats_dir, feats)


def classify_with_eer_threshold(probs, threshold):
    return np.array([1 if p > threshold else 0 for p in probs])


## random selection ranging from 10% to 90%
x_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for x_percent_random_selection in x_list:
  nr_of_random = int(x_percent_random_selection * len(Xtrain))
  print("Pruning fraction: ",x_percent_random_selection)
  #get the results from 3 random seed
  for step in range(3):
    

    random_indices = np.random.choice(len(Xtrain), nr_of_random, replace=False)

    selected_samples = [Xtrain[i] for i in random_indices]
    selected_labels = [ytrain[i] for i in random_indices]


#    print(f'nr of samples {len(selected_samples)}')

    model = LogisticRegression(random_state=42, C=1e6, max_iter=50000, verbose=False)
    model.fit(selected_samples, selected_labels)

    print("*** evaluating model...")
    y_val_prob_ITW = model.predict_proba(X_itw)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_itw, y_val_prob_ITW, pos_label=1)
    eer_itw = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    eer_thresh = interp1d(fpr, thresholds)(eer_itw)
    y_val_pred_ITW = classify_with_eer_threshold(y_val_prob_ITW, eer_thresh)
    

    y_val_prob_AI4TRUST = model.predict_proba(X_ai4t)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_ai4t, y_val_prob_AI4TRUST, pos_label=1)
    eer_ai4trust = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    eer_thresh = interp1d(fpr, thresholds)(eer_ai4trust)
    y_val_pred_AI4TRUST = classify_with_eer_threshold(y_val_prob_AI4TRUST, eer_thresh)
    print(f"step {step+1} itw : {eer_itw * 100:.2f} ; ai4trust : {eer_ai4trust * 100:.2f}")


