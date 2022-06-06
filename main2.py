from operator import index
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from parameters import *
from experiment import experiment, NBBag_experiment
from analysis import *
from datasets import datasets
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score
from imblearn.metrics import geometric_mean_score, specificity_score
from classifiers import clfs
import pandas as pd
from imblearn.pipeline import Pipeline

BASE_CLF = DecisionTreeClassifier(random_state=RANDOM_STATE)

metrics = (
    geometric_mean_score, f1_score, balanced_accuracy_score, precision_score, recall_score, specificity_score)

# experiment loop
for k in [3, 5, 7, 9, 11]:
    for fi in [1]:
        for dist_metric in ["euclidean"]:
            clf_name = f"NBBag_k{k}_fi{fi}_{dist_metric}"
            print("\n" + clf_name)
            clf = BaggingClassifier(base_estimator=BASE_CLF, n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
            scores = NBBag_experiment(clf=clf,
                                      datasets=datasets,
                                      metrics=metrics,
                                      n_repeats=n_repeats,
                                      n_splits=n_splits,
                                      random_state=RANDOM_STATE,
                                      k_neighbours=k,
                                      fi=fi,
                                      dist_metric=dist_metric)
            np.save(f'./results/NBBag_global_weights/{clf_name}', scores)  # save scores in clf_name.npy file

for k in [7]:
    for fi in [0.5, 1.5, 2]:
        for dist_metric in ["euclidean"]:
            clf_name = f"NBBag_k{k}_fi{fi}_{dist_metric}"
            print("\n" + clf_name)
            clf = BaggingClassifier(base_estimator=BASE_CLF, n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
            scores = NBBag_experiment(clf=clf,
                                      datasets=datasets,
                                      metrics=metrics,
                                      n_repeats=n_repeats,
                                      n_splits=n_splits,
                                      random_state=RANDOM_STATE,
                                      k_neighbours=k,
                                      fi=fi,
                                      dist_metric=dist_metric)
            np.save(f'./results/NBBag_global_weights/{clf_name}', scores)  # save scores in clf_name.npy file

for k in [7]:
    for fi in [1]:
        for dist_metric in ["cosine", "manhattan", "chebyshev"]:
            clf_name = f"NBBag_k{k}_fi{fi}_{dist_metric}"
            print("\n" + clf_name)
            clf = BaggingClassifier(base_estimator=BASE_CLF, n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
            scores = NBBag_experiment(clf=clf,
                                      datasets=datasets,
                                      metrics=metrics,
                                      n_repeats=n_repeats,
                                      n_splits=n_splits,
                                      random_state=RANDOM_STATE,
                                      k_neighbours=k,
                                      fi=fi,
                                      dist_metric=dist_metric)
            np.save(f'./results/NBBag_global_weights/{clf_name}', scores)  # save scores in clf_name.npy file
