import os
from operator import index
import numpy as np
from sklearn.ensemble import BaggingClassifier

from parameters import *
from experiment import experiment, NBBag_experiment
from analysis import *
from datasets import datasets
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score
from imblearn.metrics import geometric_mean_score, specificity_score
from classifiers import clfs
import pandas as pd


# experiment loop
def main_loop(clfs, datasets, metrics, n_repeats, n_splits, RANDOM_STATE):
    for clf_name, clf in clfs.items():
        print(clf_name)
        scores = experiment(clf=clf,
                            datasets=datasets,
                            metrics=metrics,
                            n_repeats=n_repeats,
                            n_splits=n_splits,
                            random_state=RANDOM_STATE)
        np.save(f'./results/{clf_name}', scores)  # save scores in clf_name.npy file


###########################################################################
# ANALYSIS
###########################################################################
def make_clfs_mean_scores(clfs_names, root_dir, metrics_names):
    path = root_dir + "clfs_mean_scores/"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

    clfs_scores = {}
    clfs_mean_scores = {}

    for clf_name in clfs_names:
        clfs_scores[clf_name] = np.load(f'{root_dir}{clf_name}.npy')
        clfs_mean_scores[clf_name] = np.mean(clfs_scores[clf_name], axis=2)

        # saving mean_scores to files (1 classfier = 1 file)
        mean_score = pd.DataFrame(np.round(clfs_mean_scores[clf_name].T, 3), columns=metrics_names, index=datasets)
        mean_score.to_csv(f'{path}{clf_name}.csv', sep='\t')


def make_metric_mean_scores(clfs_names, root_dir, met_comp_dir, metrics_names):
    clfs_scores = {}
    clfs_mean_scores = {}
    met_mean_scores = {}

    for clf_name in clfs_names:
        clfs_scores[clf_name] = np.load(f'{root_dir}{clf_name}.npy')
        clfs_mean_scores[clf_name] = np.mean(clfs_scores[clf_name], axis=2)
        #
        # # saving mean_scores to files (1 classfier = 1 file)
        # mean_score = pd.DataFrame(np.round(clfs_mean_scores[clf_name].T, 3), columns=metrics_names, index=datasets)
        # mean_score.to_csv(f'{root_dir}clfs_mean_scores/{clf_name}.csv', sep='\t')

    # saving metrics to files (1 metric = 1 file)

    path = root_dir + "metric_mean_scores/"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)


    for i, metric_name in enumerate(metrics_names):
        mean_score = np.array([ms[i] for ms in clfs_mean_scores.values()])
        met_mean_scores[metric_name] = mean_score
        mean_score = pd.DataFrame(np.round(mean_score.T, 3), columns=clfs_names, index=datasets)
        mean_score.to_csv(f'{path}{met_comp_dir}{metric_name}.csv', sep='\t')

    # testing different parameters by t-student
    g_mean_score = np.array([s[0] for s in clfs_scores.values()])
    g_mean_scores = np.swapaxes(g_mean_score, 0, 1)  # datasets x metrics x folds

    stat_better_df = []
    for i, g_mean in enumerate(g_mean_scores):
        stat_better, stat_better_list = t_student(g_mean, clfs_names)
        stat_better_df.append(stat_better_list)

    stat_better_df = pd.DataFrame(stat_better_df, columns=clfs_names, index=datasets)
    stat_better_df.to_csv(f'{path}{met_comp_dir}stat_better_table.csv', sep='\t')
    print(stat_better_df.to_string())


# scores = np.load('results.npy')
# print("\nScores:\n", scores.shape)
# # print(scores)
#
# mean of 3rd dim (folds)
# mean_scores = np.mean(scores, axis=2).T
# print("\nMean scores: \n", mean_scores)

# ranks = ranks(mean_scores)
#
# # for i, d_name in enumerate(datasets):
# #     print(f"T-stats for {d_name}")
# #     dateset_scores = scores[:,i]
# #     t_student(dateset_scores)
#
# wilcoxon(ranks)


if __name__ == '__main__':
    # used metrics
    metrics = (
        geometric_mean_score, f1_score, balanced_accuracy_score, precision_score, recall_score, specificity_score)
    metrics_names = ("geometric_mean_score", "f1_score", "balanced_accuracy_score",
                     "precision_score", "recall_score", "specificity_score")

    # metrics_names = ("geometric_mean_score", "specificity_score", "f1_score", "accuracy_score")

    # main_loop(clfs, datasets, metrics, n_repeats, n_splits, RANDOM_STATE)

    # root_dir = "./results/"
    # met_comp_dir = "final_comparison/"
    # clfs_names = [clf_name for clf_name in clfs.keys()]
    #
    # analysis(clfs_names, root_dir, met_comp_dir)

    clfs_names = [
        # "NBBag_k3_fi1_euclidean",
        # "NBBag_k5_fi1_euclidean",
        # "NBBag_k7_fi0.5_euclidean",
        # "NBBag_k7_fi1_chebyshev",
        # "NBBag_k7_fi1_cosine",
        # "NBBag_k7_fi1_euclidean",
        # "NBBag_k7_fi1_manhattan",
        # "NBBag_k7_fi1.5_euclidean",
        # "NBBag_k7_fi2_euclidean",
        # "NBBag_k9_fi1_euclidean",
        # "NBBag_k11_fi1_euclidean",
        # "AdaBoost",
        # "EBBAG",
        # "Bagging",
        # "RUSBoost",
        # "SMOTEAdaBoost"
        "NBBag_k7_fi1_euclidean_glob",
        "NBBag_k7_fi1_euclidean_loc",
    ]
    result_dir = "./results/glob_vs_loc/"

    # make_clfs_mean_scores(clfs_names, result_dir, metrics_names)
    make_metric_mean_scores(clfs_names, result_dir, "final_comparison/", metrics_names)
