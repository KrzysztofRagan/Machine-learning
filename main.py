import os
from operator import index
import numpy as np
from sklearn.ensemble import BaggingClassifier
from collections import Counter
from matplotlib import pyplot
from numpy import where

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

class Analysis:
    pass


def make_clfs_mean_scores(clfs_names, root_dir, metrics_names):
    path = root_dir + "clfs_mean_scores/"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

    clfs_scores = {}
    clfs_mean_scores = {}

    for clf_name in clfs_names:
        clfs_scores[clf_name] = np.load(f'{root_dir}scores/{clf_name}.npy')
        clfs_mean_scores[clf_name] = np.mean(clfs_scores[clf_name], axis=2)

        # saving mean_scores to files (1 classfier = 1 file)
        mean_score = pd.DataFrame(np.round(clfs_mean_scores[clf_name].T, 3), columns=metrics_names, index=datasets)
        mean_score.to_csv(f'{path}{clf_name}.csv', sep='\t')


def make_metric_mean_scores(clfs_names, root_dir, met_comp_dir, metrics_names):
    clfs_scores = {}
    clfs_mean_scores = {}
    met_mean_scores = {}

    for clf_name in clfs_names:
        clfs_scores[clf_name] = np.load(f'{root_dir}scores/{clf_name}.npy')
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

    wilcoxon_table = []
    friedmann_stats = []
    for i, metric_name in enumerate(metrics_names):
        mean_score = np.array([ms[i] for ms in clfs_mean_scores.values()])
        wilcoxon_stat_better, wilcoxon_list = wilcoxon(mean_score.T, clfs_names)
        mean_ranks, s, p = friedmann(mean_score.T)
        friedmann_stats.append([s, p])
        mean_ranks = pd.DataFrame([mean_ranks], columns=clfs_names)

        mean_ranks.to_csv(f'{path}{met_comp_dir}{metric_name}_mean_ranks.csv', sep='\t', index=False)
        met_mean_scores[metric_name] = mean_score
        mean_score = pd.DataFrame(np.round(mean_score.T, 3), columns=clfs_names, index=datasets)
        mean_score.to_csv(f'{path}{met_comp_dir}{metric_name}_scores.csv', sep='\t')

        # testing different parameters by t-student
        g_mean_score = np.array([s[i] for s in clfs_scores.values()])
        g_mean_scores = np.swapaxes(g_mean_score, 0, 1)  # datasets x metrics x folds

        t_student_table = []
        for i, g_mean in enumerate(g_mean_scores):
            t_student_stat_better, t_student_list = t_student(g_mean, clfs_names)
            t_student_table.append(t_student_list)

        wilcoxon_table.append(wilcoxon_list)
        t_student_table = pd.DataFrame(t_student_table, columns=clfs_names, index=datasets)
        t_student_table.to_csv(f'{path}{met_comp_dir}{metric_name}_t_student_table.csv', sep='\t')
        # print(t_student_table.to_string())

    friedmann_stats = pd.DataFrame(friedmann_stats, columns=["stat", "p-value"], index=metrics_names)
    friedmann_stats.to_csv(f'{path}{met_comp_dir}friedmann_test.csv', sep='\t')
    wilcoxon_table = pd.DataFrame(wilcoxon_table, columns=clfs_names, index=metrics_names)
    wilcoxon_table.to_csv(f'{path}{met_comp_dir}wilcoxon_table.csv', sep='\t')


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
    metrics = {"G-mean": geometric_mean_score,
               "F1": f1_score,
               "BAC": balanced_accuracy_score,
               "Precision": precision_score,
               "Recall": recall_score,
               "Specificity": specificity_score}

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
        # "NBBag_k3_fi2_chebyshev",
        # "NBBag_k5_fi2_chebyshev",
        # "NBBag_k7_fi2_chebyshev",
        # "NBBag_k9_fi2_chebyshev",
        # "NBBag_k11_fi2_chebyshev",
        # "NBBag_k7_fi1_chebyshev",
        # # "NBBag_k7_fi1_cosine",
        # "NBBag_k7_fi1_euclidean",
        # "NBBag_k7_fi1_manhattan",
        # "NBBag_k7_fi1.5_euclidean",
        # # "NBBag_k7_fi2_euclidean",
        # "NBBag_k9_fi1_euclidean",
        # "NBBag_k11_fi1_euclidean",

        # dist comparison
        # "NBBag_k7_fi1_chebyshev",
        # "NBBag_k7_fi1_cosine",
        # "NBBag_k7_fi1_euclidean",
        # "NBBag_k7_fi1_manhattan",

        # fi comparison
        # "NBBag_k7_fi0.5_euclidean",
        # "NBBag_k7_fi1_euclidean",
        # "NBBag_k7_fi1.5_euclidean",
        # "NBBag_k7_fi2_euclidean",

        # k comparison
        # "NBBag_k3_fi1_euclidean",
        # "NBBag_k5_fi1_euclidean",
        # "NBBag_k7_fi1_euclidean",
        # "NBBag_k9_fi1_euclidean",
        # "NBBag_k11_fi1_euclidean",

        # final comparison
        "NBBag_k7_fi2_chebyshev",
        "AdaBoost",
        "EBBAG",
        "Bagging",
        "RUSBoost",
        "SMOTEAdaBoost"

        # final2 comparison
        # "NBBag_k7_fi1_chebyshev",
        # "AdaBoost",
        # "EBBAG",
        # "Bagging",
        # "RUSBoost",
        # "SMOTEAdaBoost"

        # "NBBag_k7_fi1_euclidean_glob",
        # "NBBag_k7_fi1_euclidean_loc",
    ]
    # result_dir = "./results/NBBag_global_weights/"

    # make_clfs_mean_scores(clfs_names, result_dir, metrics.keys())
    # make_metric_mean_scores(clfs_names, result_dir, "final_comparison/", metrics.keys())


    for data_id, dataset_name in enumerate(datasets):
        print(f'dataset: {dataset_name}')
        dataset = np.genfromtxt("datasets/%s.dat" % dataset_name, delimiter=",", comments='@',
                                converters={(-1): lambda s: 0.0 if (s.strip().decode('ascii')) == 'negative' else 1.0})
        X = dataset[:, :-1]
        y = dataset[:, -1].astype(int)

        counter = Counter(y)
        print(counter)
        # scatter plot of examples by class label
        for label, _ in counter.items():
            row_ix = where(y == label)[0]
            pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
        pyplot.title(dataset_name)
        pyplot.legend()
        pyplot.show()
