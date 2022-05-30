import numpy as np
from scipy.stats import rankdata
from scipy.stats import ranksums
from classifiers import clfs
from tabulate import tabulate
from scipy.stats import ttest_rel
from datasets import datasets


# # loading classifiers results
# scores = np.load('results.npy')
# print("\nScores:\n", scores.shape)
# # print(scores)
#
# # mean of 3rd dim (folds)
# mean_scores = np.mean(scores, axis=2).T
# print("\nMean scores: \n", mean_scores)

def ranks(mean_scores):
    '''
    Gives ranks for data in each row of mean_scores

    :param mean_scores: matrix classifiers x datasets
    :type mean_scores: list
    :return: ranks: list of ranks for each row in mean_scores
    :rtype:
    '''
    ranks = []
    for ms in mean_scores:
        ranks.append(rankdata(ms).tolist())
    ranks = np.array(ranks)
    print("\nRanks:\n", ranks)
    return ranks


# ---------------- TEST FRIEDMANNA ------------------
def friedmann(ranks):
    '''
    Calculate average of the ranks

    :param ranks:
    :type ranks:
    :return:
    :rtype:
    '''
    print("\n------------ TEST FRIEDMANNA ------------")
    mean_ranks = np.mean(ranks, axis=0)
    print("\nMean ranks:\n", mean_ranks)


# ---------------- TEST T-STUDENTA ------------------
def t_student(dataset_scores):
    '''
    Oblicza tabele porównawczą klasyfikatorów dla podanej tablicy

    :param dataset_scores: columns: classifiers, rows: folds
    :type dataset_scores:
    :return:
    :rtype:
    '''
    print("\n------------ TEST T-STUDENTA ------------")
    # mean_scores = np.mean(scores, axis=2)
    # print("\nMean scores: \n", mean_scores)
    # print("\nScores\n", scores[:,0])

    # for i,d_name in enumerate(datasets):
    # print(f"T-stats for {d_name}")
    # dateset_scores = scores[:,i]
    alfa = .05
    t_statistic = np.zeros((len(clfs), len(clfs)))
    p_value = np.zeros((len(clfs), len(clfs)))

    for i in range(len(clfs)):
        for j in range(len(clfs)):
            t_statistic[i, j], p_value[i, j] = ttest_rel(dataset_scores[i], dataset_scores[j])
    # print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)

    headers = list(clfs.keys())
    names_column = np.expand_dims(np.array(list(clfs.keys())), axis=1)
    t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    # print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

    advantage = np.zeros((len(clfs), len(clfs)))
    advantage[t_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    # print("\nAdvantage:\n", advantage_table)

    significance = np.zeros((len(clfs), len(clfs)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    # print("\nStatistical significance (alpha = 0.05):\n", significance_table)

    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate(
        (names_column, stat_better), axis=1), headers)
    print("Statistically significantly better:\n", stat_better_table)


# ---------------- TEST WILCOXONA ------------------

def wilcoxon(ranks):
    '''
    Wilcoxon test

    :param ranks: matrix of ranks for each dataset; colums: classifiers, rows: datasets
    :type ranks:
    :return:
    :rtype:
    '''
    print("\n------------ TEST WILCOXONA ------------")
    alfa = .05
    w_statistic = np.zeros((len(clfs), len(clfs)))
    p_value = np.zeros((len(clfs), len(clfs)))

    for i in range(len(clfs)):
        for j in range(len(clfs)):
            w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])

    headers = list(clfs.keys())
    names_column = np.expand_dims(np.array(list(clfs.keys())), axis=1)
    w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
    w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

    advantage = np.zeros((len(clfs), len(clfs)))
    advantage[w_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    print("\nAdvantage:\n", advantage_table)

    significance = np.zeros((len(clfs), len(clfs)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    print("\nStatistical significance (alpha = 0.05):\n", significance_table)

    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate(
        (names_column, stat_better), axis=1), headers)
    print("Statistically significantly better:\n", stat_better_table)
