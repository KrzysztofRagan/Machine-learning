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
def t_student(dataset_scores, clfs_names, alfa=0.05):
    """
    Oblicza tabele porównawczą klasyfikatorów dla podanej tablicy

    :param dataset_scores: columns: classifiers, rows: folds
    :type dataset_scores:
    :param clfs_names:
    :type clfs_names:
    :param alfa:
    :type alfa:
    :return:
    :rtype:
    """

    # dateset_scores = scores[:,i]
    n_clfs = len(clfs_names)
    t_statistic = np.zeros((n_clfs, n_clfs))
    p_value = np.zeros((n_clfs, n_clfs))

    for i in range(n_clfs):
        for j in range(n_clfs):
            t_statistic[i, j], p_value[i, j] = ttest_rel(dataset_scores[i], dataset_scores[j])
    # print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)

    headers = clfs_names
    names_column = np.expand_dims(np.array(clfs_names), axis=1)
    t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    # print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

    advantage = np.zeros((n_clfs, n_clfs))
    advantage[t_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    # print("\nAdvantage:\n", advantage_table)

    significance = np.zeros((n_clfs, n_clfs))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    # print("\nStatistical significance (alpha = 0.05):\n", significance_table)

    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate(
        (names_column, stat_better), axis=1), headers)

    stat_better_list = []
    for i in range(n_clfs):
        # print(f"{clfs_names[i]} stat better than:", end="")
        tmp = []
        for j in range(n_clfs):
            if stat_better[i][j] == 1:
                tmp.append(j+1)
                # print(f"{clfs_names[j]}", end="")
        stat_better_list.append(tmp)
        # print()

    # print("Statistically significantly better:\n", stat_better_table)

    return stat_better, stat_better_list


# ---------------- TEST WILCOXONA ------------------

def wilcoxon(ranks, clfs_names, alfa=0.05):
    '''
    Wilcoxon test

    :param ranks: matrix of ranks for each dataset; colums: classifiers, rows: datasets
    :type ranks:
    :return:
    :rtype:
    '''
    print("\n------------ TEST WILCOXONA ------------")
    alfa = .05
    n_clfs = len(clfs_names)
    w_statistic = np.zeros((n_clfs, n_clfs))
    p_value = np.zeros((n_clfs, n_clfs))

    for i in range(n_clfs):
        for j in range(n_clfs):
            w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])

    headers = clfs_names
    names_column = np.expand_dims(np.array(clfs_names), axis=1)
    w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
    w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

    advantage = np.zeros((n_clfs, n_clfs))
    advantage[w_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    print("\nAdvantage:\n", advantage_table)

    significance = np.zeros((n_clfs, n_clfs))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    print("\nStatistical significance (alpha = 0.05):\n", significance_table)

    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate(
        (names_column, stat_better), axis=1), headers)
    print("Statistically significantly better:\n", stat_better_table)
