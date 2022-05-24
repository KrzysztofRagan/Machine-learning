import numpy as np
from scipy.stats import rankdata
from scipy.stats import ranksums
from classifiers import clfs
from tabulate import tabulate
from scipy.stats import ttest_rel

# loading classifiers results
scores = np.load('results.npy')
print("\nScores:\n", scores.shape)
# print(scores)

# mean of 3rd dim (folds)
mean_scores = np.mean(scores, axis=2).T
print("\nMean scores: \n", mean_scores)

# ---------------- TEST FRIEDMANN ------------------
print("\n------------ TEST FRIEDMANNA ------------")
ranks = []
for ms in mean_scores:
    ranks.append(rankdata(ms).tolist())
ranks = np.array(ranks)
print("\nRanks:\n", ranks)

mean_ranks = np.mean(ranks, axis=0)
print("\nMean ranks:\n", mean_ranks)

################### TESTY PAROWE ##########################

# ---------------- TEST WILCOXONA ------------------
print("\n------------ TEST WICOXONA ------------")
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

# ---------------- TEST T-STUDENTA ------------------
print("\n------------ TEST T-STUDENTA ------------")
mean_scores = np.mean(scores, axis=2)
print("\nMean scores: \n", mean_scores)

alfa = .05
t_statistic = np.zeros((len(clfs), len(clfs)))
p_value = np.zeros((len(clfs), len(clfs)))

for i in range(len(clfs)):
    for j in range(len(clfs)):
        t_statistic[i, j], p_value[i, j] = ttest_rel(mean_scores[i], mean_scores[j])
print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)

t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

advantage = np.zeros((len(clfs), len(clfs)))
advantage[t_statistic > 0] = 1
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
