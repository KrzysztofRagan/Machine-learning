from operator import index
import numpy as np
from parameters import *
from experiment import experiment
from analysis import *
from datasets import datasets
from sklearn.metrics import f1_score, accuracy_score
from imblearn.metrics import geometric_mean_score, specificity_score
from classifiers import clfs
import pandas as pd

# used metrics
metrics = (geometric_mean_score, specificity_score, f1_score, accuracy_score)
metrics_names = ('geometric_mean_score', 'specificity_score', 'f1_score', 'accuracy_score')

#experiment loop
# for clf_name, clf in clfs.items():
#     print(clf_name)
#     scores = experiment(clf=clf,
#                         datasets=datasets,
#                         metrics=metrics,
#                         n_repeats=n_repeats,
#                         n_splits=n_splits,
#                         random_state=RANDOM_STATE)
#     np.save(f'./results/{clf_name}', scores)  # save scores in clf_name.npy file



###########################################################################
# ANALYSIS
###########################################################################


clfs_names = ['NBBAG_k7_fi0.5_eucl', 'NBBAG_k7_fi1_eucl','NBBAG_k7_fi1.5_eucl', 'NBBAG_k7_fi2_eucl']
clfs_scores = {clf_name: np.load(f'./results/{clf_name}.npy') for clf_name in clfs_names}

clfs_mean_scores = {clf_name: np.mean(clfs_scores[clf_name], axis = 2) for clf_name in clfs_scores.keys()}

#saving mean_scores to files (1 classfier = 1 file)
for clf_name, mean_score in clfs_mean_scores.items():
    mean_score = pd.DataFrame(np.round(mean_score.T, 2), columns= metrics_names, index= datasets)
    mean_score.to_csv(f'./results/clfs_mean_scores/{clf_name}.csv', sep= '\t')

#saving metrics to files (1 metric = 1 file)
met_mean_scores = {}
for i, metric_name in enumerate(metrics_names):
    mean_score = np.array([ms[i] for ms in clfs_mean_scores.values()])
    met_mean_scores[metric_name] = mean_score
    mean_score = pd.DataFrame(np.round(mean_score.T, 2), columns= clfs_names, index= datasets)
    mean_score.to_csv(f'./results/metric_mean_scores/{metric_name}.csv', sep= '\t')   


#testing different parameters by t-student 
g_mean_score = np.array([s[0] for s in clfs_scores.values()])
g_mean_scores = np.swapaxes(g_mean_score, 0, 1) #datasets x metrics x folds
print(g_mean_scores.shape)
for i, g_mean in enumerate(g_mean_scores):
    print('\n',datasets[i])
    t_student(g_mean)



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
