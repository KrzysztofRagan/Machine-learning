import numpy as np
from parameters import *
from experiment import experiment
from analysis import *
from datasets import datasets
from sklearn.metrics import f1_score, accuracy_score
from imblearn.metrics import geometric_mean_score, specificity_score

# used metrics
metrics = (geometric_mean_score, specificity_score, f1_score, accuracy_score)

# experiment loop
for clf_name, clf in clfs.items():
    print(clf_name)
    scores = experiment(clf=clf,
                        datasets=datasets,
                        metrics=metrics,
                        n_repeats=n_repeats,
                        n_splits=n_splits,
                        random_state=RANDOM_STATE)
    np.save(clf_name, scores)  # save scores in clf_name.npy file


# print(scores)

###########################################################################
# ANALYSIS
###########################################################################

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
