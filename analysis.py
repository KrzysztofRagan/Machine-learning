import numpy as np


scores = np.load('results.npy')
print("\nScores:\n", scores.shape)

mean_scores = np.mean(scores, axis = 2).T
print("\nMean scores: \n", mean_scores)

