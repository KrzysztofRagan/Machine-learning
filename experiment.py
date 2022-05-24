import numpy as np
from sklearn import neighbors
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from parameters import *
from math import sqrt

rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)

scores = np.zeros((len(clfs), n_datasets, n_splits * n_repeats))


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


def nearest_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


def get_weight(X, y, k_neighbors=5, fi=1):
    weights = []
    n_major = np.sum(y)
    n_minor = np.sum(1 - y)
    for i in range(len(X)):
        if y[i] == 1:
            weights.append(0.5 * n_major / n_minor)
        else:
            test_row = X[i]
            train = np.delete(X.copy(), i, 0)
            neighbors = nearest_neighbors(train, test_row, k_neighbors)
            N_prim = 0
            for neighbor in neighbors:
                idx = 0
                for j in range(len(X)):
                    if np.array_equal(X[j], neighbor):
                        idx = j
                if y[idx] == 1:
                    N_prim += 1
            w = 0.5 * (N_prim ** fi / k_neighbors + 1)
            weights.append(w)
    return weights


# for data_id, dataset in enumerate(datasets):
#     dataset = np.genfromtxt("datasets/%s.dat" % dataset, delimiter=",", comments='@',
#                             converters={(-1): lambda s: 0.0 if (s.strip().decode('ascii')) == 'negative' else 1.0})
#     X = dataset[:, :-1]
#     y = dataset[:, -1].astype(int)

#     for clf_id, clf_name in enumerate(clfs):
#         clf = clfs[clf_name]
#         scores[clf_id, data_id] = cross_val_score(clf, X, y, scoring='accuracy', cv=rskf, n_jobs=-1)
#         print('Classifier {} finished.'.format(clf_name))

# np.save('results', scores)
# print(scores)


# Different version of trial of each classifier
for data_id, dataset in enumerate(datasets):
    dataset = np.genfromtxt("datasets/%s.dat" % dataset, delimiter=",", comments='@',
                            converters={(-1): lambda s: 0.0 if (s.strip().decode('ascii')) == 'negative' else 1.0})
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])
            weights = get_weight(X[train], y[train])
            print(weights)
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            scores[clf_id, data_id, fold_id] = accuracy_score(y[test], y_pred)

np.save('results', scores)
print(scores)
