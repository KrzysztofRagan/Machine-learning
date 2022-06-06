import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.base import clone


def experiment(clf: dict, datasets: list, metrics: tuple, n_splits: int, n_repeats: int, random_state: int):
    '''

    :param clf:
    :type clf:
    :param datasets:
    :type datasets:
    :param metrics:
    :type metrics:
    :param n_splits:
    :type n_splits:
    :param n_repeats:
    :type n_repeats:
    :param random_state:
    :type random_state:
    :return: scores: matrix metrics x datasets x folds
    :rtype: list
    '''
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    n_datasets = len(datasets)
    scores = np.zeros((len(metrics), n_datasets, n_splits * n_repeats))
    # scores = np.zeros((len(clfs), n_datasets, n_splits * n_repeats))

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
    for data_id, dataset_name in enumerate(datasets):
        print(f'Dataset {dataset_name}')
        dataset = np.genfromtxt("datasets/%s.dat" % dataset_name, delimiter=",", comments='@',
                                converters={(-1): lambda s: 0.0 if (s.strip().decode('ascii')) == 'negative' else 1.0})
        X = dataset[:, :-1]
        y = dataset[:, -1].astype(int)

        for fold_id, (train, test) in enumerate(rskf.split(X, y)):
            for metric_id, metric in enumerate(metrics):
                clf = clone(clf)
                clf.fit(X[train], y[train])
                y_pred = clf.predict(X[test])
                scores[metric_id, data_id, fold_id] = metric(y[test], y_pred)
    return scores

    # np.save('results', scores)
    # print(scores)


def _nearest_neighbors(X, Y, k_neighbors, metric="euclidean"):
    distances = pairwise_distances(Y, [X], metric)
    l = list(zip(distances, Y))
    l.sort(key=lambda tup: tup[0])
    neighbors = list()
    for i in range(k_neighbors):
        neighbors.append(l[i][1])
    return neighbors


def NBBag_experiment(clf, datasets: list, metrics: tuple,
                     n_splits: int, n_repeats: int, random_state: int, k_neighbours, fi, dist_metric):
    '''

    '''
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    n_datasets = len(datasets)
    scores = np.zeros((len(metrics), n_datasets, n_splits * n_repeats))

    # Different version of trial of each classifier
    for data_id, dataset_name in enumerate(datasets):
        print(f'Dataset {dataset_name}')
        dataset = np.genfromtxt("datasets/%s.dat" % dataset_name, delimiter=",", comments='@',
                                converters={(-1): lambda s: 0.0 if (s.strip().decode('ascii')) == 'negative' else 1.0})
        X = dataset[:, :-1]
        y = dataset[:, -1].astype(int)

        W = []
        counts = np.bincount(y)
        majority_class = np.argmax(counts)
        n_major = np.max(counts)
        n_minor = np.min(counts)
        for i in range(len(X)):
            if y[i] == majority_class:
                W.append(0.5 * n_major / n_minor)
            else:
                sample = X[i]
                all_neighbours = np.delete(X.copy(), i, 0)
                neighbors = _nearest_neighbors(sample,
                                               all_neighbours,
                                               k_neighbours,
                                               metric=dist_metric)
                N_prim = 0
                for neighbor in neighbors:
                    idx = np.where(np.all(X == neighbor, axis=1))[0][0]
                    if y[idx] == majority_class:
                        N_prim += 1
                w = 0.5 * (N_prim ** fi / k_neighbours + 1)
                W.append(w)

        W = np.array(W)

        for fold_id, (train, test) in enumerate(rskf.split(X, y)):
            for metric_id, metric in enumerate(metrics):
                clf = clone(clf)
                clf.fit(X[train], y[train], sample_weight=W[train])
                y_pred = clf.predict(X[test])
                scores[metric_id, data_id, fold_id] = metric(y[test], y_pred)
    return scores

    # np.save('results', scores)
    # print(scores)
