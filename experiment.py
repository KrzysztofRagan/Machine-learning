import numpy as np
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
