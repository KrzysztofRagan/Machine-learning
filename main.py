import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from parameters import *

rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)

scores = np.zeros((len(clfs), n_datasets, n_splits * n_repeats))

# for data_id, dataset in enumerate(datasets):
#     dataset = np.genfromtxt("datasets/%s.dat" % dataset, delimiter=",", comments='@',
#                             converters={(-1): lambda s: 0.0 if (s.strip().decode('ascii')) == 'negative' else 1.0})
#     X = dataset[:, :-1]
#     y = dataset[:, -1].astype(int)
#
#     for fold_id, (train, test) in enumerate(rskf.split(X, y)):
#         for clf_id, clf_name in enumerate(clfs):
#             clf = clone(clfs[clf_name])
#             clf.fit(X[train], y[train])
#             y_pred = clf.predict(X[test])
#             scores[clf_id, data_id, fold_id] = accuracy_score(y[test], y_pred)
#
# np.save('results', scores)
# print(scores)

for data_id, dataset in enumerate(datasets):
    dataset = np.genfromtxt("datasets/%s.dat" % dataset, delimiter=",", comments='@',
                            converters={(-1): lambda s: 0.0 if (s.strip().decode('ascii')) == 'negative' else 1.0})
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    for clf_id, clf_name in enumerate(clfs):
        clf = clfs[clf_name]
        scores[clf_id, data_id] = cross_val_score(clf, X, y, scoring='accuracy', cv=rskf, n_jobs=-1)

np.save('results', scores)
print(scores)