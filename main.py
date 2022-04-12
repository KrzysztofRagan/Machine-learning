import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

RANDOM_STATE = 1234
BASE_CLF = DecisionTreeClassifier(random_state = RANDOM_STATE)

N_ESTIMATORS = 50


#  1 Przygotowanie danych- napisać funkcję, które będzie importować dane z plików do klasyfikatorów
#  2 Wybór i konfiguracja odpowiednich klasyfikatorów
#  3 Główna pętla ucząca
#  4 Zbieranie danych wynikowych uczenia + eksport do csv
#  5 Testy statystyczne
#  6 Wizualizacja

clfs = {
    'ABOO': AdaBoostClassifier(base_estimator = BASE_CLF, n_estimators = N_ESTIMATORS, random_state = RANDOM_STATE),
    'BAG': BaggingClassifier(base_estimator = BASE_CLF, n_estimators = N_ESTIMATORS, random_state = RANDOM_STATE),
    'GNB': GaussianNB(),
}


datasets = ['ecoli-0_vs_1','ecoli1', 'ecoli2', 'ecoli3', 'glass-0-1-2-3_vs_4-5-6', 'glass0', 'glass1', 'glass6', 'haberman', 'iris0', 'new-thyroid1', 'newthyroid2', 'page-blocks0', 'pima', 'segment0', 'vehicle0', 'vehicle1', 'vehicle2', 'vehicle3', 'wisconsin', 'yeast1', 'yeast3']


n_datasets = len(datasets)
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state= RANDOM_STATE)

scores = np.zeros((len(clfs), n_datasets, n_splits * n_repeats))


for data_id, dataset in enumerate(datasets):
    dataset = np.genfromtxt("datasets/%s.dat" % (dataset), delimiter=",", comments = '@', converters = {(-1): lambda s: 0.0 if s.decode('ascii') == 'negative' else 1.0} )
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)
    print(X , y)

    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            scores[clf_id, data_id, fold_id] = accuracy_score(y[test], y_pred)

np.save('results', scores)




