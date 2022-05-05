from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import RUSBoostClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.under_sampling import RandomUnderSampler




RANDOM_STATE = 1234
BASE_CLF = DecisionTreeClassifier(random_state=RANDOM_STATE)

N_ESTIMATORS = 50

# clfs = {
#     'ABOO': AdaBoostClassifier(base_estimator=BASE_CLF, n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE),
#     'BAG': BaggingClassifier(base_estimator=BASE_CLF, n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE),
#     'SMOTE': [('over', SMOTE()), ('model', BASE_CLF)]
# }


clfs = {
    'ABOO': Pipeline(steps=[('model', AdaBoostClassifier(base_estimator=BASE_CLF,
                                                         n_estimators=N_ESTIMATORS,
                                                         random_state=RANDOM_STATE))]),
    'BAG': Pipeline(steps=[('model', BaggingClassifier(base_estimator=BASE_CLF,
                                                       n_estimators=N_ESTIMATORS,
                                                       random_state=RANDOM_STATE))]),
    'SMOTE': Pipeline(steps=[('over', SMOTE()), ('model', AdaBoostClassifier(base_estimator=BASE_CLF,
                                                                             n_estimators=N_ESTIMATORS,
                                                                             random_state=RANDOM_STATE))]),
    'BorderlineSMOTE': Pipeline(
        steps=[('over', BorderlineSMOTE()), ('model', AdaBoostClassifier(base_estimator=BASE_CLF,
                                                                         n_estimators=N_ESTIMATORS,
                                                                         random_state=RANDOM_STATE))]),
    'SMOTE+UNDER': Pipeline(steps=[('over', SMOTE()),
                                   ('under', RandomUnderSampler()),
                                   ('model', AdaBoostClassifier(base_estimator=BASE_CLF,
                                                                n_estimators=N_ESTIMATORS,
                                                                random_state=RANDOM_STATE))]),
    'RUSBoost': Pipeline(steps=[('model', RUSBoostClassifier(base_estimator=BASE_CLF,
                                                             n_estimators=N_ESTIMATORS,
                                                             random_state=RANDOM_STATE))]),
    'EBBAG': Pipeline(steps=[('model', BalancedBaggingClassifier(base_estimator=BASE_CLF,
                                                             n_estimators=N_ESTIMATORS,
                                                             random_state=RANDOM_STATE,
                                                             sampler=RandomUnderSampler()))])


}

datasets = ['ecoli-0_vs_1', 'ecoli1', 'ecoli2', 'ecoli3', 'glass-0-1-2-3_vs_4-5-6', 'glass0', 'glass1', 'glass6',
            'haberman', 'iris0', 'new-thyroid1', 'newthyroid2', 'page-blocks0', 'pima', 'segment0', 'vehicle0',
            'vehicle1', 'vehicle2', 'vehicle3', 'wisconsin', 'yeast1', 'yeast3']

# datasets = ['ecoli-0_vs_1', 'ecoli1', 'ecoli2', 'ecoli3']

n_datasets = len(datasets)
n_splits = 5
n_repeats = 2

#  1 Przygotowanie danych- napisać funkcję, które będzie importować dane z plików do klasyfikatorów
#  2 Wybór i konfiguracja odpowiednich klasyfikatorów
#  3 Główna pętla ucząca
#  4 Zbieranie danych wynikowych uczenia + eksport do csv
#  5 Testy statystyczne
#  6 Wizualizacja
