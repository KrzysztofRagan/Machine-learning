from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import RUSBoostClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.under_sampling import RandomUnderSampler
from NBBag import NBBaggingClassifier

from parameters import *

BASE_CLF = DecisionTreeClassifier(random_state=RANDOM_STATE)

# clfs = {
#     'ABOO': AdaBoostClassifier(base_estimator=BASE_CLF, n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE),
#     'BAG': BaggingClassifier(base_estimator=BASE_CLF, n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE),
#     'SMOTE': [('over', SMOTE()), ('model', BASE_CLF)]
# }


clfs = {
    # 'NBBAG_k7_fi0.5_euclidean': Pipeline(steps=[('model', NBBaggingClassifier(base_estimator=BASE_CLF,
    #                                                        n_estimators=N_ESTIMATORS,
    #                                                        k_neighbours=7,
    #                                                        fi=0.5,
    #                                                        random_state=RANDOM_STATE))]),
    # 'NBBAG_k7_fi1_euclidean': Pipeline(steps=[('model', NBBaggingClassifier(base_estimator=BASE_CLF,
    #                                                                         n_estimators=N_ESTIMATORS,
    #                                                                         k_neighbours=7,
    #                                                                         fi=1,
    #                                                                         dist_metric="euclidean",
    #                                                                         random_state=RANDOM_STATE))]),
    # 'NBBAG_k7_fi1_cosine': Pipeline(steps=[('model', NBBaggingClassifier(base_estimator=BASE_CLF,
    #                                                                      n_estimators=N_ESTIMATORS,
    #                                                                      k_neighbours=7,
    #                                                                      fi=1,
    #                                                                      dist_metric="cosine",
    #                                                                      random_state=RANDOM_STATE))]),
    # 'NBBAG_k7_fi1_chebyshev': Pipeline(steps=[('model', NBBaggingClassifier(base_estimator=BASE_CLF,
    #                                                                         n_estimators=N_ESTIMATORS,
    #                                                                         k_neighbours=7,
    #                                                                         fi=1,
    #                                                                         dist_metric="chebyshev",
    #                                                                         random_state=RANDOM_STATE))]),
    # 'NBBAG_k7_fi1_manhattan': Pipeline(steps=[('model', NBBaggingClassifier(base_estimator=BASE_CLF,
    #                                                                         n_estimators=N_ESTIMATORS,
    #                                                                         k_neighbours=7,
    #                                                                         fi=1,
    #                                                                         dist_metric="manhattan",
    #                                                                         random_state=RANDOM_STATE))]),
    # 'NBBAG_k7_fi2_euclidean': Pipeline(steps=[('model', NBBaggingClassifier(base_estimator=BASE_CLF,
    #                                                        n_estimators=N_ESTIMATORS,
    #                                                        k_neighbours=7,
    #                                                        fi=2,
    #                                                        random_state=RANDOM_STATE))]),
    # 'NBBAG_k7_fi1.5_euclidean': Pipeline(steps=[('model', NBBaggingClassifier(base_estimator=BASE_CLF,
    #                                                        n_estimators=N_ESTIMATORS,
    #                                                        k_neighbours=7,
    #                                                        fi=1.5,
    #                                                        random_state=RANDOM_STATE))]),
    # 'NBBAG_k3_fi1_euclidean': Pipeline(steps=[('model', NBBaggingClassifier(base_estimator=BASE_CLF,
    #                                                        n_estimators=N_ESTIMATORS,
    #                                                        k_neighbours=3,
    #                                                        fi=1,
    #                                                        random_state=RANDOM_STATE))]),
    # 'NBBAG_k5_fi1_euclidean': Pipeline(steps=[('model', NBBaggingClassifier(base_estimator=BASE_CLF,
    #                                                        n_estimators=N_ESTIMATORS,
    #                                                        k_neighbours=5,
    #                                                        fi=1,
    #                                                        random_state=RANDOM_STATE))]),
    # 'NBBAG_k7_fi1_euclidean': Pipeline(steps=[('model', NBBaggingClassifier(base_estimator=BASE_CLF,
    #                                                        n_estimators=N_ESTIMATORS,
    #                                                        k_neighbours=7,
    #                                                        fi=1,
    #                                                        random_state=RANDOM_STATE))]),
    # 'NBBAG_k9_fi1_euclidean': Pipeline(steps=[('model', NBBaggingClassifier(base_estimator=BASE_CLF,
    #                                                        n_estimators=N_ESTIMATORS,
    #                                                        k_neighbours=9,
    #                                                        fi=1,
    #                                                        random_state=RANDOM_STATE))]),
    # 'NBBAG_k11_fi1_euclidean': Pipeline(steps=[('model', NBBaggingClassifier(base_estimator=BASE_CLF,
    #                                                        n_estimators=N_ESTIMATORS,
    #                                                        k_neighbours=11,
    #                                                        fi=1,
    #                                                        random_state=RANDOM_STATE))]),
    # 'AdaBoost': Pipeline(steps=[('model', AdaBoostClassifier(base_estimator=BASE_CLF,
    #                                                     n_estimators=N_ESTIMATORS,
    #                                                     random_state=RANDOM_STATE))]),
    # 'Bagging': Pipeline(steps=[('model', BaggingClassifier(base_estimator=BASE_CLF,
    #                                                    n_estimators=N_ESTIMATORS,
    #                                                    random_state=RANDOM_STATE))]),
    # 'SMOTEAdaBoost': Pipeline(steps=[('over', SMOTE()), ('model', AdaBoostClassifier(base_estimator=BASE_CLF,
    #                                                                          n_estimators=N_ESTIMATORS,
    #                                                                          random_state=RANDOM_STATE))]),
    # 'RUSBoost': Pipeline(steps=[('model', RUSBoostClassifier(base_estimator=BASE_CLF,
    #                                                          n_estimators=N_ESTIMATORS,
    #                                                          random_state=RANDOM_STATE))]),
    # 'EBBAG': Pipeline(steps=[('model', BalancedBaggingClassifier(base_estimator=BASE_CLF,
    #                                                              n_estimators=N_ESTIMATORS,
    #                                                              random_state=RANDOM_STATE,
    #                                                              sampler=RandomUnderSampler()))])
    # 'BorderlineSMOTE': Pipeline(
    #         steps=[('over', BorderlineSMOTE()), ('model', AdaBoostClassifier(base_estimator=BASE_CLF,
    #                                                                          n_estimators=N_ESTIMATORS,
    #                                                                          random_state=RANDOM_STATE))]),
    #     'SMOTE+UNDER': Pipeline(steps=[('over', SMOTE()),
    #                                    ('under', RandomUnderSampler()),
    #                                    ('model', AdaBoostClassifier(base_estimator=BASE_CLF,
    #                                                                 n_estimators=N_ESTIMATORS,
    #                                                                 random_state=RANDOM_STATE))]),
}
