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
    'NBBAG': Pipeline(steps=[('model', NBBaggingClassifier(base_estimator=BASE_CLF,
                                                           n_estimators=N_ESTIMATORS,
                                                           k_neighbours=5,
                                                           fi=1,
                                                           random_state=RANDOM_STATE))]),
    'ADA': Pipeline(steps=[('model', AdaBoostClassifier(base_estimator=BASE_CLF,
                                                         n_estimators=N_ESTIMATORS,
                                                         random_state=RANDOM_STATE))]),
    'BAG': Pipeline(steps=[('model', BaggingClassifier(base_estimator=BASE_CLF,
                                                       n_estimators=N_ESTIMATORS,
                                                       random_state=RANDOM_STATE))]),
    'SMOTE': Pipeline(steps=[('over', SMOTE()), ('model', AdaBoostClassifier(base_estimator=BASE_CLF,
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

# 'BorderlineSMOTE': Pipeline(
#         steps=[('over', BorderlineSMOTE()), ('model', AdaBoostClassifier(base_estimator=BASE_CLF,
#                                                                          n_estimators=N_ESTIMATORS,
#                                                                          random_state=RANDOM_STATE))]),
#     'SMOTE+UNDER': Pipeline(steps=[('over', SMOTE()),
#                                    ('under', RandomUnderSampler()),
#                                    ('model', AdaBoostClassifier(base_estimator=BASE_CLF,
#                                                                 n_estimators=N_ESTIMATORS,
#                                                                 random_state=RANDOM_STATE))]),
