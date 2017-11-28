# import inspect
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.base import BaseEstimator
#
# CREATED_BY = "Joppe Massant"
# MODEL_NAME = "Decision trees"
#
# default_settings = {'criterion': 'gini', 'splitter': 'best', 'max_depth': None, 'min_samples_split': 2,
#                     'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.0, 'max_features': None, 'random_state': None,
#                     'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None,
#                     'class_weight': None, 'presort': False}
#
#
# class DecTree(BaseEstimator):
#     def __init__(self, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1,
#                  min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
#                  min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False):
#         input_args = locals()
#         del input_args["self"]
#         self.estimator = DecisionTreeClassifier(**input_args)
#         args, _, _, values = inspect.getargvalues(inspect.currentframe())
#         values.pop("self")
#
#         for arg, val, in values.items():
#             setattr(self, arg, val)
#             # print("{} = {}".format(arg,val)
#
#     def fit(self, X, y):
#         return self.estimator.fit(X, y)
#
#     def predict_proba(self, X):
#          return self.estimator.predict_proba(X)
#
#     def predict(self, X):
#          return self.estimator.predict(X)
#
#     def score(self, X, y=None):
#         return (sum(self.predict(X)))
#
#     def get_params(self, deep=True):
#         return self.estimator.get_params(deep)
#
#     def get_description(self):
#         return "{} created by {}".format(MODEL_NAME, CREATED_BY)
