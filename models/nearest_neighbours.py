import inspect
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator

CREATED_BY = "Joppe Massant"
MODEL_NAME = "K Nearest Neighbours"

default_settings = {'n_neighbors': 5, 'weights': 'uniform', 'algorithm': 'auto', 'leaf_size': 30, 'p': 2,
                    'metric': 'minkowski', 'metric_params': None, 'n_jobs': 1}


class KNN(BaseEstimator):
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski',
                 metric_params=None, n_jobs=1):
        input_args = locals()
        del input_args["self"]
        self.estimator = KNeighborsClassifier(**input_args)
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val, in values.items():
            setattr(self, arg, val)
            # print("{} = {}".format(arg,val)

    def fit(self, X, y):
         return self.estimator.fit(X, y)

    def predict_proba(self, X):
         return self.estimator.predict_proba(X)

    def predict(self, X):
         return self.estimator.predict(X)

    def score(self, X, y=None):
        return (sum(self.predict(X)))

    def get_params(self, deep=True):
        return self.estimator.get_params(deep)

    def get_description(self):
        return "{} created by {}".format(MODEL_NAME, CREATED_BY)

