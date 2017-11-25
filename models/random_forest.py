import inspect

from sklearn import ensemble
from sklearn.base import BaseEstimator

CREATED_BY = "Jeroen Van Hautte"
MODEL_NAME = "Random Forest"


class RF(BaseEstimator):
    def __init__(self, n_estimators=256, n_jobs=-1, oob_score=True, verbose=1):
        input_args = locals()
        del input_args["self"]
        self.estimator = ensemble.RandomForestClassifier(**input_args)
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
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
