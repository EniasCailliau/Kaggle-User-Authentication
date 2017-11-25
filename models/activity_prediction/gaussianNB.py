import inspect

from sklearn.naive_bayes import GaussianNB
from sklearn.base import BaseEstimator

CREATED_BY = "Enias Cailliau"
MODEL_NAME = "Gaussian Naive Bayes - activity"


# TODO: Enias investigate model parameters
class Gaussian(BaseEstimator):
    def __init__(self, priors=None):
        input_args = locals()
        del input_args["self"]
        self.estimator = GaussianNB(**input_args)

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

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
