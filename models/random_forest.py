import inspect

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import ensemble

class RF(BaseEstimator):
    def __init__(self, n_estimators=256, n_jobs=-1, oob_score=True):
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

    # This method is used for gridsearch !! needs revision
    def score(self, X, y=None):
        # counts number of values bigger than mean
        return (sum(self.predict(X)))