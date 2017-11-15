import inspect

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


CREATED_BY = "Enias Cailliau"
MODEL_NAME = "Logistic Regression"

default_settings = {'warm_start': False, 'C': 10, 'n_jobs': 8, 'verbose': 0, 'intercept_scaling': 1,
                    'fit_intercept': True, 'max_iter': 100, 'penalty': 'l2', 'multi_class': 'multinomial',
                    'random_state': None, 'dual': False, 'tol': 0.0001, 'solver': 'newton-cg',
                    'class_weight': None}


class LogReg(BaseEstimator):
    def __init__(self, warm_start=False, C=10, n_jobs=8, verbose=0, intercept_scaling=1, fit_intercept=True,
                 max_iter=100, multi_class='multinomial', tol=0.0001, solver='newton-cg',
                 class_weight=None):
        input_args = locals()
        del input_args["self"]
        self.estimator = LogisticRegression(**input_args)
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

    def get_params(self, deep=True):
        return self.estimator.get_params(deep)

    def get_description(self):
        return "{} created by {}".format(MODEL_NAME, CREATED_BY)
