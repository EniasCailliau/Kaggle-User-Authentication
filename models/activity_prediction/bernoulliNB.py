import inspect

from sklearn import svm
from sklearn.base import BaseEstimator

CREATED_BY = "Enias Cailliau"
MODEL_NAME = "Bernouilli Naive Bayes - activity"


# default_settings = {'warm_start': False, 'C': 10, 'n_jobs': 8, 'verbose': 0, 'intercept_scaling': 1,
#                     'fit_intercept': True, 'max_iter': 100, 'penalty': 'l2', 'multi_class': 'multinomial',
#                     'random_state': None, 'dual': False, 'tol': 0.0001, 'solver': 'newton-cg',
#                     'class_weight': None}



# TODO: Enias; investigate model parameters
class BernoulliNB(BaseEstimator):
    def __init__(self, alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None):
        input_args = locals()
        del input_args["self"]
        self.estimator = svm.SVC(**input_args)

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
