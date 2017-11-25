import inspect

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import ensemble

CREATED_BY = "Enias Cailliau"
MODEL_NAME = "Random Forest - activity"


class RandomForest(BaseEstimator):
    def __init__(self, n_estimators=250, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                 min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=-1, random_state=None, verbose=0,
                 warm_start=False, class_weight=None):
        input_args = locals()
        del input_args["self"]
        self.estimator = ensemble.RandomForestClassifier(**input_args)
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
