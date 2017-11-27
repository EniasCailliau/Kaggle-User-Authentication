import inspect

import xgboost
from sklearn.base import BaseEstimator

CREATED_BY = "Enias Cailliau"
MODEL_NAME = "Gradient Boosted Trees"


class xgb(BaseEstimator):
    def __init__(self, max_depth=5, learning_rate=0.1,
                 n_estimators=100, silent=True,
                 objective="binary:logistic",
                 nthread=-1, gamma=0, min_child_weight=1,
                 max_delta_step=0, subsample=0.8, colsample_bytree=0.8, colsample_bylevel=1,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, seed=0, missing=None):
        input_args = locals()
        del input_args["self"]

        self.estimator = xgboost.XGBClassifier(**input_args)
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
