import numpy as np
import os
from sklearn.utils import check_X_y, safe_sqr
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from sklearn.base import clone
from sklearn.base import is_classifier
from sklearn.externals.joblib import Parallel, delayed
from sklearn.model_selection import check_cv
from sklearn.model_selection._validation import _safe_split, _score
from sklearn.metrics.scorer import check_scoring
from sklearn.feature_selection.base import SelectorMixin
from utils import handyman

def _rfe_single_fit(rfe, estimator, X, y, train, test, scorer):
    """
    Return the score for a fit across one fold.
    """
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)
    return rfe._fit(
        X_train, y_train, lambda estimator, features:
        _score(estimator, X_test[:, features], y_test, scorer)).scores_


class RFE(BaseEstimator, MetaEstimatorMixin, SelectorMixin):
    """Feature ranking with recursive feature elimination.
    Given an external estimator that assigns weights to features (e.g., the
    coefficients of a linear model), the goal of recursive feature elimination
    (RFE) is to select features by recursively considering smaller and smaller
    sets of features. First, the estimator is trained on the initial set of
    features and the importance of each feature is obtained either through a
    ``coef_`` attribute or through a ``feature_importances_`` attribute.
    Then, the least important features are pruned from current set of features.
    That procedure is recursively repeated on the pruned set until the desired
    number of features to select is eventually reached.
    Read more in the :ref:`User Guide <rfe>`.
    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a ``fit`` method that provides
        information about feature importance either through a ``coef_``
        attribute or through a ``feature_importances_`` attribute.
    n_features_to_select : int or None (default=None)
        The number of features to select. If `None`, half of the features
        are selected.
    step : int or float, optional (default=1)
        If greater than or equal to 1, then `step` corresponds to the (integer)
        number of features to remove at each iteration.
        If within (0.0, 1.0), then `step` corresponds to the percentage
        (rounded down) of features to remove at each iteration.
    verbose : int, default=0
        Controls verbosity of output.
    Attributes
    ----------
    n_features_ : int
        The number of selected features.
    support_ : array of shape [n_features]
        The mask of selected features.
    ranking_ : array of shape [n_features]
        The feature ranking, such that ``ranking_[i]`` corresponds to the
        ranking position of the i-th feature. Selected (i.e., estimated
        best) features are assigned rank 1.
    estimator_ : object
        The external estimator fit on the reduced dataset.
    Examples
    --------
    The following example shows how to retrieve the 5 right informative
    features in the Friedman #1 dataset.
    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.feature_selection import RFE
    >>> from sklearn.svm import SVR
    >>> X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    >>> estimator = SVR(kernel="linear")
    >>> selector = RFE(estimator, 5, step=1)
    >>> selector = selector.fit(X, y)
    >>> selector.support_ # doctest: +NORMALIZE_WHITESPACE
    array([ True,  True,  True,  True,  True,
            False, False, False, False, False], dtype=bool)
    >>> selector.ranking_
    array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])
    References
    ----------
    .. [1] Guyon, I., Weston, J., Barnhill, S., & Vapnik, V., "Gene selection
           for cancer classification using support vector machines",
           Mach. Learn., 46(1-3), 389--422, 2002.
    """
    def __init__(self, estimator, estimator_name, n_features_to_select=None, step=1,
                 verbose=0):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.verbose = verbose
        self.estimator_name = estimator_name

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    def fit(self, X, y):
        """Fit the RFE model and then the underlying estimator on the selected
           features.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.
        """
        return self._fit(X, y, self.estimator_name)

    def _fit(self, X, y, estimator_name, step_score=None):
        # Parameter step_score controls the calculation of self.scores_
        # step_score is not exposed to users
        # and is used when implementing RFECV
        # self.scores_ will not be calculated when calling _fit through fit

        X, y = check_X_y(X, y, "csc")
        # Initialization
        n_features = X.shape[1]
        if self.n_features_to_select is None:
            n_features_to_select = n_features // 2
        else:
            n_features_to_select = self.n_features_to_select

        if 0.0 < self.step < 1.0:
            step = int(max(1, self.step * n_features))
        else:
            step = int(self.step)
        if step <= 0:
            raise ValueError("Step must be >0")

        support_ = np.ones(n_features, dtype=np.bool)
        ranking_ = np.ones(n_features, dtype=np.int)

        if step_score:
            self.scores_ = []

        # Elimination
        while np.sum(support_) > n_features_to_select:
            
            iteration = np.sum(support_)
            # Remaining features
            features = np.arange(n_features)[support_]

            # Rank the remaining features
            estimator = clone(self.estimator)
            if self.verbose > 0:
                print("Fitting estimator with %d features." % np.sum(support_))

            estimator.fit(X[:, features], y)
            # Get coefs
            if hasattr(estimator, 'coef_'):
                coefs = estimator.coef_
            else:
                coefs = getattr(estimator, 'feature_importances_', None)
            if coefs is None:
                raise RuntimeError('The classifier does not expose '
                                   '"coef_" or "feature_importances_" '
                                   'attributes')

            # Get ranks
            if coefs.ndim > 1:
                ranks = np.argsort(safe_sqr(coefs).sum(axis=0))
            else:
                ranks = np.argsort(safe_sqr(coefs))

            # for sparse case ranks is matrix
            ranks = np.ravel(ranks)

            # Eliminate the worse features
            threshold = min(step, np.sum(support_) - n_features_to_select)

            # Compute step score on the previous selection iteration
            # because 'estimator' must use features
            # that have not been eliminated yet
            if step_score:
                self.scores_.append(step_score(estimator, features))
            support_[features[ranks][:threshold]] = False
            ranking_[np.logical_not(support_)] += 1

            if(iteration % 100 == 0):
                print("dumping RFE intermediate with {} features".format(iteration))
                handyman.dump_pickle(
                    dict(train_features=X[:, features], train_subject_labels=y, ranking=ranking_),
                    os.path.join("feature_extraction", "_data_sets", "rfe_" + self.estimator_name + "_" + str(iteration) + ".pkl")
                )

        # Set final attributes
        features = np.arange(n_features)[support_]
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X[:, features], y)

        # Compute step score when only n_features_to_select features left
        if step_score:
            self.scores_.append(step_score(self.estimator_, features))
        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_

        return self

    @if_delegate_has_method(delegate='estimator')
    def predict(self, X):
        """Reduce X to the selected features and then predict using the
           underlying estimator.
        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.
        Returns
        -------
        y : array of shape [n_samples]
            The predicted target values.
        """
        check_is_fitted(self, 'estimator_')
        return self.estimator_.predict(self.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def score(self, X, y):
        """Reduce X to the selected features and then return the score of the
           underlying estimator.
        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.
        y : array of shape [n_samples]
            The target values.
        """
        check_is_fitted(self, 'estimator_')
        return self.estimator_.score(self.transform(X), y)

    def _get_support_mask(self):
        check_is_fitted(self, 'support_')
        return self.support_

    @if_delegate_has_method(delegate='estimator')
    def decision_function(self, X):
        check_is_fitted(self, 'estimator_')
        return self.estimator_.decision_function(self.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def predict_proba(self, X):
        check_is_fitted(self, 'estimator_')
        return self.estimator_.predict_proba(self.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def predict_log_proba(self, X):
        check_is_fitted(self, 'estimator_')
        return self.estimator_.predict_log_proba(self.transform(X))
