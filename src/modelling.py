##Modelling utilities
## Author: Romy Weinstock

# Required imports
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, balanced_accuracy_score


## Fold-scoped age deconfounding transformer
class AgeDeconfounder(BaseEstimator, TransformerMixin):
    """
    Fold-scoped linear deconfounding for a continuous confound (age).

    Regresses each feature on age (OLS), fit on whichever data fit() is
    given. transform() applies the STORED coefficients from fit() -- it
    never refits -- so residuals computed on new data reflect only what
    was learned from the data fit() was originally called on. This is
    out-of-sample deconfounding (Chyzhyk et al., 2022); used to keep age
    from leaking between training and test folds during cross-validation.
    """

    def fit(self, X, age):
        """
        Learn the feature ~ age OLS regression, one per feature column.

        Parameters:
        X (array-like, shape (n_samples, n_features)): EEG feature values.
        age (array-like, shape (n_samples,)): Subject age, same order as X.

        Returns:
        self: Fitted transformer, with slopes_ and intercepts_ stored.
        """
        X = np.asarray(X)
        age = np.asarray(age).ravel()

        age_centered = age - age.mean()
        denom = np.sum(age_centered ** 2)

        self.slopes_ = (age_centered @ (X - X.mean(axis=0))) / denom
        self.intercepts_ = X.mean(axis=0) - self.slopes_ * age.mean()
        return self

    def transform(self, X, age):
        """
        Remove the age-predicted component from X, using stored coefficients.

        Parameters:
        X (array-like, shape (n_samples, n_features)): EEG feature values.
        age (array-like, shape (n_samples,)): Subject age, same order as X.

        Returns:
        X_clean (np.ndarray, shape (n_samples, n_features)): Residuals --
        X with the age-predicted component (from fit()) subtracted out.
        """
        X = np.asarray(X)
        age = np.asarray(age).ravel()

        predicted = self.intercepts_ + np.outer(age, self.slopes_)
        return X - predicted


## Nested cross-validation harness
def run_nested_cv(X, y, age, classifier_specs, n_outer_splits, n_inner_splits, random_state):
    """
    One full nested-CV pass: fold-scoped age deconfounding (AgeDeconfounder,
    fit on train only) inside each outer fold, then fit/tune each classifier
    in classifier_specs and score on the held-out test fold.

    Parameters:
    X (array-like, shape (n_samples, n_features)): EEG feature values.
    y (array-like, shape (n_samples,)): Responder/non-responder labels.
    age (array-like, shape (n_samples,)): Subject age, same order as X, y.
    classifier_specs (dict): name -> (estimator, param_grid or None). None
    means the estimator is fit directly, no inner-loop tuning.
    n_outer_splits, n_inner_splits (int): StratifiedKFold fold counts.
    random_state (int): shared seed for outer/inner splits and any
    stochastic estimator (e.g. saga).

    Returns:
    pd.DataFrame, one row per (classifier, outer fold): classifier, fold,
    balanced_accuracy, accuracy, auc, sensitivity, specificity, ppv.
    """
    outer_cv = StratifiedKFold(n_splits=n_outer_splits, shuffle=True, random_state=random_state)
    results = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        age_train, age_test = age[train_idx], age[test_idx]

        deconf = AgeDeconfounder()
        deconf.fit(X_train, age_train)
        X_train_clean = deconf.transform(X_train, age_train)
        X_test_clean = deconf.transform(X_test, age_test)

        for clf_name, (estimator, param_grid) in classifier_specs.items():
            if param_grid is not None:
                inner_cv = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=random_state)
                search = GridSearchCV(estimator, param_grid, cv=inner_cv, scoring='balanced_accuracy')
                search.fit(X_train_clean, y_train)
                fitted_model = search.best_estimator_
            else:
                fitted_model = estimator.fit(X_train_clean, y_train)

            y_pred = fitted_model.predict(X_test_clean)
            y_proba = fitted_model.predict_proba(X_test_clean)[:, 1]

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan

            results.append({
                'classifier': clf_name,
                'fold': fold_idx,
                'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                'accuracy': accuracy_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_proba),
                'sensitivity': sensitivity,
                'specificity': specificity,
                'ppv': ppv,
            })

    return pd.DataFrame(results)