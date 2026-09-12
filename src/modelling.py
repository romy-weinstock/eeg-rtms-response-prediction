##Modelling utilities
## Author: Romy Weinstock

# Required imports
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

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
    fit on train only), then fold-scoped feature scaling (StandardScaler,
    fit on train only -- added after unscaled multi-feature arms caused
    saga convergence failures when features had very different natural
    scales, e.g. FAA vs. bounded [0,1] Kuramoto metrics), then fit/tune
    each classifier in classifier_specs and score on the held-out test fold.

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

        scaler = StandardScaler()
        X_train_clean = scaler.fit_transform(X_train_clean)
        X_test_clean = scaler.transform(X_test_clean)

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


## Paired permutation test: does an extended feature set beat a baseline?
def paired_comparison_test(X_baseline, X_extended, y, age, classifier_specs,
                            n_outer_splits, n_inner_splits, random_state, n_permutations):
    """
    Tests whether X_extended's mean balanced accuracy exceeds X_baseline's
    by more than chance would produce, per classifier -- e.g. "does adding
    Kuramoto to FAA improve on FAA alone?" (Arm 2 vs. Arm 1).

    This is NOT the same as comparing each feature set's own permutation
    test p-value: two independently-significant results can differ from
    each other by an amount well within noise. The correct test is paired,
    on the DIFFERENCE: the same StratifiedKFold splits (same random_state)
    are used for both feature sets, so fold membership is identical --
    same subjects, same folds, only the feature set differs. For each of
    n_permutations, the SAME shuffled label set is used to score both
    feature sets, and the difference is recorded; this keeps the
    comparison paired throughout, not just at the observed-data step.

    Parameters:
    X_baseline (array-like, shape (n_samples, n_features_a)): e.g. FAA alone.
    X_extended (array-like, shape (n_samples, n_features_b)): e.g. FAA + Kuramoto.
    y (array-like, shape (n_samples,)): Responder/non-responder labels.
    age (array-like, shape (n_samples,)): Subject age, same order as X, y.
    classifier_specs (dict): name -> (estimator, param_grid or None).
    n_outer_splits, n_inner_splits (int): StratifiedKFold fold counts.
    random_state (int): shared seed -- must match whatever seeded the
    original observed-data runs for both feature sets, so fold membership
    is identical.
    n_permutations (int): number of label shuffles for the null distribution.

    Returns:
    pd.DataFrame, one row per classifier: classifier, observed_diff,
    null_mean, p_value. p_value is the fraction of the null difference
    distribution at or above the observed difference (+1/+1 correction).
    """
    baseline_observed_df = run_nested_cv(X_baseline, y, age, classifier_specs,
                                          n_outer_splits, n_inner_splits, random_state)
    extended_observed_df = run_nested_cv(X_extended, y, age, classifier_specs,
                                          n_outer_splits, n_inner_splits, random_state)

    baseline_observed = baseline_observed_df.groupby('classifier')['balanced_accuracy'].mean().to_dict()
    extended_observed = extended_observed_df.groupby('classifier')['balanced_accuracy'].mean().to_dict()

    observed_diff = {name: extended_observed[name] - baseline_observed[name] for name in classifier_specs}

    rng = np.random.RandomState(random_state)
    null_diffs = {name: [] for name in classifier_specs}

    for _ in range(n_permutations):
        y_shuffled = rng.permutation(y)

        perm_extended = run_nested_cv(X_extended, y_shuffled, age, classifier_specs,
                                       n_outer_splits, n_inner_splits, random_state)
        perm_extended_scores = perm_extended.groupby('classifier')['balanced_accuracy'].mean().to_dict()

        perm_baseline = run_nested_cv(X_baseline, y_shuffled, age, classifier_specs,
                                       n_outer_splits, n_inner_splits, random_state)
        perm_baseline_scores = perm_baseline.groupby('classifier')['balanced_accuracy'].mean().to_dict()

        for name in classifier_specs:
            null_diffs[name].append(perm_extended_scores[name] - perm_baseline_scores[name])

    results = []
    for name in classifier_specs:
        null_arr = np.array(null_diffs[name])
        p_value = (np.sum(null_arr >= observed_diff[name]) + 1) / (n_permutations + 1)
        results.append({
            'classifier': name,
            'observed_diff': observed_diff[name],
            'null_mean': null_arr.mean(),
            'p_value': p_value,
        })

    return pd.DataFrame(results)