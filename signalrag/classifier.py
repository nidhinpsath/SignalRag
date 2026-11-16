"""
classifier.py — SignalRAGClassifier class.

Wraps a fitted sklearn estimator together with its StandardScaler so that
the scaler parameters (mean, std) learned on the training split are always
applied consistently to any new data, preventing the common bug of refitting
the scaler on the test split.

Usage
-----
clf = SignalRAGClassifier(RandomForestClassifier(...))
clf.fit(X_train, y_train, feat_cols, pos, neg)
metrics = clf.evaluate(X_test, y_test)
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from .config import SEED, FEATURE_COLS, LABEL_COL


class SignalRAGClassifier:
    """
    Supervised hallucination classifier trained on the 17 SignalRAG features.

    Parameters
    ----------
    estimator : sklearn-compatible estimator
        An unfitted classifier (e.g. RandomForestClassifier()).
    """

    def __init__(self, estimator):
        self.estimator = estimator
        self.scaler    = StandardScaler()
        self._fitted   = False
        self.feat_cols = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feat_cols: list[str] | None = None,
        pos: int | None = None,
        neg: int | None = None,
    ) -> "SignalRAGClassifier":
        """
        Fit the scaler on X_train, then fit the estimator.

        Parameters
        ----------
        X_train : (n_train, n_features) array
        y_train : (n_train,) array of 0/1 labels
        feat_cols : list of feature names, stored for reporting
        pos, neg : number of positive/negative training examples (unused
                   here but available for subclasses that need them)
        """
        self.feat_cols = feat_cols or FEATURE_COLS
        X_scaled = self.scaler.fit_transform(X_train)
        self.estimator.fit(X_scaled, y_train)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted probabilities for X (after applying the fitted scaler)."""
        self._check_fitted()
        return self.estimator.predict_proba(self.scaler.transform(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels for X."""
        self._check_fitted()
        return self.estimator.predict(self.scaler.transform(X))

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate on the held-out test split.

        Returns
        -------
        dict with keys AUROC, F1, Accuracy.
        """
        self._check_fitted()
        probs = self.predict_proba(X_test)[:, 1]
        preds = self.predict(X_test)
        return {
            "AUROC":    roc_auc_score(y_test, probs),
            "F1":       f1_score(y_test, preds, zero_division=0),
            "Accuracy": accuracy_score(y_test, preds),
        }

    def feature_importances(self) -> list[tuple[str, float]]:
        """
        Return (feature_name, |coefficient|) sorted descending.
        Works for LogisticRegression (coef_) and tree ensembles
        (feature_importances_).
        """
        self._check_fitted()
        if hasattr(self.estimator, "coef_"):
            imps = np.abs(self.estimator.coef_[0])
        elif hasattr(self.estimator, "feature_importances_"):
            imps = self.estimator.feature_importances_
        else:
            raise AttributeError(
                f"{type(self.estimator).__name__} has no coef_ or feature_importances_."
            )
        return sorted(zip(self.feat_cols, imps), key=lambda x: -x[1])

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit() before predict / evaluate.")


# ---------------------------------------------------------------------------
# Factory helpers — returns a grid-searched SignalRAGClassifier
#
# Each factory wraps the estimator in an sklearn Pipeline(scaler, estimator)
# so that StandardScaler is re-fitted independently on each CV fold,
# preventing scale statistics from leaking across folds. After grid search,
# the best scaler (from the winning fold's pipeline) and estimator are
# unpacked and stored in a SignalRAGClassifier for consistent inference.
# ---------------------------------------------------------------------------

def _unpack_pipeline(pipeline_clf: Pipeline) -> tuple:
    """Extract the fitted scaler and estimator from a Pipeline."""
    return pipeline_clf.named_steps["scaler"], pipeline_clf.named_steps["clf"]


def build_logistic_regression(X_tr: np.ndarray, y_tr: np.ndarray, feat_cols: list[str]) -> SignalRAGClassifier:
    """GridSearch LR inside a Pipeline and return a fitted SignalRAGClassifier."""
    pipe   = Pipeline([("scaler", StandardScaler()),
                       ("clf",    LogisticRegression(max_iter=3000, class_weight="balanced"))])
    params = {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    grid   = GridSearchCV(pipe, params, cv=cv, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_tr, y_tr)
    print(f"  LR  best params: {grid.best_params_}  CV AUROC: {grid.best_score_:.4f}")

    scaler, estimator = _unpack_pipeline(grid.best_estimator_)
    clf = SignalRAGClassifier(estimator)
    clf.scaler    = scaler
    clf._fitted   = True
    clf.feat_cols = feat_cols
    return clf


def build_mlp(X_tr: np.ndarray, y_tr: np.ndarray, feat_cols: list[str]) -> SignalRAGClassifier:
    """GridSearch MLP inside a Pipeline and return a fitted SignalRAGClassifier."""
    pipe   = Pipeline([("scaler", StandardScaler()),
                       ("clf",    MLPClassifier(max_iter=5000, random_state=SEED))])
    params = {
        "clf__hidden_layer_sizes": [(32, 16), (64, 32), (128, 64), (256, 128)],
        "clf__alpha":              [1e-5, 1e-4, 1e-3, 1e-2],
    }
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    grid = GridSearchCV(pipe, params, cv=cv, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_tr, y_tr)
    print(f"  MLP best params: {grid.best_params_}  CV AUROC: {grid.best_score_:.4f}")

    scaler, estimator = _unpack_pipeline(grid.best_estimator_)
    clf = SignalRAGClassifier(estimator)
    clf.scaler    = scaler
    clf._fitted   = True
    clf.feat_cols = feat_cols
    return clf


def build_random_forest(X_tr: np.ndarray, y_tr: np.ndarray, feat_cols: list[str]) -> SignalRAGClassifier:
    """GridSearch Random Forest inside a Pipeline and return a fitted SignalRAGClassifier."""
    pipe   = Pipeline([("scaler", StandardScaler()),
                       ("clf",    RandomForestClassifier(random_state=SEED, class_weight="balanced"))])
    params = {
        "clf__n_estimators":     [100, 200, 300],
        "clf__max_depth":        [3, 4, 5, None],
        "clf__min_samples_leaf": [1, 2, 4],
    }
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    grid = GridSearchCV(pipe, params, cv=cv, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_tr, y_tr)
    print(f"  RF  best params: {grid.best_params_}  CV AUROC: {grid.best_score_:.4f}")

    scaler, estimator = _unpack_pipeline(grid.best_estimator_)
    clf = SignalRAGClassifier(estimator)
    clf.scaler    = scaler
    clf._fitted   = True
    clf.feat_cols = feat_cols
    return clf


def build_xgboost(
    X_tr: np.ndarray, y_tr: np.ndarray, feat_cols: list[str],
    pos: int, neg: int,
) -> SignalRAGClassifier:
    """GridSearch XGBoost inside a Pipeline and return a fitted SignalRAGClassifier."""
    if not HAS_XGB:
        raise ImportError("xgboost is not installed. Run: pip install xgboost")

    pipe   = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    XGBClassifier(
            eval_metric="logloss", random_state=SEED,
            verbosity=0, scale_pos_weight=neg / max(pos, 1),
        )),
    ])
    params = {
        "clf__n_estimators":     [200, 300, 500],
        "clf__max_depth":        [3, 4, 5, 6],
        "clf__learning_rate":    [0.01, 0.05, 0.1],
        "clf__subsample":        [0.8, 1.0],
        "clf__colsample_bytree": [0.8, 1.0],
    }
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    grid = GridSearchCV(pipe, params, cv=cv, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_tr, y_tr)
    print(f"  XGB best params: {grid.best_params_}  CV AUROC: {grid.best_score_:.4f}")

    scaler, estimator = _unpack_pipeline(grid.best_estimator_)
    clf = SignalRAGClassifier(estimator)
    clf.scaler    = scaler
    clf._fitted   = True
    clf.feat_cols = feat_cols
    return clf
