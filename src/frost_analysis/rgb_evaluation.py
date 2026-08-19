"""Experiment-held-out evaluation for compact frost-image features."""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def leave_one_experiment_out_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Fit the locked RBF-SVM on all but one experiment at a time."""
    feature_columns = [column for column in frame if column.startswith("feature_")]
    predictions = []
    for experiment in sorted(frame["experiment_id"].unique()):
        test = frame.loc[frame["experiment_id"].eq(experiment)].copy()
        train = frame.loc[~frame["experiment_id"].eq(experiment)]
        if train["target"].nunique() < 2:
            continue
        model = make_pipeline(
            StandardScaler(),
            SVC(C=2.0, class_weight="balanced", random_state=0),
        )
        model.fit(train[feature_columns], train["target"])
        test["predicted_target"] = model.predict(test[feature_columns])
        test["decision_score"] = model.decision_function(test[feature_columns])
        test["held_out_experiment"] = experiment
        predictions.append(test)
    return pd.concat(predictions, ignore_index=True)
