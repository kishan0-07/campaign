import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler



NUMERIC = ["age", "previous"]
CATEGORICAL = [
    "job", "marital", "education", "housing", "loan", "poutcome"
]
FEATURES = NUMERIC + CATEGORICAL


def prepare_inputs(table, mode="labeled"):

    if mode not in {"labeled", "predict"}:
        raise ValueError("mode must be 'labeled' or 'predict'")

    if not table.columns.is_unique:
        raise ValueError("Duplicate column names are not allowed")

    if {"y", "record_id"} & set(FEATURES):
        raise ValueError("y and record_id cannot be model features")

    if mode == "predict" and "y" in table.columns:
        raise ValueError("Prediction input must not contain y")

    required = ["record_id"] + FEATURES
    if mode == "labeled":
        required = required + ["y"]

    missing = sorted(set(required) - set(table.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    ids = pd.to_numeric(table["record_id"], errors="raise").copy()

    if ids.isna().any() or not pd.api.types.is_integer_dtype(ids.dtype):
        raise ValueError("record_id must contain nonmissing integers")

    if not ids.is_unique:
        raise ValueError("record_id values must be unique")
    X = table.loc[:, FEATURES].copy()
    for column in NUMERIC:
        values = X[column].astype('string').str.strip().replace('', pd.NA)
        X[column] = pd.to_numeric(values, errors='raise').astype('float64')
        observed = X[column].dropna()
        if not np.isfinite(observed).all() or (observed < 0).any():
            raise ValueError(f'{column} must contain finite, nonnegative values or missing values')

    for column in CATEGORICAL:
        values = X[column].astype('string').str.strip().replace('', pd.NA)
        X[column] = values.astype(object).where(values.notna(), np.nan)
    y = None
    if mode == "labeled":
        labels = table["y"].astype("string").str.strip()

        if not labels.isin(["no", "yes"]).all():
            raise ValueError("y must contain only 'no' or 'yes'")

        y = labels.map({"no": 0, "yes": 1}).astype("int64")
    return ids, X, y


def build_pipeline(candidate_id):
    if {'y', 'record_id'} & set(NUMERIC + CATEGORICAL):
        raise ValueError('y and record_id cannot be model features')
    if candidate_id not in {'prior', 'lr_c1', 'lr_c01', 'rf_d8'}:
        raise ValueError(f'Unknown candidate: {candidate_id}')
    if candidate_id == 'prior':
        return Pipeline([('classify', DummyClassifier(strategy='prior'))])

    numeric_steps = [('impute', SimpleImputer(strategy='median', keep_empty_features=True)),]
    if candidate_id in {'lr_c1', 'lr_c01'}:
        numeric_steps.append(('scale', StandardScaler()))
        classifier = LogisticRegression(
            C=1.0 if candidate_id == 'lr_c1' else 0.1,
            solver='lbfgs', max_iter=2000,
        )
    else:
        classifier = RandomForestClassifier(
            n_estimators=150, max_depth=8, min_samples_leaf=10,
            random_state=42, n_jobs=1,
        )

    categorical_pipeline = Pipeline([
        ('impute', SimpleImputer(
            strategy='constant', fill_value='unknown', keep_empty_features=True,
        )),
        ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ('num', Pipeline(numeric_steps), NUMERIC),
        ('cat', categorical_pipeline, CATEGORICAL),
    ], remainder='drop')

    return Pipeline([
        ('prepare', preprocessor),
        ('classify', classifier),
    ])