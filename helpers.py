import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    log_loss, precision_score, recall_score,
)


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


def positive_scores(fitted_pipeline, X):

    column = np.flatnonzero(fitted_pipeline.classes_ == 1)
    if len(column) != 1:
        raise ValueError('The fitted model must have a class labeled 1')
    return fitted_pipeline.predict_proba(X)[:, column[0]]


def _record_ids(values):

    ids = pd.to_numeric(pd.Series(values).reset_index(drop=True), errors='raise')
    if ids.empty:
        return ids.astype('int64')
    if ids.isna().any() or not pd.api.types.is_integer_dtype(ids.dtype):
        raise ValueError('record_id must contain nonmissing integers')
    if not ids.is_unique:
        raise ValueError('record_id values must be unique')
    return ids


def rank_customers(ids, probabilities, policy):
    rule = (policy['budget_numerator'], policy['budget_denominator'], policy['tie_break'])
    if rule != (1, 10, 'record_id_ascending'):
        raise ValueError('Unsupported project ranking policy')
    ids = _record_ids(ids)
    probabilities = np.asarray(probabilities, dtype=float)
    if (probabilities.shape != (len(ids),)
            or not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError('Provide one finite probability in [0,1] per record_id')

    ranked = pd.DataFrame({
        'record_id': ids.to_numpy(),
        'estimated_probability': probabilities,
    }).sort_values(
        ['estimated_probability', 'record_id'],
        ascending=[False, True], ignore_index=True,
    )
    k = len(ranked) * policy['budget_numerator'] // policy['budget_denominator']
    ranked['rank'] = np.arange(1, len(ranked) + 1)
    ranked['selected'] = ranked['rank'] <= k
    return ranked


def evaluate_ranking(ids, probabilities, labels, policy):
    ranked = rank_customers(ids, probabilities, policy)
    if ranked.empty:
        raise ValueError('Cannot evaluate an empty cohort')
    if not labels.columns.is_unique:
        raise ValueError('Duplicate label column names are not allowed')
    labels = labels.loc[:, ['record_id', 'y']].reset_index(drop=True).copy()
    labels['record_id'] = _record_ids(labels['record_id'])
    labels['y'] = pd.to_numeric(labels['y'], errors='raise')
    if not labels['y'].isin([0, 1]).all():
        raise ValueError('Evaluation labels must be nonmissing binary 0/1 values')

    aligned = ranked.merge(
        labels, on='record_id', how='outer',
        validate='one_to_one', indicator=True,
    )
    if not aligned['_merge'].eq('both').all():
        raise ValueError('Label IDs must match the scored cohort exactly')

    y = aligned['y'].to_numpy(dtype=int)
    p = aligned['estimated_probability'].to_numpy(dtype=float)
    selected = aligned['selected'].to_numpy(dtype=int)
    n, positives, k = len(y), int(y.sum()), int(selected.sum())
    tn, fp, fn, tp = confusion_matrix(y, selected, labels=[0, 1]).ravel().tolist()
    precision = float(precision_score(y, selected)) if k else None
    recall = float(recall_score(y, selected)) if positives else None
    lift = precision / (positives / n) if k and positives else None
    ap = float(average_precision_score(y, p)) if positives else None

    undefined = {}
    if k == 0:
        undefined['precision_at_budget'] = 'k=0: no records selected'
    if positives == 0:
        undefined['recall_at_budget'] = 'P=0: no positive labels'
        undefined['ap'] = 'P=0: no positive labels'
    if k == 0 or positives == 0:
        undefined['lift'] = 'Lift requires k>0 and P>0'

    random_undefined = {}
    if k == 0:
        random_undefined['expected_precision'] = 'k=0: precision is undefined'
    if positives == 0:
        random_undefined['expected_recall'] = 'P=0: recall is undefined'
    if k == 0 or positives == 0:
        random_undefined['expected_lift'] = 'Lift requires k>0 and P>0'

    return {
        'n': n, 'k': k, 'positives': positives,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision_at_budget': precision, 'recall_at_budget': recall,
        'lift': lift, 'ap': ap,
        'brier': float(brier_score_loss(y, p, pos_label=1)),
        'log_loss': float(log_loss(y, p, labels=[0, 1])),
        'undefined': undefined,
        'random_reference': {
            'expected_tp': k * positives / n,
            'expected_precision': positives / n if k else None,
            'expected_recall': k / n if positives else None,
            'expected_lift': 1.0 if k and positives else None,
            'undefined': random_undefined,
        },
    }
