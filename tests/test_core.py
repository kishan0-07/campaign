import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
import helpers
from helpers import CATEGORICAL, FEATURES, NUMERIC, build_pipeline, prepare_inputs

@pytest.fixture
def raw():
    return pd.DataFrame({
        'record_id': [0, 1, 2, 3],
        'age': [20, 40, np.nan, 60],
        'previous': [0, 1, 2, 3],
        'job': ['admin.', 'technician', None, 'admin.'],
        'marital': ['single', 'married', 'single', 'married'],
        'education': ['high.school', 'university.degree', 'high.school', 'high.school'],
        'housing': ['yes', 'no', 'yes', 'no'],
        'loan': ['no', 'no', 'yes', 'no'],
        'poutcome': ['nonexistent', 'failure', 'success', 'nonexistent'],
        'y': ['no', 'yes', 'no', 'yes'],
        'duration': [90, 100, 110, 120],
    })


def test_input_contract_and_cleaning(raw):
    raw = raw.set_axis([17, 4, 9, 2])
    raw['age'] = raw['age'].astype(object)
    raw.loc[17, 'age'] = ' 20 '
    raw.loc[17, 'job'] = ' unknown '
    raw.loc[4, 'job'] = '   '
    before = raw.copy(deep=True)
    ids, X, y = prepare_inputs(raw)

    assert list(X.columns) == FEATURES
    assert not {'y', 'record_id', 'duration'} & set(X.columns)
    assert ids.tolist() == [0, 1, 2, 3]
    assert ids.index.equals(X.index) and X.index.equals(y.index)
    assert y.tolist() == [0, 1, 0, 1]
    assert all(X[column].dtype == np.dtype('float64') for column in NUMERIC)
    assert X.loc[17, 'job'] == 'unknown'
    assert pd.isna(X.loc[4, 'job'])
    pd.testing.assert_frame_equal(raw, before)


def test_blanks_are_missing_but_literal_categories_are_preserved(raw):
    raw['age'] = raw['age'].astype(object)
    raw.loc[0, 'age'] = '   '
    raw.loc[0, 'job'] = ' NA '
    _, X, _ = prepare_inputs(raw)
    assert pd.isna(X.loc[0, 'age'])
    assert X.loc[0, 'job'] == 'NA'


def test_all_missing_numeric_feature_keeps_its_column(raw):
    raw['age'] = np.nan
    _, X, y = prepare_inputs(raw)
    model = build_pipeline('lr_c1').fit(X, y)
    num = model.named_steps['prepare'].named_transformers_['num']
    assert num.named_steps['impute'].statistics_[0] == 0
    assert num.transform(X[NUMERIC]).shape == (len(raw), len(NUMERIC))
    assert np.isfinite(model.predict_proba(X)).all()


@pytest.mark.parametrize('column', NUMERIC)
@pytest.mark.parametrize('value', ['bad-number', np.inf, -np.inf, -1])
def test_invalid_numeric_values_fail(raw, column, value):

    raw[column] = raw[column].astype(object)
    raw.loc[0, column] = value
    with pytest.raises(ValueError):
        prepare_inputs(raw)


@pytest.mark.parametrize('column', FEATURES + ['record_id', 'y'])
def test_missing_columns_fail(raw, column):
    with pytest.raises(ValueError, match='Missing required columns'):
        prepare_inputs(raw.drop(columns=column))


def test_prediction_contract(raw):
    with pytest.raises(ValueError, match='must not contain y'):
        prepare_inputs(raw, mode='predict')
    ids, X, y = prepare_inputs(raw.drop(columns='y'), mode='predict')
    assert y is None and len(ids) == len(X) == len(raw)


def test_duplicate_ids_fail(raw):
    raw.loc[1, 'record_id'] = raw.loc[0, 'record_id']
    with pytest.raises(ValueError, match='must be unique'):
        prepare_inputs(raw)

def test_invalid_labels_fail(raw):
    raw.loc[0, 'y'] = 'maybe'
    with pytest.raises(ValueError, match='only'):
        prepare_inputs(raw)


def test_numeric_statistics_and_heldout_boundary(raw):
    _, X, y = prepare_inputs(raw)
    model = build_pipeline('lr_c1')
    model.fit(X, y)
    num = model.named_steps['prepare'].named_transformers_['num']
    imputer = num.named_steps['impute']
    scaler = num.named_steps['scale']

    expected_medians = X[NUMERIC].median()
    expected_imputed = X[NUMERIC].fillna(expected_medians)
    np.testing.assert_allclose(imputer.statistics_, expected_medians)
    np.testing.assert_allclose(scaler.mean_, expected_imputed.mean())
    np.testing.assert_allclose(scaler.var_, expected_imputed.var(ddof=0))

    before = [imputer.statistics_.copy(), scaler.mean_.copy(), scaler.var_.copy()]
    encoder = model.named_steps['prepare'].named_transformers_['cat'].named_steps['encode']
    categories_before = [values.copy() for values in encoder.categories_]
    heldout = raw.iloc[[0]].drop(columns='y').assign(age=1_000_000, job='new_job')
    _, X_heldout, _ = prepare_inputs(heldout, mode='predict')
    probabilities = model.predict_proba(X_heldout)

    assert probabilities.shape == (1, 2)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    for actual, expected in zip([imputer.statistics_, scaler.mean_, scaler.var_], before):
        np.testing.assert_allclose(actual, expected)
    for actual, expected in zip(encoder.categories_, categories_before):
        np.testing.assert_array_equal(actual, expected)


def test_reordered_columns_give_same_scores(raw):
    _, X, y = prepare_inputs(raw)
    model = build_pipeline('lr_c1').fit(X, y)
    _, reordered, _ = prepare_inputs(raw.loc[:, raw.columns[::-1]])
    pd.testing.assert_frame_equal(X, reordered)
    np.testing.assert_allclose(
        model.predict_proba(X),
        model.predict_proba(reordered.loc[:, FEATURES[::-1]]),
    )


@pytest.mark.parametrize('forbidden', ['y', 'record_id'])
def test_forbidden_model_feature_fails(monkeypatch, forbidden):
    monkeypatch.setattr(helpers, 'NUMERIC', NUMERIC + [forbidden])
    with pytest.raises(ValueError, match='cannot be model features'):
        build_pipeline('lr_c1')


@pytest.mark.parametrize('candidate_id', ['prior', 'lr_c1', 'lr_c01', 'rf_d8'])
def test_factory_returns_fresh_unfitted_models(raw, candidate_id):
    first = build_pipeline(candidate_id)
    second = build_pipeline(candidate_id)
    assert first is not second
    assert first.named_steps['classify'] is not second.named_steps['classify']
    if candidate_id != 'prior':
        assert first.named_steps['prepare'] is not second.named_steps['prepare']
    with pytest.raises(NotFittedError):
        check_is_fitted(first)
    _, X, y = prepare_inputs(raw)
    first.fit(X, y)
    with pytest.raises(NotFittedError):
        check_is_fitted(second)
    if candidate_id != 'prior':
        with pytest.raises(NotFittedError):
            check_is_fitted(second.named_steps['prepare'])
    assert np.isfinite(first.predict_proba(X)).all()


def test_unknown_candidate_fails():
    with pytest.raises(ValueError, match='Unknown candidate'):
        build_pipeline('not_a_candidate')