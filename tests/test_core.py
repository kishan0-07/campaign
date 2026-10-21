import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
import helpers , json
from helpers import CATEGORICAL, FEATURES, NUMERIC, build_pipeline, evaluate_ranking, positive_scores, prepare_inputs, rank_customers

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


@pytest.fixture
def policy():
    return {'budget_numerator': 1, 'budget_denominator': 10, 'tie_break': 'record_id_ascending'}


@pytest.fixture
def metric_fixture():
    ids = np.arange(20)
    labels = pd.DataFrame({'record_id': ids, 'y': np.isin(ids, [0, 1, 10, 11]).astype(int)})
    return ids, (20 - ids) / 21, labels


def test_known_answer_metrics(metric_fixture, policy):
    ids, scores, labels = metric_fixture
    ranked = rank_customers(ids, scores, policy)
    result = evaluate_ranking(ids, scores, labels, policy)
    assert ranked.loc[ranked['selected'], 'record_id'].tolist() == [0, 1]
    for key, expected in {
        'n': 20, 'k': 2, 'positives': 4, 'tp': 2, 'fp': 0, 'fn': 2, 'tn': 16,
        'precision_at_budget': 1, 'recall_at_budget': .5, 'lift': 5,
    }.items():
        assert result[key] == pytest.approx(expected)
    assert result['ap'] == pytest.approx((1 + 1 + 3 / 11 + 4 / 12) / 4)
    y = labels['y'].to_numpy()
    assert result['brier'] == pytest.approx(np.mean((scores - y) ** 2))
    assert result['log_loss'] == pytest.approx(-np.mean(y * np.log(scores) + (1 - y) * np.log1p(-scores)))
    assert result['random_reference'] == {
        'expected_tp': .4, 'expected_precision': .2,
        'expected_recall': .1, 'expected_lift': 1.0, 'undefined': {},
    }
    json.dumps(result, allow_nan=False)


def test_independently_reordered_labels(metric_fixture, policy):
    ids, scores, labels = metric_fixture
    original = evaluate_ranking(ids, scores, labels, policy)
    order = np.random.default_rng(42).permutation(len(ids))
    shuffled_labels = labels.sample(frac=1, random_state=7).reset_index(drop=True)
    assert evaluate_ranking(ids[order], scores[order], shuffled_labels, policy) == original


def test_nondefault_indexes_do_not_change_id_score_pairs(metric_fixture, policy):
    ids, scores, _ = metric_fixture
    pd.testing.assert_frame_equal(
        rank_customers(pd.Series(ids, index=range(100, 120)), pd.Series(scores, index=range(200, 220)), policy),
        rank_customers(ids, scores, policy),
    )


def test_ties_use_numeric_ids(policy):
    ids = ['10', '2', '30', '40', '50', '60', '70', '80', '90', '100']
    ranked = rank_customers(ids, np.full(10, .5), policy)
    assert ranked['record_id'].head(2).tolist() == [2, 10]
    assert ranked.loc[ranked['selected'], 'record_id'].tolist() == [2]


def test_sorting_uses_unrounded_scores(policy):
    ids = [2, 10, 30, 40, 50, 60, 70, 80, 90, 100]
    ranked = rank_customers(ids, [.50000000000001, .50000000000002] + [.1] * 8, policy)
    assert ranked.iloc[0]['record_id'] == 10


def test_duplicate_numeric_ids_fail(policy):
    with pytest.raises(ValueError, match='unique'):
        rank_customers(['2', 2], [.6, .4], policy)


@pytest.mark.parametrize('invalid', [-.1, 1.1, np.nan, np.inf])
def test_invalid_probabilities_fail(metric_fixture, policy, invalid):
    ids, scores, _ = metric_fixture
    scores[0] = invalid
    with pytest.raises(ValueError):
        rank_customers(ids, scores, policy)


def test_mismatched_score_length_fails(metric_fixture, policy):
    ids, scores, _ = metric_fixture
    with pytest.raises(ValueError):
        rank_customers(ids, scores[:-1], policy)


@pytest.mark.parametrize('defect', ['missing', 'extra', 'duplicate', 'invalid_y'])
def test_invalid_label_tables_fail(metric_fixture, policy, defect):
    ids, scores, labels = metric_fixture
    if defect == 'missing':
        labels = labels.iloc[:-1]
    elif defect == 'extra':
        labels = pd.concat([labels, pd.DataFrame({'record_id': [20], 'y': [0]})], ignore_index=True)
    elif defect == 'duplicate':
        labels = pd.concat([labels, labels.iloc[[0]]], ignore_index=True)
    else:
        labels.loc[0, 'y'] = 2
    with pytest.raises(ValueError):
        evaluate_ranking(ids, scores, labels, policy)


def test_small_batch_has_zero_budget(metric_fixture, policy):
    ids, scores, labels = metric_fixture
    result = evaluate_ranking(ids[:9], scores[:9], labels.iloc[:9], policy)
    assert result['k'] == 0 and result['recall_at_budget'] == 0
    assert result['precision_at_budget'] is None and result['lift'] is None
    assert result['ap'] is not None
    assert result['undefined']['precision_at_budget'] and result['undefined']['lift']
    assert result['random_reference']['expected_precision'] is None
    json.dumps(result, allow_nan=False)


def test_zero_positives_keep_probability_metrics(metric_fixture, policy):
    ids, scores, labels = metric_fixture
    result = evaluate_ranking(ids, scores, labels.assign(y=0), policy)
    assert result['precision_at_budget'] == 0
    for key in ['ap', 'recall_at_budget', 'lift']:
        assert result[key] is None and result['undefined'][key]
    assert np.isfinite(result['brier']) and np.isfinite(result['log_loss'])
    json.dumps(result, allow_nan=False)


def test_all_positive_cohort_is_valid(metric_fixture, policy):
    ids, scores, labels = metric_fixture
    result = evaluate_ranking(ids, scores, labels.assign(y=1), policy)
    assert result['ap'] == pytest.approx(1)
    assert result['recall_at_budget'] == pytest.approx(.1)
    assert result['lift'] == pytest.approx(1)


def test_empty_ranking_has_headers_but_evaluation_fails(policy):
    ranked = rank_customers([], [], policy)
    assert ranked.empty
    assert ranked.columns.tolist() == ['record_id', 'estimated_probability', 'rank', 'selected']
    with pytest.raises(ValueError, match='empty cohort'):
        evaluate_ranking([], [], pd.DataFrame(columns=['record_id', 'y']), policy)


def test_unsupported_policy_fails(metric_fixture, policy):
    ids, scores, _ = metric_fixture
    with pytest.raises(ValueError, match='Unsupported'):
        rank_customers(ids, scores, dict(policy, tie_break='descending'))


@pytest.mark.parametrize('candidate_id', ['prior', 'lr_c1', 'lr_c01', 'rf_d8'])
def test_positive_scores_works_with_completed_c(raw, candidate_id):
    _, X, y = prepare_inputs(raw)
    model = build_pipeline(candidate_id).fit(X, y)
    assert model.classes_.tolist() == [0, 1]
    np.testing.assert_allclose(positive_scores(model, X), model.predict_proba(X)[:, 1])
