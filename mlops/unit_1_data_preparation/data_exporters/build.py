from typing import Tuple
from pandas import DataFrame, Series
from scipy.sparse._csr import csr_matrix
from sklearn.base import BaseEstimator

from mlops.unit_1_data_preparation.utils.data_preparation.encoders import vectorize_features
from mlops.unit_1_data_preparation.utils.data_preparation.feature_selector import select_features

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test



@data_exporter
def export_data(
    data: Tuple[DataFrame, DataFrame, DataFrame], *args, **kwargs) ->Tuple[
        csr_matrix,
        csr_matrix,
        csr_matrix,
        Series,
        Series,
        Series,
        BaseEstimator]:
    """
    Exports data to from dataframes into matrices for features, pd.Series for the target values, and a BaseEstimator for the dictionary vectoriser.

    Args:
        data: The output from the upstream parent block, i.e. the three dataframes (df, df_train, and df_val)
        args: The output from any additional upstream blocks (if applicable) - I believe these should be specified as the global variables

    Output (optional):
        X: The combined matrix of X values (both train and val)
        X_test: The matrix of X_train values
        X_val: The combined matrix of X_val values
        y: The series of y values (both train and val)
        y_train: The y values for the train dataframe
        y_val: The y values for the validation dataframe
    """
    df, df_train, df_val = data
    target = kwargs.get('target','duration')

    X, _, _ = vectorize_features(select_features(df))
    y: Series = df[target]

    X_train, X_val, dv = vectorize_features(
        select_features(df_train),
        select_features(df_val)
    )
    y_train = df_train[target]
    y_val = df_val[target]

    return X, X_train, X_val, y, y_train, y_val, dv


@test
def test_dataset(
    X: csr_matrix,
    X_train: csr_matrix,
    X_val: csr_matrix,
    y: Series,
    y_train: Series,
    y_val: Series,
    *args,
) -> None:
    assert (
        X.shape[0] == 105870
    ), f'Entire dataset should have 105,870 examples, but it has {X.shape[0]}'
    assert(
        X.shape[1] == 7027
    ), f'Entire dataset should have 7,027 features, but it has {X.shape[1]}'
    assert(
        len(y.index) == X.shape[0]
    ), f'There are not the same number of y examples and X examples, there are {len(y.index)} y values and {X.shape[0]} X values'

@test
def test_training_set(
    X: csr_matrix,
    X_train: csr_matrix,
    X_val: csr_matrix,
    y: Series,
    y_train: Series,
    y_val: Series,
    *args,
) -> None:
    assert (
        X_train.shape[0] == 54378
    ), f'X_train should have 105,870 examples, but it has {X_train.shape[0]}'
    assert(
        X_train.shape[1] == 5094
    ), f'X_train should have 7,027 features, but it has {X_train.shape[1]}'
    assert(
        len(y_train.index) == X_train.shape[0]
    ), f'There are not the same number of y_train examples and X_train examples, there are {len(y_train.index)} y_train values and {X_train.shape[0]} X_train values'

@test
def test_validation_set(
    X: csr_matrix,
    X_train: csr_matrix,
    X_val: csr_matrix,
    y: Series,
    y_train: Series,
    y_val: Series,
    *args,
) -> None:
    assert (
        X_val.shape[0] == 51492
    ), f'X_val should have 51,492 examples, but it has {X_val.shape[0]}'
    assert(
        X_val.shape[1] == 5094
    ), f'X_val should have 5,094 features, but it has {X_val.shape[1]}'
    assert(
        len(y_val.index) == X_val.shape[0]
    ), f'There are not the same number of y_val examples and X_val examples, there are {len(y_val.index)} y_val values and {X_val.shape[0]} X_val values'

