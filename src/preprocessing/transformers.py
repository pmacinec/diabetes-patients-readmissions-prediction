from sklearn.base import TransformerMixin
from src.decoratos import transformer_time_calculation_decorator
import pandas as pd

class ColumnsFilter(TransformerMixin):
    """
    Transformer to drop columns of the dataframe.

    :param columns: list of columns to drop.
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, df, y=None, **fit_params):
        return self

    @transformer_time_calculation_decorator('ColumnsFilter')
    def transform(self, df, **transform_params):
        df = df.drop(self.columns, axis=1)
        return df


class RowsFilter(TransformerMixin):
    """
    Transformer to drop rows of the dataframe.

    :param indices: list of indices of samples to drop.
    """

    def __init__(self, indices):
        self.indices = indices

    def fit(self, df, y=None, **fit_params):
        return self

    @transformer_time_calculation_decorator('RowsFilter')
    def transform(self, df, **transform_params):
        df = df.drop(self.indices)
        return df


class ColumnsNanFilter(TransformerMixin):
    """
    Transformer to drop columns of the dataframe with higher range of
    nan values.

    :param threshold: threshold range of nan values.
    """

    def __init__(self, threshold=0.6):
        self.threshold = threshold
        self.columns = []

    def fit(self, df, y=None, **fit_params):
        for column in df.columns:
            threshold = df[column].isna().sum() / df[column].shape[0]
            if threshold > self.threshold:
                self.columns.append(column)
        return self

    @transformer_time_calculation_decorator('ColumnsNanFilter')
    def transform(self, df, **transform_params):
        df = df.drop(self.columns, axis=1)
        return df


class ColumnsValuesDiversityFilter(TransformerMixin):
    """
    Transformer to drop columns of the dataframe with small diversity of
    values for feature.

    :param threshold: ratio of the most frequented value.
    """

    def __init__(self, threshold=1):
        self.columns = []
        self.threshold = threshold

    def fit(self, df, y=None, **fit_params):
        for column in df.columns:
            values = df[column].value_counts()
            threshold = values.max() / values.sum()
            if threshold >= self.threshold:
                self.columns.append(column)
        return self

    @transformer_time_calculation_decorator('ColumnsValuesDiversityFilter')
    def transform(self, df, **transform_params):
        df = df.drop(self.columns, axis=1)
        return df


class OneHotEncoder(TransformerMixin):
    """
    Transformer to create one hot encoding.

    :param columns: list of columns to by encoded.
    """

    def __init__(self, columns=[]):
        self.columns = columns
        self.categories = {}

    def fit(self, df, y=None, **fit_params):
        for column in self.columns:
            self.categories[column] = set([
                str(column) + '_' + str(val) for val in df[column].unique()
            ])
        return self

    @transformer_time_calculation_decorator('OneHotEncoder')
    def transform(self, df, **transform_params):
        for column in self.columns:
            df = pd.concat([df, pd.get_dummies(df[column], prefix=column)],
                           axis=1)
            df = self.add_missing_columns(df, column)
        df = df.drop(self.columns, axis=1)
        return df

    def add_missing_columns(self, df, column):
        columns_to_add = self.categories[column] - set(df.columns)
        for column in columns_to_add:
            df[column] = 0
        return df
