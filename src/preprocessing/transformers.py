from sklearn.base import TransformerMixin
from src.decoratos import transformer_time_calculation_decorator


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
