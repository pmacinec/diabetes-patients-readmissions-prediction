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
