from sklearn.base import TransformerMixin
from src.decoratos import transformer_time_calculation_decorator


class ColumnsFilter(TransformerMixin):
    """
    Transformer to drop columns of the dataframe.

    :param columns: list, list of columns to drop.
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, df, y=None, **fit_params):
        return self

    @transformer_time_calculation_decorator('ColumnsFilter')
    def transform(self, df, **transform_params):
        df = df.drop(self.columns, axis=1)
        return df
