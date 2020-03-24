import numpy as np
import pandas as pd
import string
from functools import reduce
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


class RowsNanFilter(TransformerMixin):
    """
    Transformer to drop rows of dataframe with too many missing values.

    :param threshold: decimal threshold (percentage scaled to 0-1) of
        missing values in row to be dropped.
    """

    def __init__(self, threshold=0.3):
        self.threshold = threshold
        self.num_non_nan = None

    def fit(self, df, y=None, **fit_params):
        self.num_non_nan = round(df.shape[1] * (1 - self.threshold))
        return self

    @transformer_time_calculation_decorator('RowsNanFilter')
    def transform(self, df, **transform_params):
        df = df.dropna(thresh=self.num_non_nan)
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
    Transformer to drop columns of the dataframe with small diversity
    of values for feature.

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
    Transformer to one-hot encode categorical attributes.

    :param columns: list of columns to by encoded.
    """

    def __init__(self, columns):
        self.columns = columns
        self.categories = {}

    def fit(self, df, y=None, **fit_params):
        for column in self.columns:
            self.categories[column] = set([
                f'{str(column)}_{str(val)}' for val in df[column].unique()
            ])
        return self

    @transformer_time_calculation_decorator('OneHotEncoder')
    def transform(self, df, **transform_params):
        df_copy = df.copy()
        for column in self.columns:
            df_copy = pd.concat(
                [df_copy, pd.get_dummies(df_copy[column], prefix=column)],
                axis=1
            )
            df_copy = self.add_missing_encoded_columns(df_copy, column)
        df_copy.drop(self.columns, axis=1, inplace=True)
        return df_copy

    def add_missing_encoded_columns(self, df, column):
        """
        Add missing categorical encoded columns to dataframe.

        :param df: dataframe to add columns into.
        :param column: column
        :return: new dataframe with added missing columns.
        """
        columns_to_add = self.categories[column] - set(df.columns)
        for column in columns_to_add:
            df[column] = 0
        return df


class ValueMapper(TransformerMixin):
    """
    Transformer to map values in column to corresponding values in
    the mapping.

    :param mapping: dictionary with mapping.
    """

    def __init__(self, mapping={}):
        self.mapping = mapping

    def fit(self, df, y=None, **fit_params):
        return self

    @transformer_time_calculation_decorator('ValueMapper')
    def transform(self, df, **transform_params):
        df_copy = df.copy()
        for key in self.mapping.keys():
            column_mapping = self.mapping[key]
            df_copy[key] = df_copy[key].apply(
                lambda x: ValueMapper.get_value(x, column_mapping)
            )
        return df_copy

    @staticmethod
    def get_value(value, mapping):
        """
        Get value from mapping with handling special cases.

        :param value: value to be found in mapping.
        :param mapping: provided mapping of values.
        :return: mapped value or None, if valus is unknown or NaN.
        """
        if pd.isna(value) or value not in mapping.keys():
            return None
        return mapping[value]


class MissingValuesImputer(TransformerMixin):
    """
    Transformer to fill NaN values.

    :param columns: list of columns to replace NaN.
    :param strategy: one of defined strategies (mean, median or
        most_frequent).
    """

    def __init__(self, columns, strategy):
        self.columns = columns
        self.strategy = strategy
        self.mapping = {}

    def fit(self, df, y=None, **fit_params):
        for column in self.columns:
            if self.strategy == 'mean':
                self.mapping[column] = df[column].mean()
            elif self.strategy == 'median':
                self.mapping[column] = df[column].median()
            elif self.strategy == 'most_frequent':
                self.mapping[column] = df[column].mode()[0]
        return self

    @transformer_time_calculation_decorator('MissingValuesImputer')
    def transform(self, df, **transform_params):
        df_copy = df.copy()
        for column in self.columns:
            df_copy[column] = df_copy[column].fillna(self.mapping[column])
        return df_copy


class DiagnosesCodesMapper(TransformerMixin):
    """
    Transformer to map diagnoses codes to diagnoses.

    :param columns: columns with diagnose code.
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, df, y=None, **fit_params):
        self.columns = [col for col in self.columns if col in df.columns]
        return self

    @staticmethod
    def diagnosis_mapping():
        """
        Create code to diagnosis mappings.

        :return: dictionary with mappings.
        """
        mapping = dict()
        mapping['circulatory'] = list(range(390, 460)) + [785]
        mapping['respiratory'] = list(range(460, 520)) + [786]
        mapping['digestive'] = list(range(520, 580)) + [787]
        mapping['diabetes'] = [250]
        mapping['injury'] = list(range(800, 1000))
        mapping['musculoskeletal'] = list(range(710, 740))
        mapping['genitourinary'] = list(range(580, 630)) + [788]
        mapping['neoplasm'] = list(range(140, 240))

        all_codes = reduce(lambda x, y: x + mapping[y], mapping.keys(), [])
        mapping['other'] = [x for x in range(1, 1000) if x not in all_codes]
        mapping['other'] = mapping['other'] + list(string.ascii_uppercase)

        for key in mapping.keys():
            mapping[key] = [str(x) for x in mapping[key]]
        return mapping

    @staticmethod
    def map_code_to_diagnose(code, mapping):
        """
        Map code to diagnose.

        :param code: diagnose code.
        :param mapping: code->diagnose mapping.
        :return: diagnose according to given code.
        """
        code = str(code)
        if not code:
            return None
        for diagnose in mapping.keys():
            if diagnose in ['diabetes', 'other']:
                if any([code.startswith(x) for x in mapping[diagnose]]):
                    return diagnose
                else:
                    continue
            if code in mapping[diagnose]:
                return diagnose

    @transformer_time_calculation_decorator('DiagnosesCodesMapper')
    def transform(self, df, **transform_params):
        df_copy = df.copy()

        mapping = DiagnosesCodesMapper.diagnosis_mapping()

        for column in self.columns:
            df_copy[f'{column}_category'] = df_copy[column].apply(
                lambda x: DiagnosesCodesMapper.map_code_to_diagnose(x, mapping)
            )
        df_copy = df_copy.drop(self.columns, axis=1)
        return df_copy


class NumberVisitsCreator(TransformerMixin):
    """
    Transformer to create number of visits feature.

    :param columns: visits columns to sum.
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, df, y=None, **fit_params):
        return self

    @transformer_time_calculation_decorator('NumberVisitsCreator')
    def transform(self, df, **transform_params):
        df_copy = df.copy()

        df_copy['visits_sum'] = df_copy.loc[:, self.columns].sum(axis=1)

        return df_copy


class NumberMedicamentsChangesCreator(TransformerMixin):
    """
    Transformer to create number of medicaments changes feature.

    :param columns: columns with medicaments.
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, df, y=None, **fit_params):
        self.columns = [col for col in self.columns if col in df.columns]
        return self

    @staticmethod
    def counter(counter_value, medicament):
        """
        Count medicament changes under medicament value condition.

        :param counter_value: actual value of counter.
        :param medicament: concrete medicament value.
        :return: new value of counter (either incremented or not).
        """
        if medicament not in ['No', 'Steady']:
            return counter_value + 1
        return counter_value

    @transformer_time_calculation_decorator('NumberMedicamentsChangesCreator')
    def transform(self, df, **transform_params):
        df_copy = df.copy()
        col_name = 'number_medicaments_changes'

        df_copy[col_name] = 0

        for medicament in self.columns:
            df_copy[col_name] = df_copy.apply(
                lambda x: NumberMedicamentsChangesCreator.counter(
                    x[col_name], x[medicament]
                ),
                axis=1
            )

        return df_copy


class NumberMedicamentsCreator(TransformerMixin):
    """
    Transformer to create number of medicaments feature.

    :param columns: columns with medicaments.
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, df, y=None, **fit_params):
        self.columns = [col for col in self.columns if col in df.columns]
        return self

    @transformer_time_calculation_decorator('NumberMedicamentsCreator')
    def transform(self, df, **transform_params):
        df_copy = df.copy()
        col_name = 'number_medicaments'

        df_copy[col_name] = df_copy[self.columns].apply(
            lambda y: y.apply(lambda x: np.sum(0 if x == 'No' else 1)), axis=1
        ).apply(np.sum, axis=1)

        return df_copy
