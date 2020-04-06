import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_dataset():
    """
    Load dataset with custom data types.
    
    :return: loaded dataframe.
    """
    # Only needed changes in data types are listed
    data_types = {
        'admission_type_id': object,
        'discharge_disposition_id': object,
        'admission_source_id': object
    }

    return pd.read_csv(
        '../data/data.csv',
        na_values='?',
        low_memory=False,
        dtype=data_types
    )


def split_dataframe(x, y, test_size=0.2):
    """
    Split dataframe into train and test subsets.

    :param x: dataframe with attributes for training (independent
        variables).
    :param y: dataframe with label attribute (dependent variable).
    :param test_size: size of test subset (default: 0.2).
    :return: x_train, x_test, y_train, y_test subsets.
    """
    return train_test_split(x, y, test_size=test_size, random_state=42)


def transform_label(y):
    """
    Transform label column to binary classification.

    Original label column contain three values: `<30`, `>30` and `NO`,
    so those columns are transformed into just `1` and `0` - if patient
    was early readmitted or not.

    :param y: dataframe with label attribute (dependent variable).
    :return: dataframe with transformed label.
    """
    return y.apply(lambda x: 1 if x == '<30' else 0)


def describe_dataset(X_train, X_test, y_train, y_test):
    """
    Desrcribe dataset.

    :param X_train: dataframe with train data.
    :param X_test: dataframe with test data.
    :param y_train: dataframe with train labels.
    :param y_train: dataframe with test labels.
    """
    print("Number of train data: ", X_train.shape[0])
    print("Number of test data:  ", X_test.shape[0])
    categories = y_train['0'].unique()
    columns = X_train.columns.shape[0]
    print("With ", columns, " features.")
    print("In ", categories.shape[0], " classes:")
    for c in categories:
        num = (y_train['0'] == c).sum()
        print('              ', str(c), ': ', num,
              ' samples, ', f'{num / y_train["0"].shape[0]:.2f}%')

