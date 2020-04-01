from sklearn.model_selection import train_test_split


def split_dataframe(df, label_col='label', test_size=0.2):
    """
    Split dataframe into train and test subsets.

    :param df: dataframe to be splitted.
    :param label_col: name of column with label.
    :param test_size: size of test subset (default: 0.2).
    :return: x_train, x_test, y_train, y_test subsets.
    """
    x = df.drop(label_col, axis=1)
    y = df[label_col]
    return train_test_split(x, y, test_size=test_size, random_state=42)
