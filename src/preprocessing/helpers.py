from sklearn.model_selection import train_test_split


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
