from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def undersample(x, y):
    """
    Perform undersampling by randomly choosing samples.

    :param x: dataframe with attributes for training (independent
        variables).
    :param y: dataframe with label attribute (dependent variable).
    :return: x and y after undersampling performed.
    """
    rus = RandomUnderSampler(random_state=42)
    return rus.fit_resample(x, y)


def oversample(x, y):
    """
    Perform oversampling using SMOTE method.

    :param x: dataframe with attributes for training (independent
        variables).
    :param y: dataframe with label attribute (dependent variable).
    :return: x and y after oversampling performed.
    """
    smote = SMOTE(random_state=42)
    return smote.fit_resample(x, y)
