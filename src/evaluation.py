import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, precision_score,\
    recall_score, roc_auc_score, f1_score, classification_report,\
    roc_curve, auc
from sklearn.metrics import plot_confusion_matrix


def undersample(x, y):
    """
    Undersampling by randomly selecting samples from majority class.

    :param x: dataframe with attributes for training (independent
        variables).
    :param y: dataframe with label attribute (dependent variable).
    :return: x and y after undersampling performed.
    """
    rus = RandomUnderSampler(random_state=3)
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


def compare_models(models, names, x, y):
    """
    Draw table with model and their performance.

    :param models: list of trained models.
    :param names: list of names of models.
    :param x: dataframe with data for prediction.
    :param y: dataframe with expected labels.
    """
    max_len = np.max([len(x) for x in names])
    print("      ".ljust(max_len) + "     Accuracy   F1 (micro)  F1 (macro)"
                                    "  Precission   Recall    AUC ROC")
    for i, model in enumerate(models):
        y_pred = model.predict(x)
        print(f"{names[i]}" + "".ljust(max_len-len(names[i])) + "   "
              f"|   {accuracy_score(y, y_pred):.2f}   "
              f"|   {f1_score(y, y_pred, average='micro'):.2f}    "
              f"|   {f1_score(y, y_pred, average='macro'):.2f}    "
              f"|   {precision_score(y, y_pred):.2f}    "
              f"|   {recall_score(y, y_pred):.2f}   "
              f"|   {roc_auc_score(y, y_pred):.2f}   |")


def evaluate_model(model, x, y):
    """
    Print evaluation of model.

    :param model: model to be evaluated.
    :param x: dataframe with data for prediction.
    :param y: dataframe with expected labels.
    """
    y_pred = model.predict(x)

    fig, axs = plt.subplots(nrows=1, ncols=2,
                            figsize=(12, 4), constrained_layout=True)
    plot_confusion_matrix(
        model, x, y, cmap=plt.cm.Blues, normalize='true', ax=axs[0]
    )
    roc_auc(y_pred, y, ax=axs[1])
    print(classification_report(y, y_pred))
    print(f'ROC AUC score: {round(roc_auc_score(y, y_pred), 2)}')


def roc_auc(y_pred, y_true, plot=True, label="curve", ax=None):
    """
    Draw ROC curve plot.

    :param y_pred: predicted labels.
    :param y_true: true labels.
    :param plot: if True, ROC curve plot is drawn.
    :param label: label of a plot.
    :param ax: optional axis for plot.
    :return: ROC AUC score.
    """
    prob = y_pred / y_pred.max()
    fpr, tpr, _ = roc_curve(y_true, prob, drop_intermediate=True)
    auc_value = auc(fpr, tpr)

    if plot:
        if not ax:
            fig, ax = plt.subplots()
        ax.scatter(x=fpr, y=tpr, color='navy')
        ax.plot(
            fpr, tpr,
            c=tuple(np.random.rand(3, 1)[:, 0]),
            lw=2,
            label=f'{label} (AUC = {round(auc_value, 3)})'
        )
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc="lower right")

    return auc_value


def plot_feature_importance(
    importance,
    feature_names,
    max_num=-1,
    reverse_order=False
):
    """
    Plot features sorted by importance.

    :param importance: importance of features.
    :param feature_names: names of features.
    :param max_num: maximal number of features to show.
    :param reverse_order: whether importances are in reverse order.
    :return: feature names sorted by importance.
    """
    indexes = np.argsort(importance)
    names = []
    feature_importance = []

    if reverse_order:
        indexes = list(reversed(indexes))

    for i in indexes:
        names.append(feature_names[i])
        feature_importance.append(importance[i])

    plt.figure(figsize=(10, len(feature_names[:max_num]) // 2))
    plt.barh(names[-max_num::], feature_importance[-max_num::])
    plt.yticklabels = names

    return names[::-1]
