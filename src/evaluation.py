from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, precision_score,\
    recall_score, roc_auc_score, f1_score, classification_report,\
    roc_curve, auc
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


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


def compare_models(models, names, x, y):
    """
    Draw table with model and their performance.

    :param models: list of trained models.
    :param names: list of names of models.
    :param x: dataframe with data for prediction.
    :param y: dataframe with expected labels.
    """
    max_len = np.max([len(x) for x in names])
    print("      ".ljust(max_len) + "     Accuracy   F1 (micro)   F1 (macro)"
                                    "   Precission    Recall    AUC ROC")
    for i, model in enumerate(models):
        y_pred = model.predict(x)
        print(f"{names[i]}" + "".ljust(max_len-len(names[i])) + f": "
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
    plot_confusion_matrix(
        model, x, y, cmap=plt.cm.Blues, normalize='true'
    )
    print(classification_report(y, y_pred))
    print(f'ROC AUC score: {round(roc_auc_score(y, y_pred), 2)}')


def roc_auc(y_pred, y_true, plot=True, label="curve"):
    """
    Draw ROC curve plot.

    :param y_pred: predicted labels.
    :param y_true: true labels.
    :param plot: if True, ROC curve plot is drawn.
    :param label: label of a plot.
    :return: ROC AUC score.
    """
    prob = y_pred / y_pred.max()
    fpr, tpr, _ = roc_curve(y_true, prob, drop_intermediate=True)
    auc_value = auc(fpr, tpr)

    if plot:
        plt.scatter(x=fpr, y=tpr, color='navy')
        plt.plot(
            fpr, tpr,
            c=tuple(np.random.rand(3, 1)[:, 0]),
            lw=2,
            label=f'{label} (AUC = {round(auc_value, 3)})'
        )
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()

    return auc_value
