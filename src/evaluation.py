import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, precision_score,\
    recall_score, roc_auc_score, f1_score, classification_report,\
    roc_curve, auc
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import learning_curve


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


def plot_learning_curve(
        estimator, title, X, y, cv=None, train_sizes=np.linspace(.1, 1.0, 10)
):
    """
    Plot learning curve of classifier.

    Function taken and edited from:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    :param estimator: classifier of which learning curve will be drawn.
    :param title: title of learning curve plot.
    :param X: training data features.
    :param y: training data labels.
    :param cv: cross-validation (estimator or number of folds).
    :param train_sizes: train splits sizes.
    :return:
    """
    plt.figure()
    plt.title(title)
    plt.ylim(0.45, 1.01)
    plt.xlabel('Number of samples', labelpad=20)
    plt.ylabel('Score', labelpad=20)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label='Train score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label='Cross-validation score')

    plt.legend(loc='best')
    return plt


class CombinedModel:
    """
    CombinedModel class serve for combining several model to obtain one
    prediction. Model is selected to prediction with rule defined in ruler.

    :param models: List of models
    :param ruler: function that return index of model to by used. Parameter
    of the function is one row from dataframe.
    """
    def __init__(self, models, ruler):
        super().__init__()
        self.models = models
        self.ruler = ruler

    def fit(self, X, y=None):
        return self

    def predict(self, x):
        pred = np.zeros(x.shape[0])
        for i, model in enumerate(self.models):
            idxs = x[x.apply(self.ruler, axis=1) == i].index
            pred[idxs] = model.predict(x.loc[idxs])
        return pred