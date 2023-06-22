import numpy as np
import matplotlib.pyplot as plt


def f1_scores_from_cm(cm: np.ndarray) -> np.ndarray:
    """
    Using confusion matrix to calculate the f1 score of every class

    :param cm: the confusion matrix

    :return: the f1 score
    """

    def get_tp_rel_sel_from_cm(cm: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        calculate the diagonal, column sum and row sum of the matrix
        """
        tp = np.diagonal(cm)
        rel = np.sum(cm, axis=0)
        sel = np.sum(cm, axis=1)
        return tp, rel, sel

    def precision_scores_from_cm(cm: np.ndarray) -> np.ndarray:
        """
        calculate the precision of matrix
        """
        tp, rel, sel = get_tp_rel_sel_from_cm(cm)
        sel_mask = sel > 0
        precision_score = np.zeros(shape=tp.shape, dtype=np.float32)
        precision_score[sel_mask] = tp[sel_mask] / sel[sel_mask]
        return precision_score

    def recall_scores_from_cm(cm: np.ndarray) -> np.ndarray:
        """
        calculate the recall of matrix
        """
        tp, rel, sel = get_tp_rel_sel_from_cm(cm)
        rel_mask = rel > 0
        recall_score = np.zeros(shape=tp.shape, dtype=np.float32)
        recall_score[rel_mask] = tp[rel_mask] / rel[rel_mask]
        return recall_score

    precisions = precision_scores_from_cm(cm)
    recalls = recall_scores_from_cm(cm)

    # prepare arrays
    dices = np.zeros_like(precisions)

    # Compute dice
    intrs = (2 * precisions * recalls)
    union = (precisions + recalls)
    dice_mask = union > 0
    dices[dice_mask] = intrs[dice_mask] / union[dice_mask]
    return dices


def plot_confusion_matrix(cm: np.ndarray, classes=None,
                          normalize: bool = True, title: str = None,
                          cmap: str = 'Blues', path: str = '',
                          fname: str = "", fontsize: int = 24):
    """
    Draw a diagram of confusion matrix

    :param cm: the confusion matrix
    :param classes: a list of str for every class' name
    :param normalize: decide use decimals to show or not
    :param title: to give the diagram a title
    :param cmap: the color map of diagram
    :param path: the save path of diagram
    """
    if classes is None:
        classes = ["W", "N1", "N2", "N3", "REM"]
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    '''for training activation'''

    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    #     acc_list = []
    #     for i in range(5):
    #         acc_list.append(round(cm[i][i], 2))
    #     print("the accuracy of every classes:{}".format(acc_list))
    # else:
    #     print('Confusion matrix, without normalization')
    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes,
           title=title, ylabel='True label', xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = np.max(cm) / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=13)
    print(path + fname + '.pdf')
    fig.tight_layout()
    plt.savefig(path + fname + '.pdf')


def draw_training_plot(history: dict, from_fold: int, train_folds: int, output_path: str):
    """
    a function to draw a plot of training and validating metrics: accuracy and loss
    :param history: the model fit's output that including the accuracy and loss
    :param from_fold: the first fold to train
    :param train_folds: the trained folds number
    :param output_path: the results' dir
    """
    fig: plt.Figure = plt.figure(figsize=(15, 6 * train_folds))
    acc, val_acc = history['acc'], history['val_acc']
    loss, val_loss = history['loss'], history['val_loss']

    for i in range(len(acc)):
        epochs = range(1, len(acc[i]) + 1)
        ax: plt.Axes = fig.add_subplot(train_folds, 2, 2 * i + 1)
        ax.plot(epochs, acc[i], 'C0-', label='Training Accuracy')
        ax.plot(epochs, val_acc[i], 'C1-.', label='Validation Accuracy')
        ax.set_title(f'Training and Validation Accuracy in fold-{from_fold + i}')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        ax.legend()

        ax: plt.Axes = fig.add_subplot(train_folds, 2, 2 * i + 2)
        ax.plot(epochs, loss[i], 'C0-', label='Training Loss')
        ax.plot(epochs, val_loss[i], 'C1-.', label='Validation Loss')
        ax.set_title(f"Training and Validation Loss in fold-{from_fold + i}")
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        ax.legend()

    import os
    fig.savefig(os.path.join(output_path, f"f{from_fold}-{from_fold + train_folds}_accuracy_and_loss.png"))
