import os
import glob
import logging
import argparse
import itertools
from functools import reduce

import yaml
import numpy as np
import tensorflow.keras.models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from preprocess import preprocess
from load_files import load_npz_files
from loss_function import weighted_categorical_cross_entropy
from evaluation import f1_scores_from_cm, plot_confusion_matrix
from models import SingleSalientModel, TwoSteamSalientModel


def parse_args():
    """
    parser arguments and setting log formats
    :return: the arguments after parse
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", "-d", default="./data/sleepedf-2013/npzs", help="where the data is.")
    parser.add_argument("--modal", '-m', default='1',
                        help="the way to training model\n\t0: single modal.\n\tmulti modal.")
    parser.add_argument("--output_dir", '-o', default='./result', help="where you wish to set the results.")
    parser.add_argument("--valid", '-v', default='20', help="v stands for k-fold validation's k.")

    args = parser.parse_args()

    k_folds = eval(args.valid)
    if not isinstance(k_folds, int):
        logging.critical("the argument type `valid` should be an integer")
        print("ERROR: get an invalid `k_fold`")
        exit(-1)
    if k_folds <= 0:
        logging.critical(f"get an invalid `k_folds`: {k_folds}")
        print(f"ERROR: the `k_fold` should be positive, but get: {k_folds}")
        exit(-1)

    modal = eval(args.modal)
    if not isinstance(modal, int):
        logging.critical("the argument `modal` ought to be an integer")
        print("ERROR: get an invalid type `modal`")
        exit(-1)
    if modal != 1 and modal != 0:
        logging.critical(f"get an invalid `modal`: {modal}")
        print(f"ERROR: the `modal` ought to between 0 and 1, but get {modal}")
        exit(-1)

    return args


def summary_models(args: argparse.Namespace, hyper_params: dict):
    """
    summary the models
    :param args: the argument from command line input
    :param hyper_params: a dict contain model's hyper parameters
    """
    modal = eval(args.modal)
    k_folds = eval(args.valid)
    res_dir = args.output_dir

    with np.load(os.path.join(res_dir, "split.npz"), allow_pickle=True) as f:
        npz_names = f['split']

    model_names = glob.glob(os.path.join(res_dir, "fold_*_best_model.h5"))
    if len(model_names) < k_folds:
        logging.critical(f"don't have enough models to summary, need {k_folds} but only {len(model_names)}")
        exit(-1)
    model_names.sort()

    loss = weighted_categorical_cross_entropy(hyper_params['class_weights'])

    best_turn_f1, best_turn_acc = 0.0, 0.0
    best_turn_name = ''
    cm_list = []

    if modal == 0:
        eva_model: tensorflow.keras.models.Model = SingleSalientModel(**hyper_params)
    else:
        eva_model: tensorflow.keras.models.Model = TwoSteamSalientModel(**hyper_params)

    eva_model.compile(optimizer=hyper_params['optimizer'], loss=loss, metrics=['acc'])

    for i in range(k_folds):
        # load training weights
        eva_model.load_weights(model_names[i])

        # load and process test data
        test_npzs = list(itertools.chain.from_iterable(npz_names[i].tolist()))
        test_data_list, test_labels_list = load_npz_files(test_npzs)
        test_labels_list = [to_categorical(f) for f in test_labels_list]

        test_data, test_labels = preprocess(test_data_list, test_labels_list, hyper_params['preprocess'], True)

        logging.info(f"evaluate {os.path.basename(model_names[i])} with {test_data.shape[1]} samples")

        y_pred = np.array([])
        if modal == 0:
            y_pred: np.ndarray = eva_model.predict(test_data[0], batch_size=hyper_params['train']['batch_size'])
        elif modal == 1:
            y_pred: np.ndarray = eva_model.predict([test_data[0], test_data[1]],
                                                   batch_size=hyper_params['train']['batch_size'])

        y_pred = y_pred.reshape((-1, 5))
        test_labels = test_labels.reshape((-1, 5))

        acc = accuracy_score(test_labels.argmax(axis=1), y_pred.argmax(axis=1))
        f1 = f1_score(test_labels.argmax(axis=1), y_pred.argmax(axis=1), average='macro')
        cm = confusion_matrix(test_labels.argmax(axis=1), y_pred.argmax(axis=1))

        cm = np.array(cm)
        print(f"{os.path.basename(model_names[i])}'s accuracy is {acc:.4}, the f1-score is {f1:.4}")
        print("and the confusion matrix:")
        print(cm.astype('float32') / np.sum(cm).astype('float32'))
        plot_confusion_matrix(cm, classes=hyper_params['evaluation']['label_class'], title=f"cm_{i + 1}", path=res_dir)
        plot_confusion_matrix(cm, classes=hyper_params['evaluation']['label_class'],
                              normalize=False, title=f"cm_num_{i + 1}", path=res_dir)

        if f1 > best_turn_f1:
            best_turn_f1, best_turn_acc = f1, acc
            best_turn_name = os.path.basename(model_names[i])

        cm_list.append(cm)
        logging.info(f"evaluate {os.path.basename(model_names[i])} completed.")
        eva_model.reset_states()

    print(f"the best model is {best_turn_name} with accuracy={best_turn_acc} and f1-score={best_turn_f1}")

    sum_cm = reduce(lambda x, y: x + y, cm_list)
    plot_confusion_matrix(sum_cm, classes=hyper_params['evaluation']['label_class'], title='cm_total', path=res_dir)
    plot_confusion_matrix(sum_cm, classes=hyper_params['evaluation']['label_class'], title='cm_total_num',
                          normalize=False, path=res_dir)
    ave_f1 = f1_scores_from_cm(sum_cm)
    ave_acc = np.sum(np.diagonal(sum_cm)) / np.sum(sum_cm)
    print(f"the average accuracy: {ave_acc} and the average f1-score: {ave_f1}")


if __name__ == '__main__':
    args = parse_args()
    with open("hyperparameters.yaml", encoding='utf-8') as f:
        hyper_params = yaml.full_load(f)

    summary_models(args, hyper_params)
