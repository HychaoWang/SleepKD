import numpy as np

import tf_GPU
from tensorflow import keras
from os import path
from glob import glob
from data_loader import Loader
from tensorflow.keras.models import load_model, Model
from trainer import get_seq_acc, get_acc
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from imblearn.metrics import specificity_score, sensitivity_score
from evaluation import plot_confusion_matrix

if __name__ == "__main__":
    # workpath = "weights/SleepEDF_model"
    # workpath = "weights/ISRUC_model"
    workpath = "weights/Baseline"
    # workpath = "weights/Baseline/models/pre_DGKD_BASE_student_sleepEDF_128_0.h5"
    # workpath = "/root/autodl-nas/ablation/SleepEDF_model"
    # workpath = "/root/autodl-nas/ablation/ISRUC_model"
    # preType = "pre_epochLoss_student_ISRUCIII_128_*.h5"
    # seqtype = "seq_epochLoss_student_ISRUCIII_128_*.h5"
    # preType = "pre_seqLoss_student_ISRUCIII_128_*.h5"
    # seqtype = "seq_seqLoss_student_ISRUCIII_128_*.h5"
    # preType = "pre_NOSOFT_student_ISRUCIII_128_*.h5"
    # seqtype = "seq_NOSOFT_student_ISRUCIII_128_*.h5"
    # preType = "pre_NOHARD_student_ISRUCIII_128_*.h5"
    # seqtype = "seq_NOHARD_student_ISRUCIII_128_*.h5"
    # preType = "pre_NOHARD_student_ISRUCIII_128_*.h5"
    # seqtype = "seq_NOHARD_student_ISRUCIII_128_*.h5"
    # preType = "pre_logitLoss_student_ISRUCIII_128_*.h5" #TAKD
    # seqtype = "seq_logitLoss_student_ISRUCIII_128_*.h5" #TAKD
    # preType = "pre_DKD_BASE_student_ISRUCIII_128_*.h5"
    # seqtype = "seq_DKD_BASE_student_ISRUCIII_128_*.h5"
    # preType = "pre_KD_BASE_student_ISRUCIII_128_*.h5"
    # seqtype = "seq_KD_BASE_student_ISRUCIII_128_*.h5"
    # preType = "pre_FITNET_BASE_student_ISRUCIII_128_*.h5"
    # seqtype = "seq_FITNET_BASE_student_ISRUCIII_128_*.h5"
    # preType = "pre_NST_BASE_student_ISRUCIII_128_*.h5"
    # seqtype = "seq_NST_BASE_student_ISRUCIII_128_*.h5"
    preType = "pre_DGKD_BASE_student_ISRUCIII_128_*.h5"
    seqtype = "seq_DGKD_BASE_student_ISRUCIII_128_*.h5"
    # preType = "pre_TA_ISRUCIII_256_*.h5"
    # seqtype = "seq_TA_ISRUCIII_256_*.h5"
    # preType = "pre_teacher_ISRUCIII_4.h5"
    # seqtype = "seq_teacher_ISRUCIII_4.h5"
    # preType = "pre_T_student_ISRUCIII_128_*.h5"
    # seqtype = "seq_T_student_ISRUCIII_128_*.h5"
    # preType = "pre_student_ISRUCIII_128_*.h5"
    # seqtype = "seq_student_ISRUCIII_128_*.h5"
    # preType = "pre_student_sleepEDF_128_2.h5"
    # seqtype = "seq_student_sleepEDF_128_2.h5"
    # preType = "pre_epochLoss_student_sleepEDF_128_*.h5"
    # seqtype = "seq_epochLoss_student_sleepEDF_128_*.h5"
    # preType = "pre_seqLoss_student_sleepEDF_128_*.h5"
    # seqtype = "seq_seqLoss_student_sleepEDF_128_*.h5"
    # preType = "pre_logitLoss_student_sleepEDF_128_*.h5" #TAKD
    # seqtype = "seq_logitLoss_student_sleepEDF_128_*.h5" #TAKD
    # preType = "pre_NOSOFT_TA_sleepEDF_256_*.h5"
    # seqtype = "seq_NOSOFT_TA_sleepEDF_256_*.h5"
    # preType = "pre_NOSOFT_student_sleepEDF_128_*.h5"
    # seqtype = "seq_NOSOFT_student_sleepEDF_128_*.h5"
    # preType = "pre_NOHARD_student_sleepEDF_128_*.h5"
    # seqtype = "seq_NOHARD_student_sleepEDF_128_*.h5"
    # preType = "pre_NOSEQ_student_sleepEDF_128_*.h5"
    # seqtype = "seq_NOSEQ_student_sleepEDF_128_*.h5"
    # preType = "pre_NOEPOCH_student_sleepEDF_128_*.h5"
    # seqtype = "seq_NOEPOCH_student_sleepEDF_128_*.h5"
    # preType = "pre_seqLoss_TA_sleepEDF_256_*.h5"
    # seqtype = "seq_seqLoss_TA_sleepEDF_256_*.h5"
    # preType = "pre_TA_sleepEDF_256_*.h5"
    # seqtype = "seq_TA_sleepEDF_256_*.h5"
    # preType = "pre_teacher_sleepEDF_18.h5"
    # seqtype = "seq_teacher_sleepEDF18.h5"
    # preType = "pre_T_student_sleepEDF_128_*.h5"
    # seqtype = "seq_T_student_sleepEDF_128_*.h5"
    # test = "weights/Baseline/models/pre_DGKD_BASE_student_ISRUCIII_128_0.h5"
    # preType = "pre_DKD_BASE_student_sleepEDF_128_*.h5"
    # seqtype = "seq_DKD_BASE_student_sleepEDF_128_*.h5"
    # preType = "pre_FITNET_BASE_student_ISRUCIII_128_*.h5"
    # seqtype = "seq_FITNET_BASE_student_ISRUCIII_128_*.h5"
    # d = Loader(
    #     '/media/liughost/Seagate/project/DeepSleepNet_Implementation_in_Keras-master (raw)/20_fold_data/deep/deep_eeg_fp_cz_20')
    # # d = Loader("/media/liughost/Seagate/project/DeepSleepNet_Implementation_in_Keras-master (raw)/20_fold_data/deep/deep_F3-A2_20")
    d = Loader("/root/autodl-nas/deep_F3-A2_20")
    # d = Loader('/root/autodl-nas/deep_eeg_fp_cz_20')
    # n = 18
    n = 4
    print(path.join(workpath, preType))
    print(path.join(workpath, seqtype))
    preList = glob(path.join(workpath + "/models", preType))
    seqList = glob(path.join(workpath + "/models", seqtype))
    print(preList)
    print(seqtype)
    seqACC, preACC = [], []
    seqF1, preF1 = [], []
    preSpe, seqSpe = [], []
    preSen, seqSen = [], []

    classes = ["W", "N1", "N2", "N3", "REM"]

    for i, f in enumerate(preList):
        d.load_pretrain(n)
        m = load_model(f)
        m.compile()
        # m.summary()
        # res = m.evaluate(d.X_test, d.y_test)
        y_pred = m.predict(d.X_test)
        preACC.append(get_acc(y_pred, d.y_test))
        y_pred_argmax = np.rint(y_pred)
        preF1.append(f1_score(d.y_test, y_pred_argmax, average="macro"))
        print(classification_report(d.y_test, y_pred_argmax))
        # seqR.append(res[1])
        # print(res)

    for i, f in enumerate(seqList):
        name = f.split("/")[-1][:-3]
        print(f.split("/")[-1][:-3])
        print(workpath)
        d.load_finetune(n)
        m = load_model(f)
        m.compile()
        y_pred = m.predict(d.X_seq_test)
        seqACC.append(get_seq_acc(y_pred, d.y_seq_test))

        y_pred = y_pred.reshape((-1, 5))
        y_labels = d.y_seq_test.reshape((-1, 5))

        y_pred_argmax = y_pred.argmax(axis=1)
        y_labels_argmax = y_labels.argmax(axis=1)

        cm = confusion_matrix(y_labels_argmax, y_pred_argmax)
        cm = np.array(cm)
        # print(cm.astype("float32"))
        # print(np.sum(cm))

        sense = sensitivity_score(y_labels_argmax, y_pred_argmax, average="macro")
        specific = specificity_score(y_labels_argmax, y_pred_argmax, average="macro")

        seqSen.append(sense)
        seqSpe.append(specific)

        print()
        # y_pred_argmax = np.rint(y_pred)
        # print(y_pred_argmax.shape)
        seqF1.append(f1_score(y_labels_argmax, y_pred_argmax, average="macro"))
        print(classification_report(y_labels_argmax, y_pred_argmax))

        # tittle_dict = {
        #     "hardLoss": "Loss1",
        #     "logitLoss": "Loss2",
        #     "epochLoss": "Loss3",
        #     "student": "Loss4",
        #     "T": "varient a"
        # }
        #
        # cm_tittle = tittle_dict[seqtype.split("_")[1]]
        # print(cm_tittle)
        # cm_tittle = "Fitnets"
        # cm_tittle = "varient b"
        # cm_tittle = "TAKD"
        # cm_tittle = "TEST"

        # plot_confusion_matrix(cm, classes=classes, title="  ", path=workpath + "/CM_Images/", fname=name, normalize=False)
        # plot_confusion_matrix(cm, classes=classes, title=cm_tittle, path=workpath + "/CM_Images/",
        #                       fname=name + "_TA_Abalation")
        # plot_confusion_matrix(cm, classes=classes, title=cm_tittle, path=workpath + "/CM_Images/",
        #                       fname=name + "_TAKD_BASE")
        # print(res)

    print("PRE_RESULTS")
    print(f"MAX_PRE: {max(preACC)}")
    print(f"AVG_PRE: {sum(preACC) / len(preACC)}")
    print("SEQ_RESULTS")
    print(f"MAX_SEQ: {max(seqACC)}")
    print(f"AVG_SEQ: {sum(seqACC) / len(seqACC)}")

    print(f"PRE_ACC: {preACC}")
    print(f"PRE_F1: {preF1}")
    print(f"SEQ_ACC: {seqACC}")
    print(f"SEQ_F1: {seqF1}")

    print(f"Spec: {seqSpe}")
    print(f"Sense: {seqSen}")
