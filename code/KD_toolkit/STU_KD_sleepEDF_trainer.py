import tf_GPU
import os
import tensorflow as tf
from os.path import join
from STU_KD_sleepEDF_sleepnet import STU_featurenet, STU_sleepnet
from tensorflow.keras.callbacks import EarlyStopping
from data_loader import Loader
import pickle
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import MaxPool2D, TimeDistributed, Reshape
from tensorflow import keras


# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def get_pre_acc(y_pred, y_test):
    cnt = 0
    for i in range(len(y_pred)):
        if np.argmax(y_pred[i]) == np.argmax(y_test[i]):
            cnt += 1
    acc = float(cnt) / len(y_pred)
    print("Pre-train ACC:", acc)
    return "ACC: " + str(acc)


def get_seq_acc(y_seq_pred, y_seq_test):
    cnt = 0
    cnt2 = 0
    for i in range(len(y_seq_pred)):
        for j in range(len(y_seq_pred[i])):
            cnt2 += 1
            if np.argmax(y_seq_pred[i][j]) == np.argmax(y_seq_test[i][j]):
                cnt += 1
    acc = float(cnt) / cnt2
    print("Finetune ACC:", acc)
    return "ACC: " + str(acc)


def get_student_seq_acc(y_seq_pred, y_seq_test):
    y_seq_pred = y_seq_pred[0]
    # print(y_seq_pred.shape, y_seq_test.shape)
    cnt = 0
    cnt2 = 0
    for i in range(len(y_seq_pred)):
        for j in range(len(y_seq_pred[i])):
            cnt2 += 1
            if np.argmax(y_seq_pred[i][j]) == np.argmax(y_seq_test[i][j]):
                cnt += 1
    acc = float(cnt) / cnt2
    print("Finetune ACC:", acc)
    # return "ACC: " + str(acc)
    return acc


def get_student_pre_acc(y_pred, y_test):
    y_pred = y_pred[0]
    cnt = 0
    for i in range(len(y_pred)):
        if np.argmax(y_pred[i]) == np.argmax(y_test[i]):
            cnt += 1
    acc = float(cnt) / len(y_pred)
    print("Pre-train ACC:", acc)
    # return "ACC: " + str(acc)
    return acc


if __name__ == '__main__':
    path = ('../weights', '../history')
    for p in path:
        if os.path.exists(p) is False:
            os.mkdir(p)

    VERBOSE = 1
    early_stopping = EarlyStopping(patience=50)

    PRE_EPOCH = 100
    PRE_BATCH = 100

    SEQ_EPOCH = 100
    SEQ_BATCH = 20

    seq_TBoard = keras.callbacks.TensorBoard(
        log_dir='TA_log_dir/seq_log',
        histogram_freq=1,
        embeddings_freq=1, )

    pre_TBoard = keras.callbacks.TensorBoard(
        log_dir='TA_log_dir/pre_log',
        histogram_freq=1,
        embeddings_freq=1, )

    pre_acc_record = []
    seq_acc_record = []

    for j in range(0, 5, 1):
        print(j, '-th fold out of 20-fold cross validation')
        i = 18
        n = 18
        d = Loader('/root/autodl-tmp/deep_eeg_fp_cz_20')
        d.load_pretrain(i)
        pre_teacher = keras.models.load_model(join(path[0], f'pre_teacher_sleepEDF_18.h5'))
        seq_teacher = keras.models.load_model(join(path[0], f'seq_teacher_sleepEDF18.h5'))
        seq_teacher.summary()
        pre_teacher.summary()

        T_lstms = 256
        TA_lstms = 512
        stu_filters = 128
        stu_lstms = 128
        T_maxpooling = 2 * T_lstms - 2 * stu_lstms + 1
        TA_maxpooling = 2 * TA_lstms - 2 * stu_lstms + 1
        # print(T_maxpooling, TA_maxpooling)

        # create pre_train data
        pre_train_data = [d.X_train, d.y_train, pre_teacher.predict(d.X_train)]
        pre_valid_data = [d.X_valid, d.y_valid, pre_teacher.predict(d.X_valid)]
        pre_test_data = [d.X_test, d.y_test, pre_teacher.predict(d.X_test)]

        # pre training
        pre_model = STU_featurenet(stu_filters)
        pre_model.summary()
        print([data.shape for data in pre_train_data])

        print("Pre_Teacher_ACC:")
        print(get_pre_acc(pre_test_data[2], pre_test_data[1]))

        pre_history = pre_model.fit(
            pre_train_data,
            batch_size=PRE_BATCH,
            epochs=PRE_EPOCH,
            verbose=VERBOSE,
            validation_data=pre_valid_data,
            callbacks=[early_stopping]
        )

        pre_acc_record.append(get_student_pre_acc(pre_model.predict(pre_test_data), d.y_test))

        # pre_model.save(join(path[0], 'pre_model_' + str(i) + '.weights'))
        pre_model_saved = Model(pre_model.get_layer('input_signal').input, pre_model.get_layer("pre_softmax").output)
        pre_model_saved.save(join(path[0], 'pre_KD_BASE_student_sleepEDF_' + str(stu_lstms) + '_' + str(j) + '.h5'))

        # fine tuning

        # getting data
        d.load_finetune(i)

        # building seq model
        seq_model = STU_sleepnet(pre_model, n_LSTM=stu_lstms)

        seq_model.summary()

        # create seq_train data
        seq_train_data = [d.X_seq_train, d.y_seq_train, seq_teacher.predict(d.X_seq_train)]
        seq_valid_data = [d.X_seq_valid, d.y_seq_valid, seq_teacher.predict(d.X_seq_valid)]
        seq_test_data = [d.X_seq_test, d.y_seq_test, seq_teacher.predict(d.X_seq_test)]

        print([data.shape for data in seq_train_data])

        print("Seq_Teacher_ACC:")
        print(get_seq_acc(seq_test_data[2], seq_test_data[1]))

        seq_history = seq_model.fit(
            seq_train_data,
            batch_size=SEQ_BATCH,
            epochs=SEQ_EPOCH,
            verbose=VERBOSE,
            validation_data=seq_valid_data,
            # callbacks=[seq_TBoard]
        )

        print()
        print("-----------Test for teacher-----------")
        get_seq_acc(seq_test_data[2], seq_test_data[1])

        print("-------------Test for STU--------------")
        tmp_Seq_acc = get_student_seq_acc(seq_model.predict(seq_test_data), d.y_seq_test)
        print("--------------------------------------")

        seq_acc_record.append(tmp_Seq_acc)

        print()
        seq_model_saved = Model(seq_model.get_layer("Input_Seq_Signal").input,
                                seq_model.get_layer("seq_softmax").output)
        seq_model_saved.save(join(path[0], f'seq_KD_BASE_student_sleepEDF_' + str(stu_lstms) + '_' + str(j) + '.h5'))

        print()
    avg_acc = sum(seq_acc_record) / len(seq_acc_record)
    print(pre_acc_record)
    print(seq_acc_record)
    print("AVG_ACC:" + str(avg_acc))
