import tf_GPU
import os
import tensorflow as tf
from os.path import join
from TA_sleepnet import TA_featurenet, TAsleepnet
from tensorflow.keras.callbacks import EarlyStopping
from microsleepnet_master.data_loader import Loader
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
    path = ('../weights/SleepEDF_model/models', '../history')
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
        embeddings_freq=1,)

    pre_TBoard = keras.callbacks.TensorBoard(
        log_dir='TA_log_dir/pre_log',
        histogram_freq=1,
        embeddings_freq=1,)

    seq_acc_record = []
    pre_acc_record = []

    for j in range(0, 5, 1):
        print(j, '-th fold out of 20-fold cross validation')
        i = 18
        d = Loader('/root/autodl-tmp/deep_eeg_fp_cz_20')
        d.load_pretrain(i)
        # seq_teacher = load_model(join(path[0], 'seq_teacher_ISRUCIII_' + str(i) + '.h5'))
        # pre_teacher = load_model(join(path[0], 'pre_teacher_ISRUCIII_' + str(i) + '.h5'))
        seq_teacher = load_model(join(path[0], 'seq_teacher_sleepEDF' + str(i) + '.h5'))
        pre_teacher = load_model(join(path[0], 'pre_teacher_sleepEDF_' + str(i) + '.h5'))
        pre_teacher.summary()
        seq_teacher.summary()

        pre_teacher.evaluate(d.X_test, d.y_test)
        get_pre_acc(pre_teacher.predict(d.X_test), d.y_test)

        n_filters = 128
        n_lstms = 256
        n_maxpooling = 1024 - 2*n_lstms +1
        print(n_maxpooling)

        # building seq_label_generator
        # output_shape =（input_shape-pool_size + 1）/strides
        seq_label_input = seq_teacher.input
        seq_label_output = seq_teacher.get_layer(f'sequence_layer').output
        print("pooling shape:")
        print(seq_label_output.shape)
        seq_label_output = tf.expand_dims(seq_label_output, axis=-1)
        print(seq_label_output.shape)
        seq_label_output = MaxPool2D(pool_size=(1, n_maxpooling), strides=(1, 1))(seq_label_output)
        print(seq_label_output.shape)
        seq_label_generator = Model(seq_label_input, seq_label_output)

        # building epoch_label_generator
        epoch_label_input = pre_teacher.input
        epoch_label_output = pre_teacher.get_layer(f'epoch_layer').output
        epoch_label_generator = Model(epoch_label_input, epoch_label_output)

        # create pre_train data
        pre_train_data = [d.X_train, d.y_train, pre_teacher.predict(d.X_train), epoch_label_generator.predict(d.X_train)]
        pre_valid_data = [d.X_valid, d.y_valid, pre_teacher.predict(d.X_valid), epoch_label_generator.predict(d.X_valid)]
        pre_test_data = [d.X_test, d.y_test, pre_teacher.predict(d.X_test), epoch_label_generator.predict(d.X_test)]

        # pre training
        pre_model = TA_featurenet(n_filters)
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
            # callbacks=[early_stopping, pre_TBoard]
        )


        pre_acc_record.append(get_student_pre_acc(pre_model.predict(pre_test_data), d.y_test))

        pre_model_saved = Model(pre_model.get_layer('input_signal').input, pre_model.get_layer("pre_softmax").output)

        # pre_model.save(join(path[0], 'pre_model_' + str(i) + '.weights'))
        # pre_model_saved.save(join(path[0], 'pre_TA_sleepEDF_' + str(n_lstms) + '_' + str(j) + '.h5'))
        # pre_model_saved.save(join(path[0], 'pre_epochLoss_TA_sleepEDF_' + str(n_lstms) + '_' + str(j) + '.h5'))
        # pre_model_saved.save(join(path[0], 'pre_seqLoss_TA_sleepEDF_' + str(n_lstms) + '_' + str(j) + '.h5'))
        # pre_model_saved.save(join(path[0], 'pre_logitLoss_TA_sleepEDF_' + str(n_lstms) + '_' + str(j) + '.h5'))
        # pre_model_saved.save(join(path[0], 'pre_NOSOFT_TA_sleepEDF_' + str(n_lstms) + '_' + str(j) + '.h5'))
        pre_model_saved.save(join(path[0], 'pre_NOHARD_TA_sleepEDF_' + str(n_lstms) + '_' + str(j) + '.h5'))

        # with open(join(path[1], 'pre_history_' + str(i) + '.bin'), 'wb') as f:
        #     pickle.dump(pre_history.history, f)

        # del pre_history
        # del d.X_test, d.X_train, d.X_valid

        # fine tuning

        # getting data
        d.load_finetune(i)

        # building seq model
        seq_model = TAsleepnet(pre_model, n_LSTM=n_lstms)

        # seq_model.summary()
        seq_teacher.evaluate(d.X_seq_test, d.y_seq_test)
        get_seq_acc(seq_teacher.predict(d.X_seq_test), d.y_seq_test)

        # create seq_train data
        seq_train_data = [d.X_seq_train, d.y_seq_train, seq_teacher.predict(d.X_seq_train), seq_label_generator.predict(d.X_seq_train)]
        seq_valid_data = [d.X_seq_valid, d.y_seq_valid, seq_teacher.predict(d.X_seq_valid), seq_label_generator.predict(d.X_seq_valid)]
        seq_test_data = [d.X_seq_test, d.y_seq_test, seq_teacher.predict(d.X_seq_test), seq_label_generator.predict(d.X_seq_test)]

        print([data.shape for data in seq_train_data])

        print("Seq_Teacher_ACC:")
        print(get_seq_acc(seq_test_data[2], seq_test_data[1]))

        seq_train_data[-1] = np.reshape(seq_train_data[-1], seq_train_data[-1].shape[:-1])
        seq_test_data[-1] = np.reshape(seq_test_data[-1], seq_test_data[-1].shape[:-1])
        seq_valid_data[-1] = np.reshape(seq_valid_data[-1], seq_valid_data[-1].shape[:-1])

        print([data.shape for data in seq_train_data])

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


        print("-----------Test for TA-----------")
        tmp_Seq_acc = get_student_seq_acc(seq_model.predict(seq_test_data), d.y_seq_test)
        print("--------------------------------------")

        seq_acc_record.append(tmp_Seq_acc)

        print()


        # get_seq_acc(seq_model.predict(d.X_seq_test), d.y_seq_test)

        # seq_model.save(join(path[0], 'seq_model_' + str(i) + '.weights'))
        seq_model_saved = Model(seq_model.get_layer("Input_Seq_Signal").input, seq_model.get_layer("seq_softmax").output)
        # seq_model_saved.save(join(path[0], f'seq_TA_sleepEDF_' + str(n_lstms) + '_' + str(j) + '.h5'))
        # seq_model_saved.save(join(path[0], f'seq_epochLoss_TA_sleepEDF_' + str(n_lstms) + '_' + str(j) + '.h5'))
        # seq_model_saved.save(join(path[0], f'seq_seqLoss_TA_sleepEDF_' + str(n_lstms) + '_' + str(j) + '.h5'))
        # seq_model_saved.save(join(path[0], f'seq_logitLoss_TA_sleepEDF_' + str(n_lstms) + '_' + str(j) + '.h5'))
        seq_model_saved.save(join(path[0], f'seq_NOHARD_TA_sleepEDF_' + str(n_lstms) + '_' + str(j) + '.h5'))
        # seq_model_saved.save(join(path[0], f'seq_NOSOFT_TA_sleepEDF_' + str(n_lstms) + '_' + str(j) + '.h5'))

        # with open(join(path[1], 'seq_history_' + str(i) + '.bin'), 'wb') as f:
        #     pickle.dump(seq_history.history, f)
        #
        # del pre_model, seq_model,

        print()
    avg_acc = sum(seq_acc_record) / len(seq_acc_record)
    print(pre_acc_record)
    print(seq_acc_record)
