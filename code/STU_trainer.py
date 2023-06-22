import tf_GPU
import os
import tensorflow as tf
from os.path import join
from STU_sleepnet import STU_featurenet, STU_sleepnet, MySeqLoss, MyPreLoss
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
    path = ('./weights', './history')
    for p in path:
        if os.path.exists(p) is False:
            os.mkdir(p)

    VERBOSE = 0
    early_stopping = EarlyStopping(patience=50)

    PRE_EPOCH = 100
    PRE_BATCH = 100

    SEQ_EPOCH = 200
    SEQ_BATCH = 20

    seq_TBoard = keras.callbacks.TensorBoard(
        log_dir='TA_log_dir/seq_log',
        histogram_freq=1,
        embeddings_freq=1, )

    pre_TBoard = keras.callbacks.TensorBoard(
        log_dir='TA_log_dir/pre_log',
        histogram_freq=1,
        embeddings_freq=1, )

    acc_record = []

    my_seq_objects = {'MySeqLoss': MySeqLoss}
    my_pre_objects = {"MyPreLoss": MyPreLoss}

    for j in range(0, 5, 1):
        print(j, '-th fold out of 20-fold cross validation')
        i = 4
        n = 1
        d = Loader()
        d.load_pretrain(i)
        pre_teacher = keras.models.load_model(join(path[0], f'pre_TA_ISRUCIII_256_{n}.h5'), custom_objects=my_pre_objects)
        seq_teacher = keras.models.load_model(join(path[0], f'seq_TA_ISRUCIII_256_{n}.h5'), custom_objects=my_seq_objects)
        # seq_teacher.summary()
        # pre_teacher.summary()
        # seq_teacher = Model(inputs=seq_teacher.get_layer("Input_Seq_Signal").input, outputs=seq_teacher.get_layer('seq_softmax').output)
        # pre_teacher = Model(inputs=pre_teacher.get_layer('input_signal').input, outputs=pre_teacher.get_layer('pre_softmax').output)
        seq_teacher.summary()
        pre_teacher.summary()

        TA_lstms = 256
        n_filters = 128
        n_lstms = 128
        n_maxpooling = 2 * TA_lstms - 2 * n_lstms + 1
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
        # epoch_label_output = pre_teacher.get_layer(f'dropout_{4*n+2}').output
        epoch_label_generator = Model(epoch_label_input, epoch_label_output)

        # create pre_train data
        pre_train_data = [d.X_train, d.y_train, pre_teacher.predict(d.X_train),
                          epoch_label_generator.predict(d.X_train)]
        pre_valid_data = [d.X_valid, d.y_valid, pre_teacher.predict(d.X_valid),
                          epoch_label_generator.predict(d.X_valid)]
        pre_test_data = [d.X_test, d.y_test, pre_teacher.predict(d.X_test), epoch_label_generator.predict(d.X_test)]

        # pre training
        pre_model = STU_featurenet(n_filters)
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
            callbacks=[early_stopping, pre_TBoard]
        )

        get_student_pre_acc(pre_model.predict(pre_test_data), d.y_test)

        # pre_model.save(join(path[0], 'pre_model_' + str(i) + '.weights'))
        pre_model_saved = Model(pre_model.get_layer('input_signal').input, pre_model.get_layer("pre_softmax").output)
        # pre_model_saved.save(join(path[0], 'pre_student_sleepEDF_' + str(n_lstms) + '_' + str(j) + '.h5'))
        pre_model_saved.save(join(path[0], 'pre_student_ISRUCIII_' + str(n_lstms) + '_' + str(j) + '.h5'))

        # with open(join(path[1], 'pre_history_' + str(i) + '.bin'), 'wb') as f:
        #     pickle.dump(pre_history.history, f)

        # del pre_history

        # fine tuning

        # getting data
        d.load_finetune(i)

        # building seq model
        seq_model = STU_sleepnet(pre_model, n_LSTM=n_lstms)

        seq_model.summary()

        # create seq_train data
        seq_train_data = [d.X_seq_train, d.y_seq_train, seq_teacher.predict(d.X_seq_train),
                          seq_label_generator.predict(d.X_seq_train)]
        seq_valid_data = [d.X_seq_valid, d.y_seq_valid, seq_teacher.predict(d.X_seq_valid),
                          seq_label_generator.predict(d.X_seq_valid)]
        seq_test_data = [d.X_seq_test, d.y_seq_test, seq_teacher.predict(d.X_seq_test),
                         seq_label_generator.predict(d.X_seq_test)]

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
            callbacks=[seq_TBoard]
        )

        print()
        print("-----------Test for teacher-----------")
        get_seq_acc(seq_test_data[2], seq_test_data[1])

        print("-------------Test for STU--------------")
        tmp_Seq_acc = get_student_seq_acc(seq_model.predict(seq_test_data), d.y_seq_test)
        print("--------------------------------------")

        acc_record.append(tmp_Seq_acc)

        print()

        # get_seq_acc(seq_model.predict(d.X_seq_test), d.y_seq_test)

        # seq_model.save(join(path[0], 'seq_model_' + str(i) + '.weights'))
        seq_model_saved = Model(seq_model.get_layer("Input_Seq_Signal").input,
                                seq_model.get_layer("seq_softmax").output)
        # seq_model_saved.save(join(path[0], f'seq_student_sleepEDF_' + str(n_lstms) + '_' + str(j) + '.h5'))
        seq_model_saved.save(join(path[0], f'seq_student_ISRUCIII_' + str(n_lstms) + '_' + str(j) + '.h5'))
        # with open(join(path[1], 'seq_history_' + str(i) + '.bin'), 'wb') as f:
        #     pickle.dump(seq_history.history, f)
        #
        # del pre_model, seq_model,

        print()
    avg_acc = sum(acc_record) / len(acc_record)
    print(acc_record)
    print("AVG_ACC:" + str(avg_acc))
