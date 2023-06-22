import tf_GPU
import os
from os.path import join
from sklearn import preprocessing
from lh_deepsleepnet import *
from tensorflow.keras.callbacks import EarlyStopping
from data_loader import Loader
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np



def get_acc(y_pred, y_test):
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


def get_student_acc(y_seq_pred, y_seq_test):
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


if __name__ == '__main__':
    path = ('./weights', './history')
    for p in path:
        if os.path.exists(p) is False:
            os.mkdir(p)

    VERBOSE = 1
    early_stopping = EarlyStopping(patience=50)

    PRE_EPOCH = 100
    PRE_BATCH = 200

    # SEQ_EPOCH = 100
    SEQ_EPOCH = 50
    SEQ_BATCH = 20
    ch = 25
    d = Loader()

    acc_records = []


    # seq_model.summary()


    # pre_model.summary()

    for j in range(0, 5, 1):
        i = 2
        d.load_finetune(i)
        seq_model = keras.models.load_model(join(path[0], 'seq_model_ISRUC_' + str(i) + '.h5'))
        get_seq_acc(seq_model.predict(d.X_seq_test), d.y_seq_test)
        pre_model = keras.models.load_model(join(path[0], 'pre_model_ISRUC_' + str(i) + '.h5'))
        print(j, '-th fold out of 20-fold cross validation')
        d.load_pretrain(i)
        student_model = student_net(ch)

        # middle_epoch_teacher
        input_seq = Input(shape=(None, 3000, 1))
        middle_layer = Model(inputs=pre_model.input, outputs=pre_model.get_layer(f'dropout_{4*i+2}').output)
        middle_layer1 = TimeDistributed(middle_layer)(input_seq)
        print(middle_layer1.shape)
        middle_layer1 = tf.expand_dims(middle_layer1, axis=-1)
        print(middle_layer1.shape)
        # output_shape =（input_shape-pool_size + 1）/strides
        middle_layer1 = TimeDistributed(MaxPool1D(pool_size=57, strides=11, ))(middle_layer1)
        print(middle_layer1.shape)
        output = TimeDistributed(Flatten())(middle_layer1)
        middle_layer = Model(input_seq, output)
        # middle_layer.summary()
        middle_epoch_teacher = middle_layer.predict(d.X_seq_train)
        # print(middle_epoch_teacher)
        # print(middle_epoch_teacher.shape)

        # middle_seq_teacher
        middle_seq_layer = Model(inputs=seq_model.input, outputs=seq_model.get_layer(f'bidirectional_{i}').output)
        middle_layer1 = (middle_seq_layer)(input_seq)
        middle_layer1 = tf.expand_dims(middle_layer1, axis=-1)
        # output_shape =（input_shape-pool_size + 1）/strides
        print(middle_layer1.shape)
        middle_layer1 = TimeDistributed(MaxPool1D(pool_size=321, strides=64, ))(middle_layer1)
        # middle_layer1 = MaxPool1D(pool_size=321, strides=64, )(middle_layer1)
        print(middle_layer1.shape)
        output = TimeDistributed(Flatten())(middle_layer1)
        middle_seq_layer = Model(input_seq, output)
        # middle_seq_layer.summary()
        middle_seq_teacher = middle_seq_layer.predict(d.X_seq_train)

        soft_label = seq_model.predict(d.X_seq_train)
        true_label = d.y_seq_train

        train_data = [d.X_seq_train, true_label, soft_label, middle_epoch_teacher, middle_seq_teacher]
        valid_data = [d.X_seq_valid, d.y_seq_valid, seq_model.predict(d.X_seq_valid), middle_layer.predict(d.X_seq_valid),
                      middle_seq_layer.predict(d.X_seq_valid)]
        
        seq_history = student_model.fit(
            train_data,
            batch_size=SEQ_BATCH,
            epochs=SEQ_EPOCH,
            verbose=VERBOSE,
            validation_data=valid_data,
            callbacks=[early_stopping]
        )

        middle_epoch_teacher = middle_layer.predict(d.X_seq_test)
        # print(middle_epoch_teacher.shape)
        middle_seq_teacher = middle_seq_layer.predict(d.X_seq_test)

        soft_label = seq_model.predict(d.X_seq_test)
        true_label = d.y_seq_test

        test_data = [d.X_seq_test, true_label, soft_label, middle_epoch_teacher, middle_seq_teacher]
        acc = get_student_acc(student_model.predict(test_data), d.y_seq_test)
        acc_records.append(acc)

        # student_model.save(join(path[0], 'seq_model_' + str(i) + '.weights'))
        # with open(join(path[1], 'seq_history_' + str(i) + '.bin'), 'wb') as f:
        #     pickle.dump(seq_history.history, f)
    print(acc_records)
    print("AVG: "+str(sum(acc_records)/len(acc_records)))
