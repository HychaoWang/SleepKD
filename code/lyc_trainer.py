import os
from os.path import join
import tf_GPU
from sklearn import preprocessing
from lyc_deepsleepnet import *
from TA_sleepnet import MyPreLoss, MySeqLoss
from tensorflow.keras.callbacks import EarlyStopping
from data_loader import Loader
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


os.environ['CUDA_VISIBLE_DEVICES'] = '1'


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
    return acc
    # return "ACC: " + str(acc)


if __name__ == '__main__':
    path = ('./weights', './history')
    for p in path:
        if os.path.exists(p) is False:
            os.mkdir(p)

    VERBOSE = 1
    early_stopping = EarlyStopping(patience=150)

    PRE_EPOCH = 100
    PRE_BATCH = 200

    # SEQ_EPOCH = 100
    SEQ_EPOCH = 75
    SEQ_BATCH = 20
    ch = 25
    d = Loader()


    # seq_model.summary()

    my_seq_objects = {'MySeqLoss': MySeqLoss}
    my_pre_objects = {"MyPreLoss":MyPreLoss}

    acc_records = []

    # pre_model.summary()

    for i in range(0, 5, 1):
        d.load_finetune(2)

        pre_model = keras.models.load_model(join(path[0], 'TA/TA_pre_ISRUC_III_256_3.h5'), custom_objects=my_pre_objects)
        # get_seq_acc(seq_model.predict(d.X_seq_test), d.y_seq_test)
        seq_model = keras.models.load_model(join(path[0], 'TA/TA_seq_ISRUC_III_256_3.h5'), custom_objects=my_seq_objects)
        # print(i, '-th fold out of 20-fold cross validation')

        seq_model.summary()
        pre_model.summary()

        d.load_pretrain(2)
        student_model = student_net(ch)

        # middle_epoch_teacher
        input_seq = Input(shape=(None, 3000, 1))
        middle_layer = Model(inputs=pre_model.get_layer("input_signal").input, outputs=pre_model.get_layer('epoch_layer').output)
        middle_layer1 = TimeDistributed(middle_layer)(input_seq)
        middle_layer1 = tf.expand_dims(middle_layer1, axis=-1)
        
        # output_shape =（input_shape-pool_size + 1）/strides
        middle_layer1 = TimeDistributed(MaxPool1D(pool_size=57, strides=11, ))(middle_layer1)
        output = TimeDistributed(Flatten())(middle_layer1)
        middle_layer = Model(input_seq, output)
        
        # middle_layer.summary()
        middle_epoch_teacher = middle_layer.predict(d.X_seq_train)

        # middle_seq_teacher
        middle_seq_layer = Model(inputs=seq_model.get_layer("Input_Seq_Signal").input, outputs=seq_model.get_layer('bidirectional_3').output)
        middle_layer1 = (middle_seq_layer)(input_seq)
        middle_layer1 = tf.expand_dims(middle_layer1, axis=-1)
        # output_shape =（input_shape-pool_size + 1）/strides
        # print(middle_layer1.shape)
        # pool_size=161, strides=32, 
        # pool_size=65, strides=64, 
        middle_layer1 = TimeDistributed(MaxPool1D(pool_size=161, strides=32, ))(middle_layer1)
        # middle_layer1 = MaxPool1D(pool_size=321, strides=64, )(middle_layer1)
        # print(middle_layer1.shape)
        output = TimeDistributed(Flatten())(middle_layer1)
        middle_seq_layer = Model(input_seq, output)
        # middle_seq_layer.summary()
        middle_seq_teacher = middle_seq_layer.predict(d.X_seq_train)

        seq_teacher = Model(inputs=seq_model.get_layer("Input_Seq_Signal").input, outputs=seq_model.get_layer('seq_softmax').output)
        pre_teacher = Model(inputs=pre_model.get_layer('input_signal').input, outputs=pre_model.get_layer('pre_softmax').output)


        soft_label = seq_teacher.predict(d.X_seq_train)

        true_label = d.y_seq_train

        train_data = [d.X_seq_train, true_label, soft_label, middle_epoch_teacher, middle_seq_teacher]
        valid_data = [d.X_seq_valid, d.y_seq_valid, seq_teacher.predict(d.X_seq_valid), middle_layer.predict(d.X_seq_valid),
                      middle_seq_layer.predict(d.X_seq_valid)]

        print(train_data[0].shape)
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

        soft_label = seq_teacher.predict(d.X_seq_test)
        true_label = d.y_seq_test

        print("-----------Test for teacher-----------")

        get_seq_acc(soft_label, true_label)

        print("--------------------------------------")

        test_data = [d.X_seq_test, true_label, soft_label, middle_epoch_teacher, middle_seq_teacher]

        print("-----------Test for student-----------")

        cur_acc = get_student_acc(student_model.predict(test_data), d.y_seq_test)

        print("--------------------------------------")
        acc_records.append(cur_acc)
        
        saving_model = Model(inputs=student_model.get_layer("Input_Seq_Signal").input, outputs=student_model.get_layer('softmax').output)
        saving_model.compile()
        saving_model.save(join(path[0], 'stu_model_TA_all_ISRUC_III_' + str(i) + '.h5'))
        # with open(join(path[1], 'seq_history_' + str(i) + '.bin'), 'wb') as f:
        #     pickle.dump(seq_history.history, f)
    print(acc_records)
    print("ACG: "+str(sum(acc_records)/len(acc_records)))
    print("MAX: "+str(max(acc_records)))