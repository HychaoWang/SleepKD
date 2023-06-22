import tf_GPU
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Dropout, MaxPool1D, Activation, Conv2D, MaxPool2D, Flatten, \
    Permute, LayerNormalization
from tensorflow.keras.layers import Reshape, LSTM, TimeDistributed, Bidirectional, BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras import layers as KL
from tensorflow.keras import models as KM
from tensorflow.keras.metrics import categorical_accuracy, KLDivergence
from sklearn import preprocessing
from copy import deepcopy
import numpy as np


def TA_featurenet(n_filters=128, context=20):
    activation = tf.nn.relu
    # activation = tf.nn.leaky_relu
    padding = 'same'

    ######### Input ########
    input_signal = Input(shape=(30 * 100, 1), name='input_signal')
    true_label = Input(shape=(5), name="true_label")
    soft_label = Input(shape=(5), name="soft_label")
    epoch_label = Input(shape=(1024), name="epoch_label")
    # print("input_signal:",input_signal.shape)

    ######### CNNs with small filter size at the first layer #########
    # print("\nCNN1")
    cnn0 = Conv1D(
        kernel_size=50,
        filters=64,
        strides=6, kernel_regularizer=keras.regularizers.l2(0.001))
    s = cnn0(input_signal)
    s = BatchNormalization()(s)
    s = Activation(activation=activation)(s)
    # print("cnn0:",s.shape)
    cnn1 = MaxPool1D(pool_size=8, strides=8)
    s = cnn1(s)
    # print("cnn1:",s.shape)
    cnn2 = Dropout(0.5)
    s = cnn2(s)
    # print("cnn2:",s.shape)
    # cnn3 = Conv1D(kernel_size=8,filters=n_filters,strides=1,padding=padding)
    # s = cnn3(s)
    # s = BatchNormalization()(s)
    # s = Activation(activation=activation)(s)
    # print("cnn3:",s.shape)
    # cnn4 = Conv1D(kernel_size=8,filters=n_filters,strides=1,padding=padding)
    # s = cnn4(s)
    # s = BatchNormalization()(s)
    # s = Activation(activation=activation)(s)
    # print("cnn4:",s.shape)
    cnn5 = Conv1D(kernel_size=8, filters=n_filters, strides=1, padding=padding)
    s = cnn5(s)
    s = BatchNormalization()(s)
    s = Activation(activation=activation)(s)
    # print("cnn5:",s.shape)
    cnn6 = MaxPool1D(pool_size=4, strides=4)
    s = cnn6(s)
    # print("cnn6:",s.shape)
    cnn7 = Reshape((int(s.shape[1]) * int(s.shape[2]),))  # Flatten
    s = cnn7(s)
    # print("cnn7:",s.shape)

    ######### CNNs with large filter size at the first layer #########
    # print('\nCNN2')
    cnn8 = Conv1D(
        kernel_size=400,
        filters=64, strides=50, kernel_regularizer=keras.regularizers.l2(0.001))
    l = cnn8(input_signal)
    l = BatchNormalization()(l)
    l = Activation(activation=activation)(l)
    # print("cnn8:",l.shape)
    cnn9 = MaxPool1D(pool_size=4, strides=4)
    l = cnn9(l)
    # print("cnn9:",l.shape)
    cnn10 = Dropout(0.5)
    l = cnn10(l)
    # print("cnn10:",l.shape)
    # cnn11 = Conv1D(kernel_size=6,filters=n_filters,strides=1,padding=padding)
    # l = cnn11(l)
    # l = BatchNormalization()(l)
    # l = Activation(activation=activation)(l)
    # print("cnn11:",l.shape)
    # cnn12 = Conv1D(kernel_size=6,filters=n_filters,strides=1,padding=padding)
    # l = cnn12(l)
    # l = BatchNormalization()(l)
    # l = Activation(activation=activation)(l)
    # print("cnn12:",l.shape)
    cnn13 = Conv1D(kernel_size=6, filters=n_filters, strides=1, padding=padding)
    l = cnn13(l)
    l = BatchNormalization()(l)
    l = Activation(activation=activation)(l)
    # print("cnn13:",l.shape)
    cnn14 = MaxPool1D(pool_size=2, strides=2)
    l = cnn14(l)
    # print("cnn14:",l.shape)
    cnn15 = Reshape((int(l.shape[1]) * int(l.shape[2]),))
    l = cnn15(l)
    # print("cnn15:",l.shape)

    # print('\nMERGED')
    merged = keras.layers.concatenate([s, l])
    # print("merged:",merged.shape)
    merged = Dense(1024)(merged)
    merged = Dropout(0.5, name="epoch_layer")(merged)
    epoch_output = merged
    merged = Dense(5, name='merged')(merged)
    # print('merged',merged.shape)
    pre_softmax = Activation(activation='softmax', name="pre_softmax")(merged)
    # print('pre_softmax',pre_softmax.shape)
    # myLoss = WbceLoss()([true_label, soft_label, pre_softmax])

    myLoss = MyPreLoss()([true_label, soft_label, epoch_label, epoch_output, pre_softmax])

    # print("epochShape")
    # print(epoch_label.shape)

    # pre_model = Model(input_signal,pre_softmax)
    pre_model = Model([input_signal, true_label, soft_label, epoch_label], [pre_softmax, myLoss])
    pre_opt = keras.optimizers.Adam(lr=1e-4)
    # pre_model.compile(optimizer=pre_opt,loss='categorical_crossentropy',metrics=['acc'])
    pre_model.compile(optimizer=pre_opt, run_eagerly=True)
    # pre_model.compile(optimizer=pre_opt)

    return pre_model


def TAsleepnet(pre_model, n_LSTM=512, context=20):
    activation_seq = 'relu'

    true_label = Input([context, 5], name="true_label")
    soft_label = Input([context, 5], name="soft_label")
    # epoch_label = Input([1024], name="epoch_label")
    sequence_label = Input([context, 2 * n_LSTM], name="sequence_label")

    input_signal = pre_model.get_layer(name='input_signal').input
    merged = pre_model.get_layer(name='merged').output
    epoch_layer = pre_model.get_layer(name="epoch_layer").output

    epoch_model = Model(input_signal, epoch_layer)
    cnn_part = Model(input_signal, merged)  # pre train 된 부분

    input_seq = Input(shape=(None, 3000, 1), name="Input_Seq_Signal")  # sequence 길이 모르므로 None
    # print('input_seq', input_seq.shape)

    signal_sequence = TimeDistributed(cnn_part)(input_seq)  # TimeDistributed 로 시퀀스를 입력 받을 수 있음
    # epoch_output = TimeDistributed(epoch_model)(input_seq)

    # print('signal_sequence',signal_sequence.shape)

    bidirection = Bidirectional(LSTM(n_LSTM, dropout=0.5, activation=activation_seq, return_sequences=True),
                                merge_mode='concat', name="sequence_layer")(signal_sequence)
    # print('bidirection',bidirection.shape)

    fc1024 = Dense(2 * n_LSTM)(signal_sequence)
    fc1024 = BatchNormalization()(fc1024)
    fc1024 = Activation(activation=activation_seq)(fc1024)
    # print('fc1024',fc1024.shape)
    residual = keras.layers.add([bidirection, fc1024])  # skip-connection
    residual = Dropout(0.5)(residual)
    # print('residual',residual.shape)

    dense_seq = Dense(5)(residual)
    # print('dense_seq',dense_seq.shape)

    seq_softmax = Activation(activation='softmax', name="seq_softmax")(dense_seq)
    # print('seq_softmax',seq_softmax.shape)
    sequence_ouput = bidirection

    # print(epoch_output.shape)
    # print(sequence_ouput.shape)

    # print([t.shape for t in [true_label, soft_label, sequence_label, sequence_ouput, seq_softmax]])

    # NormalizeLayer = LayerNormalization(axis=1 , center=True , scale=True)

    # sequence_label = NormalizeLayer(sequence_label)
    # sequence_ouput = NormalizeLayer(sequence_ouput)

    myLoss = MySeqLoss()([true_label, soft_label, sequence_label, sequence_ouput, seq_softmax])

    seq_model = Model([input_seq, true_label, soft_label, sequence_label], [seq_softmax, myLoss])
    # seq_model = Model(input_seq, seq_softmax)
    seq_opt = keras.optimizers.Adam(lr=1e-5)
    seq_model.compile(optimizer=seq_opt, run_eagerly=True)
    # seq_model.compile(optimizer=seq_opt)

    return seq_model


class MyPreLoss(KL.Layer):
    def __init__(self, **kwargs):
        super(MyPreLoss, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        # inputs：Input tensor, or list/tuple of input tensors.
        如上，父类KL.Layer的call方法明确要求inputs为一个tensor，或者包含多个tensor的列表/元组
        所以这里不能直接接受多个入参，需要把多个入参封装成列表/元组的形式然后在函数中自行解包，否则会报错。
        """
        # 解包入参
        true_label, soft_label, middle_epoch_teacher, middle_epoch_student, output = inputs

        kl = tf.keras.losses.KLDivergence()

        # 复杂的损失函数
        e1 = 0.1
        e2 = 0.8
        e3 = 0.09

        # 归一化处理
        if (K.int_shape(middle_epoch_teacher)[0] != None):
            middle_epoch_teacher = K.eval(middle_epoch_teacher)
            middle_epoch_student = K.eval(middle_epoch_student)

            zscore_scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1))
            for a in range(0, len(middle_epoch_teacher)):
                middle_epoch_teacher[a] = zscore_scaler1.fit_transform(np.array([middle_epoch_teacher[a]]).T).T

            for a in range(0, len(middle_epoch_student)):
                middle_epoch_student[a] = zscore_scaler1.fit_transform(np.array([middle_epoch_student[a]]).T).T
                # print(middle_epoch_student[a].shape)

        true_loss = categorical_crossentropy(true_label, output)
        soft_loss = categorical_crossentropy(soft_label, output)

        middle_epoch_loss = kl(middle_epoch_teacher, middle_epoch_student)

        true_loss = K.mean(true_loss)
        soft_loss = K.mean(soft_loss)

        self.add_loss(e1 * true_loss, inputs=True)
        self.add_metric(e1 * true_loss, aggregation="mean", name="true_loss")

        self.add_loss(e2 * soft_loss, inputs=True)
        self.add_metric(e2 * soft_loss, aggregation="mean", name="soft_loss")

        self.add_loss(e3 * middle_epoch_loss, inputs=True)
        self.add_metric(e3 * middle_epoch_loss, aggregation="mean", name="epoch_loss")

        # self.add_loss(e4 * middle_seq_loss, inputs=True)
        # self.add_metric(e4 * middle_seq_loss, aggregation="mean", name="seq_loss")

        self.add_metric(categorical_accuracy(true_label, output), name="Acc")

        return e1 * true_loss + e2 * soft_loss + e3 * middle_epoch_loss
        # return e1 * true_loss + e2 * soft_loss


class MySeqLoss(KL.Layer):
    def __init__(self, **kwargs):
        super(MySeqLoss, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        # inputs：Input tensor, or list/tuple of input tensors.
        如上，父类KL.Layer的call方法明确要求inputs为一个tensor，或者包含多个tensor的列表/元组
        所以这里不能直接接受多个入参，需要把多个入参封装成列表/元组的形式然后在函数中自行解包，否则会报错。
        """
        # 解包入参
        # true_label, soft_label, middle_epoch_teacher, middle_epoch_student, middle_seq_teacher, middle_seq_student, \
        # output = inputs
        true_label, soft_label, middle_seq_teacher, middle_seq_student, output = inputs

        # print(type(middle_seq_student), type(middle_seq_teacher))
        # 复杂的损失函数
        e1 = 0.1
        e2 = 0.8
        # e3 = 0.5
        e4 = 0.01

        # 归一化处理
        if (K.int_shape(middle_seq_teacher)[0] != None):
            # middle_epoch_teacher = K.eval(middle_seq_teacher)
            # middle_epoch_student = K.eval(middle_seq_student)
            middle_seq_teacher = K.eval(middle_seq_teacher)
            middle_seq_student = K.eval(middle_seq_student)

            zscore_scaler2 = preprocessing.MinMaxScaler(feature_range=(0, 1))
            for a in range(0, len(middle_seq_teacher)):
                middle_seq_teacher[a] = zscore_scaler2.fit_transform(np.array(middle_seq_teacher[a]).T).T

            for a in range(0, len(middle_seq_student)):
                middle_seq_student[a] = zscore_scaler2.fit_transform(np.array(middle_seq_student[a]).T).T

        true_loss = categorical_crossentropy(true_label, output)
        soft_loss = categorical_crossentropy(soft_label, output)

        kl = tf.keras.losses.KLDivergence()

        middle_seq_loss = kl(middle_seq_teacher, middle_seq_student)

        true_loss = K.mean(true_loss)
        soft_loss = K.mean(soft_loss)

        self.add_loss(e1 * true_loss, inputs=True)
        self.add_metric(e1 * true_loss, aggregation="mean", name="true_loss")

        self.add_loss(e2 * soft_loss, inputs=True)
        self.add_metric(e2 * soft_loss, aggregation="mean", name="soft_loss")

        # self.add_loss(e3 * middle_epoch_loss, inputs=True)
        # self.add_metric(e3 * middle_epoch_loss, aggregation="mean", name="epoch_loss")

        self.add_loss(e4 * middle_seq_loss, inputs=True)
        self.add_metric(e4 * middle_seq_loss, aggregation="mean", name="seq_loss")

        self.add_metric(categorical_accuracy(true_label, output), name="Acc")

        # return e1 * true_loss + e2 * soft_loss + e3 * middle_epoch_loss + e4 * middle_seq_loss
        return e1 * true_loss + e2 * soft_loss + e4 * middle_seq_loss
        # return e1 * true_loss + e2 * soft_loss
        # return true_loss


class WbceLoss(KL.Layer):
    def __init__(self, **kwargs):
        super(WbceLoss, self).__init__(**kwargs)

    def get_seq_acc(self, y_seq_pred, y_seq_test):
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

    def call(self, inputs, **kwargs):
        """
        # inputs：Input tensor, or list/tuple of input tensors.
        如上，父类KL.Layer的call方法明确要求inputs为一个tensor，或者包含多个tensor的列表/元组
        所以这里不能直接接受多个入参，需要把多个入参封装成列表/元组的形式然后在函数中自行解包，否则会报错。
        """
        # 解包入参
        true_label, soft_label, output = inputs
        # 复杂的损失函数
        e1 = 0.5
        e2 = 0.5

        true_loss = categorical_crossentropy(true_label, output)
        soft_loss = categorical_crossentropy(soft_label, output)

        # print(middle_teacher.shape)
        # print(middle_student.shape)
        # middle_loss = kl_divergence(middle_teacher, middle_student)

        true_loss = K.mean(true_loss)
        soft_loss = K.mean(soft_loss)
        accuracy = categorical_accuracy(true_label, output)
        # acc = self.get_seq_acc(np.array(output), np.array(true_label))

        # print(np.shape(middle_loss))
        # middle_loss = K.mean(middle_loss)
        # print(np.shape(middle_loss))

        self.add_loss(e1 * true_loss, inputs=True)
        self.add_metric(e1 * true_loss, aggregation="mean", name="true_loss")

        self.add_loss(e2 * soft_loss, inputs=True)
        self.add_metric(e2 * soft_loss, aggregation="mean", name="soft_loss")
        self.add_metric(accuracy, name="Acc")
        # self.add_metric(acc, name="ACC")

        # self.add_loss(e3 * middle_loss, inputs=True)
        # self.add_metric(e3 * middle_loss, aggregation="mean", name="middle_loss")
        middle_loss = 0
        return e1 * true_loss + e2 * soft_loss


if __name__ == '__main__':
    TA_pre_model = TA_featurenet()
    TA_pre_model.summary()
    print("\n")
    TA_seq_model = TAsleepnet(TA_pre_model)