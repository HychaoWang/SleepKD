from unicodedata import name
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Dropout, MaxPool1D, Activation, LayerNormalization, Reshape, \
    LSTM, TimeDistributed, Bidirectional, BatchNormalization
from tensorflow.keras.layers import Conv2D, Permute, Flatten, MaxPool2D, Concatenate
from tensorflow.keras.losses import *
from tensorflow.keras import backend as K
from tensorflow.keras import layers as KL
from tensorflow.keras import models as KM
from tensorflow.python.keras.metrics import categorical_accuracy
from sklearn import preprocessing


def featurenet():
    activation = tf.nn.relu
    # activation = tf.nn.leaky_relu
    padding = 'same'

    ######### Input ########
    input_signal = Input(shape=(30 * 100, 1), name='input_signal')
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
    cnn3 = Conv1D(kernel_size=8, filters=128, strides=1, padding=padding)
    s = cnn3(s)
    s = BatchNormalization()(s)
    s = Activation(activation=activation)(s)
    # print("cnn3:",s.shape)
    cnn4 = Conv1D(kernel_size=8, filters=128, strides=1, padding=padding)
    s = cnn4(s)
    s = BatchNormalization()(s)
    s = Activation(activation=activation)(s)
    # print("cnn4:",s.shape)
    cnn5 = Conv1D(kernel_size=8, filters=128, strides=1, padding=padding)
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
    cnn11 = Conv1D(kernel_size=6, filters=128, strides=1, padding=padding)
    l = cnn11(l)
    l = BatchNormalization()(l)
    l = Activation(activation=activation)(l)
    # print("cnn11:",l.shape)
    cnn12 = Conv1D(kernel_size=6, filters=128, strides=1, padding=padding)
    l = cnn12(l)
    l = BatchNormalization()(l)
    l = Activation(activation=activation)(l)
    # print("cnn12:",l.shape)
    cnn13 = Conv1D(kernel_size=6, filters=128, strides=1, padding=padding)
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
    merged = Dropout(0.5)(merged)
    merged = Dense(5, name='merged')(merged)
    # print('merged',merged.shape)
    pre_softmax = Activation(activation='softmax')(merged)
    # print('pre_softmax',pre_softmax.shape)

    pre_model = Model(input_signal, pre_softmax)
    pre_opt = keras.optimizers.Adam(lr=1e-4)
    pre_model.compile(optimizer=pre_opt, loss='categorical_crossentropy', metrics=['acc'])

    return pre_model


def deepsleepnet(pre_model):
    input_signal = pre_model.get_layer(name='input_signal').input
    merged = pre_model.get_layer(name='merged').output

    activation_seq = 'relu'

    cnn_part = Model(input_signal, merged)  # pre train 된 부분

    input_seq = Input(shape=(25, 3000, 1))  # sequence 길이 모르므로 None
    # print('input_seq', input_seq.shape)

    signal_sequence = TimeDistributed(cnn_part)(input_seq)  # TimeDistributed 로 시퀀스를 입력 받을 수 있음
    print('signal_sequence', signal_sequence.shape)

    bidirection = Bidirectional(LSTM(512, dropout=0.5, activation=activation_seq, return_sequences=True),
                                merge_mode='concat')(signal_sequence)
    # print('bidirection', bidirection.shape)

    fc1024 = Dense(1024)(signal_sequence)
    fc1024 = BatchNormalization()(fc1024)
    fc1024 = Activation(activation=activation_seq)(fc1024)
    # print('fc1024',fc1024.shape)
    residual = keras.layers.add([bidirection, fc1024])  # skip-connection
    residual = Dropout(0.5)(residual)
    # print('residual',residual.shape)

    dense_seq = Dense(5)(residual)
    # print('dense_seq',dense_seq.shape)

    seq_softmax = Activation(activation='softmax')(dense_seq)
    # print('seq_softmax',seq_softmax.shape)

    seq_model = Model(input_seq, seq_softmax)
    seq_opt = keras.optimizers.Adam(lr=1e-6)
    seq_model.compile(loss='categorical_crossentropy', optimizer=seq_opt, metrics=['acc'])

    return seq_model


# def student_net(ch, mode="predict"):
#     assert mode in ("train", "predict"), "only 'train' and 'predict' mode supported"

def student_net(ch=25):
    Input_data = keras.Input(shape=(ch, 3000, 1), name="Input_Seq_Signal")
    padding = 'same'
    # if mode == "train":
    true_label = Input([ch, 5], name="true_label")
    soft_label = Input([ch, 5], name="soft_label")
    middle_epoch_teacher = Input([ch, 88], name="middle_epoch_teacher")
    middle_seq_teacher = Input([ch, 11], name="middle_seq_teacher")

    cov1 = Conv2D(filters=8, kernel_size=(1, 64), strides=(1, 1), activation='relu', padding=padding)(Input_data)
    cov1 = BatchNormalization()(cov1)

    cov2 = Conv2D(filters=8, kernel_size=(1, 64), strides=(1, 1), activation='relu', padding=padding)(cov1)
    cov2 = BatchNormalization()(cov2)

    cov3 = MaxPool2D(pool_size=(1, 16), strides=(1, 16))(cov2)

    cov4 = Conv2D(filters=8, kernel_size=(1, 64), strides=(1, 1), activation='relu', padding=padding)(cov3)
    cov4 = BatchNormalization()(cov4)

    cov5 = MaxPool2D(pool_size=(1, 16), strides=(1, 16))(cov4)

    middle_epoch_layer = TimeDistributed(Flatten())(cov5)

    cov6 = Conv2D(filters=ch, kernel_size=(ch, 1), strides=(1, 1),
                  activation='linear')(cov5)

    middle_seq_layer = Permute((3, 2, 1))(cov6)
    middle_seq_layer = TimeDistributed(Flatten())(middle_seq_layer)

    cov6 = BatchNormalization()(cov6)

    cov7 = Permute((3, 2, 1))(cov6)

    cov8 = TimeDistributed(Flatten())(cov7)
    # print(cov8.shape)

    cov9 = Dropout(0.5)(cov8)
    # print(cov9.shape)

    output = TimeDistributed(Dense(5, activation='softmax'), name='softmax')(cov9)
    # print(output.shape)

    middle_epoch_student = middle_epoch_layer
    middle_seq_student = middle_seq_layer

    # if mode == "train":
    my_loss = WbceLoss()(
        [true_label, soft_label, middle_epoch_teacher, middle_epoch_student, middle_seq_teacher, middle_seq_student,
         output])
    model = KM.Model(inputs=[Input_data, true_label, soft_label, middle_epoch_teacher, middle_seq_teacher],
                     outputs=[output, my_loss])
    model.compile(optimizer="adam", run_eagerly=True)

    return model


class WbceLoss(KL.Layer):
    def __init__(self, **kwargs):
        super(WbceLoss, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        # inputs：Input tensor, or list/tuple of input tensors.
        如上，父类KL.Layer的call方法明确要求inputs为一个tensor，或者包含多个tensor的列表/元组
        所以这里不能直接接受多个入参，需要把多个入参封装成列表/元组的形式然后在函数中自行解包，否则会报错。
        """
        # 解包入参
        true_label, soft_label, middle_epoch_teacher, middle_epoch_student, middle_seq_teacher, middle_seq_student, \
        output = inputs
        # 复杂的损失函数
        e1 = 0.25
        e2 = 0.25
        e3 = 0.35
        e4 = 0.15

        # NormalizeLayer = LayerNormalization(axis=1, center=True, scale=True)

        # 归一化处理
        if (K.int_shape(middle_epoch_teacher)[0] != None):
            # middle_epoch_student = NormalizeLayer(middle_epoch_student)
            # middle_epoch_teacher = NormalizeLayer(middle_epoch_teacher)
            # middle_seq_student = NormalizeLayer(middle_seq_student)
            # middle_seq_teacher = NormalizeLayer(middle_seq_teacher)

            middle_epoch_teacher = K.eval(middle_epoch_teacher)
            middle_epoch_student = K.eval(middle_epoch_student)
            middle_seq_teacher = K.eval(middle_seq_teacher)
            middle_seq_student = K.eval(middle_seq_student)
            zscore_scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1))
            for a in range(0, len(middle_epoch_teacher)):
                middle_epoch_teacher[a] = zscore_scaler1.fit_transform(middle_epoch_teacher[a].T).T
            # for a in range(0, len(middle_epoch_teacher)):
            #     for b in range(0, len(middle_epoch_teacher[a])):
            #         for c in range(0, len(middle_epoch_teacher[a][b])):
            #             middle_epoch_teacher[a][b][c] /= np.sum(middle_epoch_teacher[a][b])

            for a in range(0, len(middle_epoch_student)):
                middle_epoch_student[a] = zscore_scaler1.fit_transform(middle_epoch_student[a].T).T
            # for a in range(0, len(middle_epoch_student)):
            #     for b in range(0, len(middle_epoch_student[a])):
            #         for c in range(0, len(middle_epoch_student[a][b])):
            #             middle_epoch_student[a][b][c] /= np.sum(middle_epoch_student[a][b])

            zscore_scaler2 = preprocessing.MinMaxScaler(feature_range=(0, 1))
            for a in range(0, len(middle_seq_teacher)):
                middle_seq_teacher[a] = zscore_scaler1.fit_transform(middle_seq_teacher[a].T).T
            # for a in range(0, len(middle_seq_teacher)):
            #     for b in range(0, len(middle_seq_teacher[a])):
            #         for c in range(0, len(middle_seq_teacher[a][b])):
            #             middle_seq_teacher[a][b][c] /= np.sum(middle_seq_teacher[a][b])

            for a in range(0, len(middle_seq_student)):
                middle_seq_student[a] = zscore_scaler1.fit_transform(middle_seq_student[a].T).T
            # for a in range(0, len(middle_seq_student)):
            #     for b in range(0, len(middle_seq_student[a])):
            #         for c in range(0, len(middle_seq_student[a][b])):
            #             middle_seq_student[a][b][c] /= np.sum(middle_seq_student[a][b])

        # print(middle_epoch_teacher)
        # print(middle_epoch_teacher.shape)
        # print(middle_epoch_student)
        # print(middle_epoch_student.shape)

        # print(middle_seq_teacher)
        # print(middle_seq_teacher.shape)
        # print(middle_seq_student)
        # print(middle_seq_student.shape)

        true_loss = categorical_crossentropy(true_label, output)
        soft_loss = categorical_crossentropy(soft_label, output)

        kl = tf.keras.losses.KLDivergence()
        middle_epoch_loss = kl(middle_epoch_teacher, middle_epoch_student)
        middle_seq_loss = kl(middle_seq_teacher, middle_seq_student)
        # middle_epoch_loss = mean_squared_error(middle_epoch_teacher, middle_epoch_student)
        # middle_seq_loss = mean_squared_error(middle_seq_teacher, middle_seq_student)
        # print(middle_epoch_loss)
        # print(middle_seq_loss)
        # print(middle_epoch_loss.shape)
        # print(middle_seq_loss.shape)
        true_loss = K.mean(true_loss)
        soft_loss = K.mean(soft_loss)

        self.add_loss(e1 * true_loss, inputs=True)
        self.add_metric(e1 * true_loss, aggregation="mean", name="true_loss")

        # self.add_loss(e2 * soft_loss, inputs=True)
        # self.add_metric(e2 * soft_loss, aggregation="mean", name="soft_loss")

        # self.add_loss(e3 * middle_epoch_loss, inputs=True)
        # self.add_metric(e3 * middle_epoch_loss, aggregation="mean", name="middle_epoch_loss")

        # self.add_loss(e4 * middle_seq_loss, inputs=True)
        # self.add_metric(e4 * middle_seq_loss, aggregation="mean", name="middle_seq_loss")

        self.add_metric(categorical_accuracy(true_label, output), name="train_Acc")

        # return e1 * true_loss + e2 * soft_loss + e3 * middle_epoch_loss + e4 * middle_seq_loss
        # return e1 * true_loss + e2 * soft_loss + e3 * middle_epoch_loss
        # return e1 * true_loss + e2 * soft_loss+ e4 * middle_seq_loss
        return e1* true_loss

        # final_loss = e * true_loss + (1 - e) * soft_loss
        # wbce_loss = K.mean(final_loss)  # ?

        # 重点：把自定义的loss添加进层使其生效，同时加入metric方便在KERAS的进度条上实时追踪
        # self.add_loss(wbce_loss, inputs=True)
        # self.add_metric(wbce_loss, aggregation="mean", name="wbce_loss")
        # return wbce_loss

import sys

sys.path.append("..")
import tf_GPU
import tensorflow as tf
from keras.layers import Reshape, LSTM, Bidirectional, BatchNormalization, Reshape, LSTM, \
    TimeDistributed, BatchNormalization, Input, Conv1D, Activation, MaxPool1D, add, AveragePooling1D, \
    GlobalAveragePooling1D, Dense, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow import keras


def mm():
    input_EEG = Input((512, 1), name="EEG_input")
    x = Dense(1024)(input_EEG)
    x = Dense(2)(x)
    model = Model(input_EEG, x)
    model.summary()


def build_extractor():
    activation = tf.nn.relu
    input_signal = Input(shape=(30 * 100, 1), name='input_signal')
    Sd, Fs = int(30 * 100 / 1000), int(30 * 100 / 100)

    print(Sd, Fs)

    cnn0 = Conv1D(
        kernel_size=Fs,
        filters=128,
        strides=Sd, kernel_regularizer=keras.regularizers.l2(0.001))
    x = cnn0(input_signal)
    x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)

    print(x.shape)

    m = int(Fs * 3.14)

    print(m)

    max_0 = MaxPool1D(pool_size=Fs, strides=Sd, padding="same")(x)
    max_1 = MaxPool1D(pool_size=m, strides=Sd, padding="same")(x)

    print(max_1.shape, max_0.shape)

    x = add([max_0, max_1])
    x = BatchNormalization()(x)

    print(x.shape)

    residual = AveragePooling1D(pool_size=2, strides=2,
                                padding="same")(x)

    print(x.shape)

    #     CNN block 1
    cnn1_1 = Conv1D(
        kernel_size=1,
        filters=128,
        strides=2, kernel_regularizer=keras.regularizers.l2(0.001),
        padding="same")
    x = cnn1_1(x)
    x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)

    print(x.shape)

    cnn1_2 = Conv1D(
        kernel_size=3,
        filters=64,
        strides=1, kernel_regularizer=keras.regularizers.l2(0.001),
        padding="same")
    x = cnn1_2(x)
    x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)

    print(x.shape)

    cnn1_3 = Conv1D(
        kernel_size=1,
        filters=128,
        strides=1, kernel_regularizer=keras.regularizers.l2(0.001),
        padding="same")
    x = cnn1_3(x)
    x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)

    print(x.shape, residual.shape)

    x = add([residual, x])
    x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)

    print(x.shape)

    residual = AveragePooling1D(pool_size=2, strides=2,
                                padding="same")(x)

    #     CNN block 2
    cnn2_1 = Conv1D(
        kernel_size=1,
        filters=128,
        strides=2, kernel_regularizer=keras.regularizers.l2(0.001),
        padding="same")
    x = cnn2_1(x)
    x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)

    print(x.shape)

    cnn2_2 = Conv1D(
        kernel_size=3,
        filters=64,
        strides=1, kernel_regularizer=keras.regularizers.l2(0.001),
        padding="same")
    x = cnn2_2(x)
    x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)

    print(x.shape)

    cnn2_3 = Conv1D(
        kernel_size=1,
        filters=128,
        strides=1, kernel_regularizer=keras.regularizers.l2(0.001),
        padding="same")
    x = cnn2_3(x)
    x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)

    print(x.shape, residual.shape)

    x = add([residual, x])
    x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)

    GAP_feature = GlobalAveragePooling1D(name="GAP_output")(x)
    print(GAP_feature.shape)

    model = Model(input_signal, GAP_feature)

    model.summary()
    return model


def build_seq_model(extractor):
    input_seq = Input(shape=(None, 3000, 1), name="Input_Seq_Signal")
    signal_sequence = TimeDistributed(extractor)(input_seq)
    # print(signal_sequence.shape)
    sequnce_output = MultiHeadAttention(num_heads=8, key_dim=64, dropout=0.8)(signal_sequence, signal_sequence)

    print(sequnce_output.shape)

    residual = BatchNormalization()(sequnce_output)

    densed = Dense(2048, activation="ReLU")(sequnce_output)
    densed = Dense(128)(densed)

    added = add([densed, residual])

    softmax_output = Dense(5, activation="softmax")(added)

    print(softmax_output.shape)

    model = Model(input_seq, softmax_output)

    model.summary()



if __name__ == '__main__':
    # pre_model = featurenet()
    # pre_model.summary()
    # seq_model = deepsleepnet(pre_model)
    # seq_model.summary()
    #
    # student_model = student_net(25)
    # student_model.summary()

    extractor = build_extractor()
    build_seq_model(extractor)