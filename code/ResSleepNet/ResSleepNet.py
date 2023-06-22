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


if __name__ == "__main__":
    extractor = build_extractor()
    model = build_seq_model(extractor)
