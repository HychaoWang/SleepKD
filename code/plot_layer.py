import tf_GPU
import os
import tensorflow as tf
from os.path import join
from matplotlib import pyplot as plt
import matplotlib
from tensorflow.keras.callbacks import EarlyStopping
from data_loader import Loader
import pickle
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import MaxPool2D, TimeDistributed, Reshape
from tensorflow import keras

# matplotlib.use('Agg')

if __name__ == "__main__":
    path = ('./weights', './history')
    i = 4
    n = 4
    d = Loader()
    d.load_pretrain(i)
    d.load_finetune(i)
    pre_teacher = keras.models.load_model(join(path[0], f'pre_TA_ISRUCIII_256_{n}.h5'))
    seq_teacher = keras.models.load_model(join(path[0], f'seq_TA_ISRUCIII_256_{n}.h5'))
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
    seq_matrix = seq_label_generator.predict(d.X_seq_test)
    epoch_matrix = epoch_label_generator.predict(d.X_test)
    np.savez("test_array", a=epoch_matrix)
    # plt.matshow(seq_matrix)
    plt.matshow(epoch_matrix)
    plt.savefig("123.png")
    plt.show()

