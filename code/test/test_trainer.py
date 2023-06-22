import os
from os.path import join
import tf_GPU
# from knowledge_distillation import Distiller

from test_deepsleepnet import *
from tensorflow.keras.callbacks import EarlyStopping
from data_loader import Loader
import pickle
# import keras
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


if __name__ == '__main__':
    path = ('./weights', './history')
    for p in path:
        if os.path.exists(p) is False:
            os.mkdir(p)

    VERBOSE = 1
    early_stopping = EarlyStopping(patience=20)

    PRE_EPOCH = 100
    PRE_BATCH = 200

    # SEQ_EPOCH = 100
    SEQ_EPOCH = 30
    SEQ_BATCH = 10

    d = Loader()
    d.load_finetune(6)
    # seq_model = keras.models.load_model('../weights/seq_model_6.h5')

    # get_seq_acc(seq_model.predict(d.X_seq_test), d.y_seq_test)
    # print(np.shape(seq_model.predict(d.X_seq_test)))
    # print(np.shape(d.y_seq_test))
    # seq_model.evaluate(seq_model.predict(d.X_seq_test), d.y_seq_test)

    for i in range(2, 3, 1):
        print(i, '-th fold out of 20-fold cross validation')
        d.load_finetune(i)
        student_model = student_net(25)

        seq_history = student_model.fit(
            d.X_seq_train,
            seq_model.predict(d.X_seq_train),  #
            batch_size=SEQ_BATCH,
            epochs=SEQ_EPOCH,
            verbose=VERBOSE,
            validation_data=(d.X_seq_valid, seq_model.predict(d.X_seq_valid)),  #
            callbacks=[early_stopping]
        )

        get_seq_acc(student_model.predict(d.X_seq_test), d.y_seq_test)

        # student_model.save(join(path[0], 'seq_model_' + str(i) + '.weights'))
        # with open(join(path[1], 'seq_history_' + str(i) + '.bin'), 'wb') as f:
        #     pickle.dump(seq_history.history, f)
