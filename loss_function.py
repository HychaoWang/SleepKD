import numpy as np
import tensorflow.keras.backend as K
import yaml
import tensorflow as tf
from tensorflow.keras.metrics import categorical_accuracy, categorical_crossentropy
from nst import *
from dkd import *


class SleepKD(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SleepKD, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        true_label, soft_label, epoch_teacher, epoch_student, seq_teacher, seq_student, \
        output = inputs
        
        e1 = 0.1
        e2 = 0.4
        e3 = 0.3
        e4 = 0.2

        if (K.int_shape(epoch_teacher)[0] != None):
            epoch_teacher = K.eval(epoch_teacher)
            epoch_student = K.eval(epoch_student)
            seq_teacher = K.eval(seq_teacher)
            seq_student = K.eval(seq_student)

        true_loss = categorical_crossentropy(true_label, output)
        soft_loss = categorical_crossentropy(soft_label, output)

        kl = tf.keras.losses.KLDivergence()
        epoch_loss = kl(epoch_teacher, epoch_student)
        seq_loss = kl(seq_teacher, seq_student)
        true_loss = K.mean(true_loss)
        soft_loss = K.mean(soft_loss)
        

        self.add_loss(e1 * true_loss, inputs=True)
        self.add_metric(e1 * true_loss, aggregation="mean", name="true_loss")

        self.add_loss(e2 * soft_loss, inputs=True)
        self.add_metric(e2 * soft_loss, aggregation="mean", name="soft_loss")

        self.add_loss(e3 * epoch_loss, inputs=True)
        self.add_metric(e3 * epoch_loss, aggregation="mean", name="epoch_loss")

        self.add_loss(e4 * seq_loss, inputs=True)
        self.add_metric(e4 * seq_loss, aggregation="mean", name="seq_loss")

        self.add_metric(categorical_accuracy(true_label, output), name="acc")
        return e1 * true_loss + e2 * soft_loss + e3 * epoch_loss + e4 * seq_loss