import tf_GPU
import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import backend as K
from tensorflow.keras import layers as KL
from tensorflow.keras.metrics import categorical_accuracy, KLDivergence
import numpy as np
from sklearn import preprocessing


class MMLKDLoss(KL.Layer):
    def __init__(self, e1=1, e2=1, e3=1, e4=1, **kwargs):
        super(MMLKDLoss, self).__init__(**kwargs)
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3
        self.e4 = e4
        self.intermediate_sequence = tf.keras.losses.MeanSquaredError()
        self.intermediate_epoch = tf.keras.losses.KLDivergence()
        self.hard = tf.keras.losses.categorical_crossentropy
        self.soft = tf.keras.losses.KLDivergence()

    def call(self, inputs, **kwargs):
        """
        # inputs：Input tensor, or list/tuple of input tensors.
        如上，父类KL.Layer的call方法明确要求inputs为一个tensor，或者包含多个tensor的列表/元组
        所以这里不能直接接受多个入参，需要把多个入参封装成列表/元组的形式然后在函数中自行解包，否则会报错。
        """
        # 解包入参
        true_label, soft_label, epoch_teacher, epoch_student, seq_teacher, seq_student, output = inputs

        true_loss = self.e1 * self.hard(true_label, output)
        soft_loss = self.e2 * self.soft(soft_label, output)
        epoch_loss = self.e3 * self.intermediate_epoch(epoch_teacher, epoch_student)
        seq_loss = self.e4 * self.intermediate_sequence(seq_teacher, seq_student)

        true_loss = K.mean(true_loss)
        soft_loss = K.mean(soft_loss)

        self.add_loss(true_loss, inputs=True)
        self.add_metric(true_loss, aggregation="mean", name="true_loss")

        self.add_loss(soft_loss, inputs=True)
        self.add_metric(soft_loss, aggregation="mean", name="soft_loss")

        self.add_loss(epoch_loss, inputs=True)
        self.add_metric(epoch_loss, aggregation="mean", name="epoch_loss")

        self.add_loss(seq_loss, inputs=True)
        self.add_metric(seq_loss, aggregation="mean", name="seq_loss")

        self.add_metric(categorical_accuracy(true_label, output), name="Acc")

        return true_loss + soft_loss + epoch_loss + seq_loss
