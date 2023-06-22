#!/usr/bin/env python
"""
Maximum Mean Discrepancy (MMD)
The MMD is implemented as keras regularizer that can be used for
shared layers. This implementation uis tested under keras 1.1.0.
- Gretton, Arthur, et al. "A kernel method for the two-sample-problem."
Advances in neural information processing systems. 2007.
__author__ = "Werner Zellinger"
__copyright__ = "Copyright 2017, Werner Zellinger"
__credits__ = ["Thomas Grubinger, Robert Pollak"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Werner Zellinger"
__email__ = "werner.zellinger@jku.at"
"""

import tensorflow as tf
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras import backend as K
from tensorflow.keras import layers as KL
from tensorflow.keras.metrics import categorical_accuracy, KLDivergence


def mmd(x1, x2, beta):
    """
    maximum mean discrepancy (MMD) based on Gaussian kernel
    function for keras models (theano or tensorflow backend)

    - Gretton, Arthur, et al. "A kernel method for the two-sample-problem."
    Advances in neural information processing systems. 2007.
    """
    x1x1 = gaussian_kernel(x1, x1, beta)
    x1x2 = gaussian_kernel(x1, x2, beta)
    x2x2 = gaussian_kernel(x2, x2, beta)
    diff = K.mean(x1x1) - 2 * K.mean(x1x2) + K.mean(x2x2)
    return diff


def gaussian_kernel(x1, x2, beta=1.0):
    # r = x1.dimshuffle(0, 'x', 1)
    r = K.expand_dims(x1, axis=1)
    return K.exp(K.sum(-beta * K.square(r - x2), axis=-1))


class MMDRegularizer(Regularizer):
    """
    class structure to use the MMD as activity regularizer of a
    keras shared layer
    """

    def __init__(self, l=1, beta=1.0):
        self.uses_learning_phase = 1
        self.l = l
        self.beta = beta

    def set_layer(self, layer):
        # needed for keras layer
        self.layer = layer

    def __call__(self, loss):
        if not hasattr(self, 'layer'):
            raise Exception('Need to call `set_layer` on '
                            'ActivityRegularizer instance '
                            'before calling the instance.')
        regularizer_loss = loss
        sim = 0
        if len(self.layer.inbound_nodes) > 1:
            # we are in a shared keras layer
            sim = mmd(self.layer.get_output_at(0),
                      self.layer.get_output_at(1),
                      self.beta)
        add_loss = K.switch(K.equal(len(self.layer.inbound_nodes), 2), sim, 0)
        regularizer_loss += self.l * add_loss
        return K.in_train_phase(regularizer_loss, loss)

    def get_config(self):
        # needed for keras layer
        return {'name': self.__class__.__name__,
                'l': float(self.l)}


class NSTLoss(KL.Layer):
    def __init__(self, e1=1, e2=1, e3=1, **kwargs):
        super(NSTLoss, self).__init__(**kwargs)
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3

    def call(self, inputs, **kwargs):
        """
        # inputs：Input tensor, or list/tuple of input tensors.
        如上，父类KL.Layer的call方法明确要求inputs为一个tensor，或者包含多个tensor的列表/元组
        所以这里不能直接接受多个入参，需要把多个入参封装成列表/元组的形式然后在函数中自行解包，否则会报错。
        """
        # 解包入参
        true_label, soft_label, middle_teacher, middle_student, output = inputs

        MSE = tf.keras.losses.MeanSquaredError()
        CE = tf.keras.losses.categorical_crossentropy
        KL = tf.keras.losses.KLDivergence()

        true_loss = self.e1 * CE(true_label, output)
        soft_loss = self.e2 * KL(soft_label, output)
        middle_loss = self.e3 * mmd(middle_teacher, middle_student, 1)

        true_loss = K.mean(true_loss)
        soft_loss = K.mean(soft_loss)

        self.add_loss(true_loss, inputs=True)
        self.add_metric(true_loss, aggregation="mean", name="true_loss")

        self.add_loss(soft_loss, inputs=True)
        self.add_metric(soft_loss, aggregation="mean", name="soft_loss")

        self.add_loss(middle_loss, inputs=True)
        self.add_metric(middle_loss, aggregation="mean", name="MMD_loss")

        self.add_metric(categorical_accuracy(true_label, output), name="Acc")

        return true_loss + soft_loss + middle_loss
