from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import backend as K
from tensorflow.keras import layers as KL
from tensorflow.keras.metrics import categorical_accuracy
import tensorflow as tf


class KD_Loss(KL.Layer):
    def __init__(self, e1=1, e2=1, **kwargs):
        super(KD_Loss, self).__init__(**kwargs)
        self.e1 = e1
        self.e2 = e2
        self.Hard = tf.keras.losses.categorical_crossentropy
        self.soft = tf.keras.losses.KLDivergence()

    def call(self, inputs, **kwargs):
        """
        # inputs：Input tensor, or list/tuple of input tensors.
        如上，父类KL.Layer的call方法明确要求inputs为一个tensor，或者包含多个tensor的列表/元组
        所以这里不能直接接受多个入参，需要把多个入参封装成列表/元组的形式然后在函数中自行解包，否则会报错。
        """
        # 解包入参
        true_label, T_soft_label, output = inputs

        true_loss = self.e1 * self.Hard(true_label, output)
        T_soft_loss = self.e2 * self.soft(T_soft_label, output)

        true_loss = K.mean(true_loss)
        T_soft_loss = K.mean(T_soft_loss)

        self.add_loss(true_loss, inputs=True)
        self.add_metric(true_loss, aggregation="mean", name="true_loss")

        self.add_loss(T_soft_loss, inputs=True)
        self.add_metric(T_soft_loss, aggregation="mean", name="T_loss")

        self.add_metric(categorical_accuracy(true_label, output), name="Acc")

        return true_loss + T_soft_loss
