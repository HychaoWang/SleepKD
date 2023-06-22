import numpy as np
import numpy.random
from tensorflow.keras import backend as K
from tensorflow.keras import layers as KL
from tensorflow.keras.metrics import categorical_accuracy
import tensorflow as tf


def DKD_epoch(out_s, out_t, true_label):
    if (K.int_shape(out_s)[0] != None):
        kl = tf.keras.losses.KLDivergence()
        out_s = K.eval(out_s)
        out_t = K.eval(out_t)
        true_label = K.eval(true_label)
        target_y_index = np.argmax(true_label, axis=1)
        # print(target_y_index.shape)
        batch_size = len(out_s)
        x_index = np.arange(0, batch_size)
        # print(target_y_index)
        # print(out_t.shape)
        # print(out_s.shape)
        # print(true_label)

        target_s = out_s[x_index, target_y_index]
        target_t = out_t[x_index, target_y_index]
        # print(target_t.shape, target_s.shape)

        tckd = kl(target_t, target_s)
        tckd = tf.cast(tckd / batch_size, dtype=tf.float32)

        # untarget_s = np.array(list(map(lambda t: np.setdiff1d(out_s[t[0]], t[1]), enumerate(target_s))))
        # untarget_t = np.array(list(map(lambda t: np.setdiff1d(out_t[t[0]], t[1]), enumerate(target_t))))
        all_index = [0, 1, 2, 3, 4]
        untarget_index = list(map(lambda x: np.setdiff1d(all_index, x), target_y_index))
        # untarget_s = np.array([np.setdiff1d(out_s[t[0]], t[1]) for t in target_s])

        untarget_s = np.array(list(map(lambda x: x[1][x[0]], zip(untarget_index, out_s))))
        untarget_t = np.array(list(map(lambda x: x[1][x[0]], zip(untarget_index, out_t))))

        # print(untarget_index.shape)

        # for i in untarget_s:
        #     print(i.shape)

        # untarget_s = untarget_s.reshape((100, 4))

        # print(untarget_t.shape)
        # print(untarget_s.shape)
        #
        # print(target_y_index)
        # print(target_s)
        # print(untarget_s)

        nckd = kl(untarget_t, untarget_s)
        nckd = tf.cast(nckd / batch_size, dtype=tf.float32)
        # print()
        # print(target_t)
        # print(target_s)
        # print(untarget_t)
        # print(untarget_s)
        # print()
        loss = tckd + nckd
        return loss
    return -1


def DKD_seq(out_s, out_t, true_label):
    if (K.int_shape(out_s)[0] != None):
        out_s = K.reshape(out_s, (K.int_shape(out_s)[0] * K.int_shape(out_s)[1], K.int_shape(out_s)[2]))
        out_t = K.reshape(out_t, (K.int_shape(out_t)[0] * K.int_shape(out_t)[1], K.int_shape(out_t)[2]))
        true_label = K.reshape(true_label,
                               (K.int_shape(true_label)[0] * K.int_shape(true_label)[1], K.int_shape(true_label)[2]))
        return DKD_epoch(out_s, out_t, true_label)
    return -1


class DKD_epoch_Loss(KL.Layer):
    def __init__(self, e1=1, e2=1, **kwargs):
        super(DKD_epoch_Loss, self).__init__(**kwargs)
        self.e1 = e1
        self.e2 = e2
        self.Hard = tf.keras.losses.categorical_crossentropy
        self.soft = DKD_epoch

    def call(self, inputs, **kwargs):
        """
        # inputs：Input tensor, or list/tuple of input tensors.
        如上，父类KL.Layer的call方法明确要求inputs为一个tensor，或者包含多个tensor的列表/元组
        所以这里不能直接接受多个入参，需要把多个入参封装成列表/元组的形式然后在函数中自行解包，否则会报错。
        """
        # 解包入参
        true_label, T_soft_label, output = inputs

        true_loss = self.e1 * self.Hard(true_label, output)
        DKD_soft_loss = self.e2 * self.soft(output, T_soft_label, true_label)

        true_loss = K.mean(true_loss)

        self.add_loss(true_loss, inputs=True)
        self.add_metric(true_loss, aggregation="mean", name="true_loss")

        self.add_loss(DKD_soft_loss, inputs=True)
        self.add_metric(DKD_soft_loss, aggregation="mean", name="DKD_loss")

        self.add_metric(categorical_accuracy(true_label, output), name="Acc")

        return true_loss + DKD_soft_loss


class DKD_seq_Loss(KL.Layer):
    def __init__(self, e1=1, e2=1, **kwargs):
        super(DKD_seq_Loss, self).__init__(**kwargs)
        self.e1 = e1
        self.e2 = e2
        self.Hard = tf.keras.losses.categorical_crossentropy
        self.soft = DKD_seq

    def call(self, inputs, **kwargs):
        """
        # inputs：Input tensor, or list/tuple of input tensors.
        如上，父类KL.Layer的call方法明确要求inputs为一个tensor，或者包含多个tensor的列表/元组
        所以这里不能直接接受多个入参，需要把多个入参封装成列表/元组的形式然后在函数中自行解包，否则会报错。
        """
        # 解包入参
        true_label, T_soft_label, output = inputs

        true_loss = self.e1 * self.Hard(true_label, output)
        DKD_soft_loss = self.e2 * self.soft(output, T_soft_label, true_label)

        true_loss = K.mean(true_loss)

        self.add_loss(true_loss, inputs=True)
        self.add_metric(true_loss, aggregation="mean", name="true_loss")

        self.add_loss(DKD_soft_loss, inputs=True)
        self.add_metric(DKD_soft_loss, aggregation="mean", name="DKD_loss")

        self.add_metric(categorical_accuracy(true_label, output), name="Acc")

        return true_loss + DKD_soft_loss


if __name__ == "__main__":
    s = numpy.random.random((10, 5))
    t = numpy.random.random((10, 5))
    y = numpy.random.random((10, 5))
    DKD_epoch(s, t, y)
