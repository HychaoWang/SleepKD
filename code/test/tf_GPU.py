import tensorflow as tf
import os


os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # use GPU with ID=0
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7  # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True  # allocate dynamically
sess = tf.compat.v1.Session(config=config)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
