import tensorflow as tf
from config import cfg


class Model:

    def __init__(self):
        print('Create placeholders')
        self.encoder_inputs = tf.placeholder(tf.int32, shape=[None, None])
        self.decoder_inputs = tf.placeholder(tf.int32, shape=[None, None])
        self.decoder_masks = tf.placeholder(tf.int32, shape=[None, None])

        # Our targets are decoder inputs shifted by one (to ignore <bos> symbol)
        self.targets = [tf.cast(x, tf.float32) for x in self.decoder_inputs[1:]]

    def forward_model(self):
        pass
