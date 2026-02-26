import tensorflow as tf
from tensorflow.keras import backend as K

def categorical_focal_loss(alpha, gamma):
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.-K.epsilon())
        ce = -y_true*tf.math.log(y_pred)
        weight = alpha*tf.pow(1 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(weight*ce, axis=-1))
    return loss

class AdaptiveCategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = tf.Variable(tf.convert_to_tensor(alpha, dtype=tf.float32), trainable=False)
        self.gamma = tf.Variable(tf.convert_to_tensor(gamma, dtype=tf.float32), trainable=False)
    def call(self, y_true, y_pred):
        y_pred = tf.nn.softmax(y_pred)
        ce = -y_true*tf.math.log(tf.clip_by_value(y_pred,1e-8,1.0))
        weight = self.alpha*tf.pow(1 - y_pred, self.gamma)
        return tf.reduce_sum(weight*ce, axis=1)
