from keras.models import Sequential
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.regularizers import l2
import tensorflow as tf
import random
from keras.layers.core import Lambda, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.legacy.layers import Highway
from keras.layers import TimeDistributed
import numpy as np

class MatchLayer(Layer):
    def __init__(self ,unit=16,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        self.unit = unit
        self.epsilon = 1e-6
        super(MatchLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重
        # self.w2 =self.add_weight(name='w2',
        #                           shape=(input_shape[0][-1], input_shape[0][1]),
        #                           initializer='glorot_normal',
        #                           regularizer=l2(0.000001),
        #                           trainable=True)
        self.w1 = self.add_weight(name='w1',
                                  shape=(input_shape[0][1],self.unit),
                                  initializer='glorot_normal',
                                  regularizer=l2(0.000001),
                                  trainable=True)
        #
        # self.w2 = self.add_weight(name='w2',
        #                           shape=(input_shape[0][1], self.unit),
        #                           initializer='glorot_normal',
        #                           regularizer=l2(0.000001),
        #                           trainable=True)
        #
        # self.w3 = self.add_weight(name='w3',
        #                           shape=(input_shape[0][1], self.unit),
        #                           initializer='glorot_normal',
        #                           regularizer=l2(0.000001),
        #                           trainable=True)
        # self.we = self.add_weight(name='we',
        #                           shape=(self.unit, 1),
        #                           initializer='glorot_normal',
        #                           regularizer=l2(0.000001),
        #                           trainable=True)
        # if self.bias:
        #     self.b1 = self.add_weight(name='b1',shape=(self.unit,),
        #                              initializer='zeros',
        #                               regularizer=l2(0.00001),
        #                               trainable=True)
        # self.bh = self.add_weight(name='bh',shape=None,
        #                          initializer='zeros',
        #                           trainable=True)
        super(MatchLayer, self).build(input_shape)  # 一定要在最后调用它

    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None

    def call(self, x, mask=None):
        assert isinstance(x, list)
        x1, x2 = x

        match1 = self._cosine_matrix(x1,x2)
        match2 = self._bilinear_dot(x1, x2)
        match3 = self.match(x1, x2)
        #
        #return [match1,match3]
        return [match1, match2,match3]
        # if mask != None:
        #     ms1 = tf.cast(mask[0], 'float32')
        #     ms1 = K.reshape(ms1, (-1, input_shape[1], 1))
        #     final_match = final_match * ms1


    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0], shape_a[1], shape_a[1]),(shape_a[0], shape_a[1], self.unit),(shape_a[0], shape_a[1], shape_a[2]*6)]
        #return [(shape_a[0], shape_a[1], shape_a[1]),(shape_a[0], shape_a[1], shape_a[2] * 6)]
        # return (shape_a[0], shape_a[1], shape_a[1])

    def _cosine_similarity(self, x1, x2):
        """Compute cosine similarity.
        # Arguments:
            x1: (..., embedding_size)
            x2: (..., embedding_size)
        """

        cos = K.sum(x1 * x2, axis=-1)
        x1_norm = K.sqrt(K.maximum(K.sum(K.square(x1), axis=-1), self.epsilon))
        x2_norm = K.sqrt(K.maximum(K.sum(K.square(x2), axis=-1), self.epsilon))
        cos = (cos / x1_norm) / x2_norm
        return cos
    def _bilinear_dot(self, x1, x2):
        # t = tf.matmul(x1,self.w1)
        # temp = tf.matmul(t,x2)
        bi_dot = K.dot(K.batch_dot(x1,x2,axes=2), self.w1)
        # input_shape = K.shape(bi_dot)
        # bi_dot = K.reshape(bi_dot, (input_shape[0], input_shape[1], input_shape[-1]))
        return bi_dot
        #return temp

    def match(self,x1,x2):
        add = x1 + x2
        sub = x1 - x2
        mult = x1 * x2
        ks = K.abs(sub)
        norm = K.l2_normalize(sub, axis=-1)
        out = K.concatenate([x1, x2, add, mult, ks, norm], axis=-1)
        return out
    def _cosine_matrix_mean(self, x1, x2):
        x1_mean = K.mean(x1,axis=2)
        input_shape = K.shape(x1)
        x1_mean = tf.tile(K.expand_dims(x1_mean, 2), [1,1,input_shape[2]])  # bs,sl,sl 对当前张量内的数据进行一定规则的复制

        x1_new = x1 - x1_mean

        x2_mean = K.mean(x2,axis=2)
        x2_mean = tf.tile(K.expand_dims(x2_mean, 2), [1, 1, input_shape[2]])  # bs,sl,sl 对当前张量内的数据进行一定规则的复制
        x2_new = x2 - x2_mean

        # expand h1 shape to (batch_size, x1_timesteps, 1, embedding_size)
        x1 = K.expand_dims(x1_new, axis=2)
        # expand x2 shape to (batch_size, 1, x2_timesteps, embedding_size)
        x2 = K.expand_dims(x2_new, axis=1)
        # cosine matrix (batch_size, h1_timesteps, h2_timesteps)
        cos_matrix = self._cosine_similarity(x1, x2)
        return cos_matrix

    def _cosine_matrix(self, x1, x2):
        # expand h1 shape to (batch_size, x1_timesteps, 1, embedding_size)
        x1 = K.expand_dims(x1, axis=2)
        # expand x2 shape to (batch_size, 1, x2_timesteps, embedding_size)
        x2 = K.expand_dims(x2, axis=1)
        # cosine matrix (batch_size, h1_timesteps, h2_timesteps)
        cos_matrix = self._cosine_similarity(x1, x2)
        return cos_matrix

