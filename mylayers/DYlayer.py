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

#上下文编码层
class ContextLayer(object):
    """Word context layer
    """
    def __init__(self, rnn_dim, rnn_unit='gru', input_shape=(0,),
                 dropout=0.0, highway=False, return_sequences=False,
                 dense_dim=0):
        if rnn_unit == 'gru':
            rnn = GRU
        else:
            rnn = LSTM
        self.model = Sequential()
        self.model.add(
            Bidirectional(rnn(rnn_dim,
                              dropout=dropout,
                              recurrent_dropout=dropout,
                              return_sequences=return_sequences),
                          input_shape=input_shape))
        # self.model.add(rnn(rnn_dim,
        #                    dropout=dropout,
        #                    recurrent_dropout=dropout,
        #                    return_sequences=return_sequences,
        #                    input_shape=input_shape))
        if highway:
            if return_sequences:
                self.model.add(TimeDistributed(Highway(activation='tanh')))
            else:
                self.model.add(Highway(activation='tanh'))

        if dense_dim > 0:
            self.model.add(TimeDistributed(Dense(dense_dim,
                                                 activation='relu')))
            self.model.add(TimeDistributed(Dropout(dropout)))
            self.model.add(TimeDistributed(BatchNormalization()))

    def __call__(self, inputs):
        return self.model(inputs)

class PredictLayer(object):
    """Prediction layer.

    """
    def __init__(self, dense_dim, input_dim=0,
                 dropout=0.0):
        self.model = Sequential()
        self.model.add(Dense(dense_dim,
                             activation='relu',
                             input_shape=(input_dim,)))
        self.model.add(Dropout(dropout))
        self.model.add(BatchNormalization())
        self.model.add(Dense(1, activation='sigmoid'))

    def __call__(self, inputs):
        return self.model(inputs)

class WKS(Layer):
    def __init__(self ,bias=True ,sr=10 ,unit=16 ,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        super(WKS, self).__init__(**kwargs)
        self.bias =bias
        self.sr =sr
        self.unit =unit
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重
        self.w1 =self.add_weight(name='w1',
                                  shape=(input_shape[0][-1], self.unit),
                                  initializer='glorot_normal',
                                  regularizer=l2(0.000001),
                                  trainable=True)

        self.w2 = self.add_weight(name='w2',
                                  shape=(input_shape[0][1], self.unit),
                                  initializer='glorot_normal',
                                  regularizer=l2(0.000001),
                                  trainable=True)

        self.w3 = self.add_weight(name='w3',
                                  shape=(input_shape[0][1], self.unit),
                                  initializer='glorot_normal',
                                  regularizer=l2(0.000001),
                                  trainable=True)
        self.we = self.add_weight(name='we',
                                  shape=(self.unit, 1),
                                  initializer='glorot_normal',
                                  regularizer=l2(0.000001),
                                  trainable=True)
        # if self.bias:
        #     self.b1 = self.add_weight(name='b1',shape=(self.unit,),
        #                              initializer='zeros',
        #                               regularizer=l2(0.00001),
        #                               trainable=True)
        # self.bh = self.add_weight(name='bh',shape=None,
        #                          initializer='zeros',
        #                           trainable=True)
        super(WKS, self).build(input_shape)  # 一定要在最后调用它

    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None

    def call(self, x, mask=None):
        assert isinstance(x, list)
        x1, x2 = x
        input_shape = K.shape(x1)
        pq = K.batch_dot(x1, x2, axes=2) #x1=(?,15,128) x2=(?,15,128) axes=2 表示128作为矩阵相乘相同的维度 因此pq=(?,15,15)
        pp = K.batch_dot(x1, x1, axes=2)
        qp = K.batch_dot(x2, x1, axes=2)
        qq = K.batch_dot(x2, x2, axes=2)
        #w1=(128,16) w2=(15,16) w3=(15,16) we=(16,1)
        eij1 = K.dot(K.tanh(K.dot(pq, self.w2) + K.dot(pp, self.w3) + K.dot(x1, self.w1)), self.we) * self.sr # eij1=(?,15,1)
        # k = K.expand_dims(K.dot(x2, self.w2)+self.b2, 1)
        ai1 = K.exp(eij1) #ai1=(?,15,1)
        if mask != None:
            ms1 = tf.cast(mask[0], 'float32')
            ms1 = K.reshape(ms1, (-1, input_shape[1], 1))
            ai1 = ai1 * ms1

        ww1 = ai1 / (K.sum(ai1, axis=1, keepdims=True) + K.epsilon())

        ot1 = x1 * ww1
        # e2 = (e1 - K.min(e2, axis=1, keepdims=True))/(K.max(e2, axis=1, keepdims=True)-K.min(e2, axis=1, keepdims=True)+ K.epsilon())
        eij2 = K.dot(K.tanh(K.dot(qp, self.w2) + K.dot(qq, self.w3) + K.dot(x2, self.w1)), self.we) * self.sr
        # eij2 = *self.sr
        ai2 = K.exp(eij2)
        if mask != None:
            ms2 = tf.cast(mask[1], 'float32')
            ms2 = K.reshape(ms2, (-1, input_shape[1], 1))
            ai2 = ai2 * ms2
        ww2 = ai2 / (K.sum(ai2, axis=1, keepdims=True) + K.epsilon())
        ot2 = x2 * ww2
        # return weighted_input.sum(axis=1)
        # oot1=K.sum(ot1,axis=1)
        # oot2=K.sum(ot2,axis=1)
        oot1 = K.sum(ot1, axis=1)
        oot2 = K.sum(ot2, axis=1)
        # return [ww1,ww2]
        return [ww1, ww2, ot1, ot2, oot1, oot2]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0], shape_a[1], 1), (shape_b[0], shape_b[1], 1), (shape_a[0], shape_a[1], shape_a[2]),
                (shape_b[0], shape_b[1], shape_b[2]), (shape_a[0], shape_a[2]), (shape_b[0], shape_b[2])]
        # return [(shape_a[0],shape_a[1],1), (shape_b[0],shape_b[1],1)]


class CrossAttention(Layer):
    def __init__(self ,bias=True ,sr=10 ,unit=16 ,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        super(CrossAttention, self).__init__(**kwargs)
        self.bias =bias
        self.sr =sr
        self.unit =unit
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重
        self.w1 =self.add_weight(name='w1',
                                  shape=(input_shape[0][-1], self.unit),
                                  initializer='glorot_normal',
                                  regularizer=l2(0.000001),
                                  trainable=True)

        self.w2 = self.add_weight(name='w2',
                                  shape=(input_shape[0][1], self.unit),
                                  initializer='glorot_normal',
                                  regularizer=l2(0.000001),
                                  trainable=True)

        # self.w3 = self.add_weight(name='w3',
        #                           shape=(input_shape[0][1], self.unit),
        #                           initializer='glorot_normal',
        #                           regularizer=l2(0.000001),
        #                           trainable=True)
        self.we = self.add_weight(name='we',
                                  shape=(self.unit, 1),
                                  initializer='glorot_normal',
                                  regularizer=l2(0.000001),
                                  trainable=True)
        self.w4 = self.add_weight(name='w4',
                                  shape=(input_shape[0][-1], self.unit),
                                  initializer='glorot_normal',
                                  regularizer=l2(0.000001),
                                  trainable=True)

        # if self.bias:
        #     self.b1 = self.add_weight(name='b1',shape=(self.unit,),
        #                              initializer='zeros',
        #                               regularizer=l2(0.00001),
        #                               trainable=True)
        # self.bh = self.add_weight(name='bh',shape=None,
        #                          initializer='zeros',
        #                           trainable=True)
        super(CrossAttention, self).build(input_shape)  # 一定要在最后调用它

    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None

    def call(self, x, mask=None):
        assert isinstance(x, list)
        x1, x2 = x
        input_shape = K.shape(x1)
        pq = K.batch_dot(x1, x2, axes=2) #x1=(?,15,128) x2=(?,15,128) axes=2 表示128作为矩阵相乘相同的维度 因此pq=(?,15,15)
        pp = K.batch_dot(x1, x1, axes=2)
        qp = K.batch_dot(x2, x1, axes=2)
        qq = K.batch_dot(x2, x2, axes=2)
        # eij1 = K.dot(K.tanh(K.dot(pq, self.w2) + K.dot(x1, self.w4) + K.dot(x2, self.w1)), self.we) * self.sr # we=(?,16,1) eij1=(?,15,1)
        eij1 = K.dot(K.tanh(K.dot(pq, self.w2)),self.we)
        #----test
        # eij1 = K.tanh(K.dot(pq, self.w2))

        # eij1 = K.dot(K.tanh(K.dot(pq, self.w2)+ K.dot(x2, self.w1)), self.we)
        ai1 = K.exp(eij1) #ai1=(?,15,1)
        # if mask != None:
        #     ms1 = tf.cast(mask[0], 'float32')
        #     ms1 = K.reshape(ms1, (-1, input_shape[1], 1))
        #     ai1 = ai1 * ms1

        ww1 = ai1 / (K.sum(ai1, axis=1, keepdims=True) + K.epsilon())
        ot1 = x1 * ww1

        # e2 = (e1 - K.min(e2, axis=1, keepdims=True))/(K.max(e2, axis=1, keepdims=True)-K.min(e2, axis=1, keepdims=True)+ K.epsilon())
        eij2 = K.dot(K.tanh(K.dot(qp, self.w2) ), self.we)
        # eij2 = K.tanh(K.dot(qp, self.w2))
        # eij2 = K.dot(K.tanh(K.dot(pq, self.w2)+ K.dot(x1, self.w4)), self.we)
        ai2 = K.exp(eij2)
        # if mask != None:
        #     ms2 = tf.cast(mask[1], 'float32')
        #     ms2 = K.reshape(ms2, (-1, input_shape[1], 1))
        #     ai2 = ai2 * ms2
        ww2 = ai2 / (K.sum(ai2, axis=1, keepdims=True) + K.epsilon())
        ot2 = x2 * ww2

        return [ot1, ot2]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0], shape_a[1], shape_a[2]),(shape_a[0], shape_a[1], shape_a[2])]
        #return [(shape_a[0], shape_a[1], shape_a[2]), (shape_a[0], shape_a[1], shape_a[2]), (shape_a[0], shape_a[1], shape_a[1]), (shape_a[0], shape_a[1], shape_a[1])]
        # return [(shape_a[0],shape_a[1],1), (shape_b[0],shape_b[1],1)]

class MyAttention(Layer):
    def __init__(self ,bias=True ,sr=10 ,unit=16 ,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        super(MyAttention, self).__init__(**kwargs)
        self.bias =bias
        self.sr =sr
        self.unit =unit
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重
        self.w1 =self.add_weight(name='w1',
                                  shape=(input_shape[0][-1], self.unit),
                                  initializer='glorot_normal',
                                  regularizer=l2(0.000001),
                                  trainable=True)

        self.w2 = self.add_weight(name='w2',
                                  shape=(input_shape[0][1], self.unit),
                                  initializer='glorot_normal',
                                  regularizer=l2(0.000001),
                                  trainable=True)

        self.w3 = self.add_weight(name='w3',
                                  shape=(input_shape[0][1], self.unit),
                                  initializer='glorot_normal',
                                  regularizer=l2(0.000001),
                                  trainable=True)
        self.we = self.add_weight(name='we',
                                  shape=(self.unit, 1),
                                  initializer='glorot_normal',
                                  regularizer=l2(0.000001),
                                  trainable=True)
        # if self.bias:
        #     self.b1 = self.add_weight(name='b1',shape=(self.unit,),
        #                              initializer='zeros',
        #                               regularizer=l2(0.00001),
        #                               trainable=True)
        # self.bh = self.add_weight(name='bh',shape=None,
        #                          initializer='zeros',
        #                           trainable=True)
        super(MyAttention, self).build(input_shape)  # 一定要在最后调用它

    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None

    def call(self, x, mask=None):
        assert isinstance(x, list)
        x1, x2 = x
        input_shape = K.shape(x1)
        pq = K.batch_dot(x1, x2, axes=2) #x1=(?,15,128) x2=(?,15,128) axes=2 表示128作为矩阵相乘相同的维度 因此pq=(?,15,15)
        pp = K.batch_dot(x1, x1, axes=2)
        qp = K.batch_dot(x2, x1, axes=2)
        qq = K.batch_dot(x2, x2, axes=2)
        eij1 = K.dot(K.tanh(K.dot(pq, self.w2) + K.dot(pp, self.w3) + K.dot(x1, self.w1)), self.we) * self.sr # we=(?,16,1) eij1=(?,15,1)
        # k = K.expand_dims(K.dot(x2, self.w2)+self.b2, 1)
        ai1 = K.exp(eij1) #ai1=(?,15,1)
        # if mask != None:
        #     ms1 = tf.cast(mask[0], 'float32')
        #     ms1 = K.reshape(ms1, (-1, input_shape[1], 1))
        #     ai1 = ai1 * ms1

        ww1 = ai1 / (K.sum(ai1, axis=1, keepdims=True) + K.epsilon())

        ot1 = x1 * ww1
        # e2 = (e1 - K.min(e2, axis=1, keepdims=True))/(K.max(e2, axis=1, keepdims=True)-K.min(e2, axis=1, keepdims=True)+ K.epsilon())
        eij2 = K.dot(K.tanh(K.dot(qp, self.w2) + K.dot(qq, self.w3) + K.dot(x2, self.w1)), self.we) * self.sr
        # eij2 = *self.sr
        ai2 = K.exp(eij2)
        # if mask != None:
        #     ms2 = tf.cast(mask[1], 'float32')
        #     ms2 = K.reshape(ms2, (-1, input_shape[1], 1))
        #     ai2 = ai2 * ms2
        ww2 = ai2 / (K.sum(ai2, axis=1, keepdims=True) + K.epsilon())
        ot2 = x2 * ww2
        # return weighted_input.sum(axis=1)
        # oot1=K.sum(ot1,axis=1)
        # oot2=K.sum(ot2,axis=1)
        oot1 = K.sum(ot1, axis=1)
        oot2 = K.sum(ot2, axis=1)
        # return [ww1,ww2]
        return [ot1, ot2]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [ (shape_a[0], shape_a[1], shape_a[2]),(shape_b[0], shape_b[1], shape_b[2])]
        # return [(shape_a[0],shape_a[1],1), (shape_b[0],shape_b[1],1)]

class MaskLayer(Layer):
    def __init__(self,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        super(MaskLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        super(MaskLayer, self).build(input_shape)
    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return input_mask
    def call(self,x,mask=None):
        input_shape = K.shape(x)
        if mask!=None:
            ms=tf.cast(mask,'float32')
            ms=K.reshape(ms,(-1,input_shape[1],1)) #可以理解为reshape为(?,15,1）
            x=x*ms
        # return [ww1,ww2]
        return x
    def compute_output_shape(self, input_shape):
        return input_shape
        # return [(shape_a[0],shape_a[1],1), (shape_b[0],shape_b[1],1)]

class MaskLayer1(Layer): #允许传入新的mask 传入参数mymask
    def __init__(self,mymask,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        self.mymask = mymask
        super(MaskLayer1, self).__init__(**kwargs)
    def build(self, input_shape):
        super(MaskLayer1, self).build(input_shape)
    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return input_mask
    def call(self,x,mask=None):
        input_shape = K.shape(x)
        if mask!=None:
            # ms=tf.cast(mask,'float32') #先把bool类型转为float32 可能变成类似于[0,0,0,0,0,1,1,1,1,1,1,1]
            ms = tf.cast(self.mymask, 'float32')
            ms=K.reshape(ms,(-1,input_shape[1],1)) #可以理解为reshape为(?,15,1）
            x=x*ms
        # return [ww1,ww2]
        return [x,mask,ms]
    def compute_output_shape(self, input_shape):
        return [input_shape,(input_shape[0],input_shape[1]),(input_shape[0],input_shape[0],1)]
        # return [(shape_a[0],shape_a[1],1), (shape_b[0],shape_b[1],1)]
