
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.regularizers import l2
from keras.constraints import max_norm
from keras.layers import BatchNormalization
import tensorflow as tf

class Attention(Layer):
 
    def __init__(self,bias=True,sr=10,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        super(Attention, self).__init__(**kwargs)
        self.bias=bias
        self.sr=sr
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重

        # self.w1 =self.add_weight(name='w1',
        #                               shape=(input_shape[0][-1],1),
        #                               initializer='glorot_normal',
        #                                trainable=True)
                                      
        # self.w2 =self.add_weight(name='w2',
        #                               shape=(input_shape[1][-1],1),
        #                               initializer='glorot_normal',
        #                                trainable=True)
        # if self.bias:
        #     self.b1 = self.add_weight(name='b1',shape=None,
        #                              initializer='zeros',
        #                               trainable=True)
        #     self.b2 = self.add_weight(name='b2',shape=None,
        #                              initializer='zeros',
        #                               trainable=True)
        super(Attention, self).build(input_shape)  # 一定要在最后调用它
    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None
    def call(self,x,mask=None):
        assert isinstance(x, list)
        #isinstance() 函数来判断一个对象是否是一个已知的类型
        #assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常
        x1,x2=x

        dot1 = K.batch_dot(x1, x2, axes=2)
        dot2 = K.batch_dot(x1, x1, axes=2)
        dot3 = K.batch_dot(x2, x2, axes=2)
        # max_ = K.maximum(K.sqrt(dot2 * dot3), K.epsilon())
        e=dot1 / (K.sqrt(dot2 * dot3)+K.epsilon())

        eij1 = K.tanh(e)
        ai1 = K.exp(eij1)
        ww1 = ai1/(K.sum(ai1, axis=1,keepdims=True)+ K.epsilon())
        ot1 = K.sum(K.batch_dot(x2,ww1,axes=1),axis=1)
        ot1=K.expand_dims(ot1,axis=-1)

        eij2 = K.tanh(K.permute_dimensions(e, (0, 2, 1)))
        ai2 = K.exp(eij2)
        ww2 = ai1/(K.sum(ai2, axis=1,keepdims=True)+ K.epsilon())
        ot2 = K.sum(K.batch_dot(x1,ww2,axes=1),axis=1)
        ot2=K.expand_dims(ot2,axis=-1)
        # return weighted_input.sum(axis=1)
        # oot1=K.sum(ot1,axis=1)
        # oot2=K.sum(ot2,axis=1)
        # oot1=K.sum(ot1,axis=1)
        # oot2=K.sum(ot2,axis=1)
        # return [ww1,ww2]
        return [ot1,ot2]
 
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        # return [(shape_a[0],shape_a[1],shape_a[2]), (shape_b[0],shape_b[1],shape_b[2]),(shape_a[0],shape_a[2]), (shape_b[0],shape_b[2])]
        return [(shape_a[0],shape_a[1],1), (shape_b[0],shape_b[1],1)]

# import numpy as np
# x1=K.cast(np.random.randn(32,25,100),dtype=float)
# x2=K.cast(np.random.randn(32,25,100),dtype=float)

# dot1 = K.batch_dot(x1, x2, axes=2)
# dot2 = K.batch_dot(x1, x1, axes=2)
# dot3 = K.batch_dot(x2, x2, axes=2)
# # max_ = K.maximum(K.sqrt(dot2 * dot3), K.epsilon())
# e=dot1 / (K.sqrt(dot2 * dot3)+K.epsilon())

# eij1 = e
# ai1 = K.exp(eij1)
# ww1 = ai1/(K.sum(ai1, axis=1,keepdims=True)+ K.epsilon())
# ot1 = K.sum(K.batch_dot(x2,ww1,axes=1),axis=1)
# ot1=K.expand_dims(ot1,axis=-1)

# eij2 = K.permute_dimensions(e, (0, 2, 1))
# ai2 = K.exp(eij2)
# ww2 = ai1/(K.sum(ai2, axis=1,keepdims=True)+ K.epsilon())
# ot2 = K.sum(K.batch_dot(x1,ww2,axes=1),axis=1)
# ot2=K.expand_dims(ot2,axis=-1)

# xx1 = K.l2_normalize(x1, axis=-1)

# sss=K.batch_dot(s3,s1)
# with tf.Session() as sess:
#     print(max_[0][0].eval())
#     print(e[0][0].eval())
#     print(K.epsilon().eval())
#     print(d.eval())
#     print(d1.eval())