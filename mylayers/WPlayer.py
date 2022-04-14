from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.regularizers import l2
import tensorflow as tf


class WKS1(Layer):
    def __init__(self,bias=True,sr=10,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        super(WKS1, self).__init__(**kwargs)
        self.bias=bias
        self.sr=sr
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重
        self.w1 =self.add_weight(name='w1',
                                      shape=(input_shape[0][-1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
                                      
        self.w2 =self.add_weight(name='w2',
                                      shape=(input_shape[1][-1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        self.wp =self.add_weight(name='wp',
                                      shape=(input_shape[0][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
                                      
        self.wq =self.add_weight(name='wq',
                                      shape=(input_shape[1][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        self.wpq =self.add_weight(name='wpq',
                                      shape=(input_shape[0][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        self.wqp =self.add_weight(name='wqp',
                                      shape=(input_shape[1][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        super(WKS1, self).build(input_shape)  # 一定要在最后调用它
    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None
    def call(self,x,mask=None):
        assert isinstance(x, list)
        x1,x2=x
        input_shape = K.shape(x1)
        # input_len = input_shape[1]
        # q = K.expand_dims(K.dot(x1, self.w1)+self.b1, 2)
        pq=K.batch_dot(x1,x2,axes=2)
        pp=K.batch_dot(x1,x1,axes=2)
        qp=K.batch_dot(x2,x1,axes=2)
        qq=K.batch_dot(x2,x2,axes=2)
        eij1 = K.tanh(K.dot(pq,self.wpq)+K.dot(pp,self.wp)+K.dot(x1,self.w1))*self.sr
        # k = K.expand_dims(K.dot(x2, self.w2)+self.b2, 1)
        ai1 = K.exp(eij1)
        ai1 = K.exp(eij1)
        if mask!=None:
            ms1=tf.cast(mask[0],'float32')
            ms1=K.reshape(ms1,(-1,input_shape[1],1))
            ai1 = ai1*ms1
        ww1 = ai1/(K.sum(ai1, axis=1,keepdims=True)+ K.epsilon())
        ot1 = x1*ww1

        # e2 = (e1 - K.min(e2, axis=1, keepdims=True))/(K.max(e2, axis=1, keepdims=True)-K.min(e2, axis=1, keepdims=True)+ K.epsilon())
        eij2 = K.tanh(K.dot(qp,self.wqp)+K.dot(qq,self.wq)+K.dot(x2,self.w2))*self.sr
        ai2 = K.exp(eij2)
        if mask!=None:
            ms2=tf.cast(mask[1],'float32')
            ms2=K.reshape(ms2,(-1,input_shape[1],1))
            ai2 = ai2*ms2
        ww2 = ai2/(K.sum(ai2, axis=1,keepdims=True)+ K.epsilon())
        ot2 = x2*ww2
        # return weighted_input.sum(axis=1)
        # oot1=K.sum(ot1,axis=1)
        # oot2=K.sum(ot2,axis=1)
        oot1=K.sum(ot1,axis=1)
        oot2=K.sum(ot2,axis=1)
        # return [ww1,ww2]
        return [ww1,ww2,ot1,ot2,oot1,oot2]
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0],shape_a[1],1), (shape_b[0],shape_b[1],1),(shape_a[0],shape_a[1],shape_a[2]), (shape_b[0],shape_b[1],shape_b[2]),(shape_a[0],shape_a[2]), (shape_b[0],shape_b[2])]
        # return [(shape_a[0],shape_a[1],1), (shape_b[0],shape_b[1],1)]


# class WKS(Layer):
#     def __init__(self,bias=True,sr=10,**kwargs):
#         # self.output_dim = output_dim
#         self.supports_masking = True
#         super(WKS, self).__init__(**kwargs)
#         self.bias=bias
#         self.sr=sr
#     def build(self, input_shape):
#         assert isinstance(input_shape, list)
#         # 为该层创建一个可训练的权重
#         self.w1 =self.add_weight(name='w1',
#                                       shape=(input_shape[0][-1],1),
#                                       initializer='glorot_normal',
#                                       regularizer=l2(0.00001), 
#                                        trainable=True)

#         self.w2 =self.add_weight(name='w2',
#                                       shape=(input_shape[0][1],1),
#                                       initializer='glorot_normal',
#                                       regularizer=l2(0.00001), 
#                                        trainable=True)

#         self.w3 =self.add_weight(name='w3',
#                                       shape=(input_shape[0][1],1),
#                                       initializer='glorot_normal',
#                                       regularizer=l2(0.00001), 
#                                        trainable=True)
#             # self.bh = self.add_weight(name='bh',shape=None,
#             #                          initializer='zeros',
#             #                           trainable=True)
#         super(WKS, self).build(input_shape)  # 一定要在最后调用它
#     def compute_mask(self, input, input_mask=None):
#         # need not to pass the mask to next layers
#         return None
#     def call(self,x,mask=None):
#         assert isinstance(x, list)
#         x1,x2=x
#         input_shape = K.shape(x1)
#         pq=K.batch_dot(x1,x2,axes=2)
#         pp=K.batch_dot(x1,x1,axes=2)
#         qp=K.batch_dot(x2,x1,axes=2)
#         qq=K.batch_dot(x2,x2,axes=2)
#         eij1 = K.tanh(K.dot(pq,self.w2)+K.dot(pp,self.w3)+K.dot(x1,self.w1))*self.sr
#         # k = K.expand_dims(K.dot(x2, self.w2)+self.b2, 1)
#         ai1 = K.exp(eij1)
#         if mask!=None:
#             ms1=tf.cast(mask[0],'float32')
#             ms1=K.reshape(ms1,(-1,input_shape[1],1))
#             ai1 = ai1*ms1
        
#         ww1 = ai1/(K.sum(ai1, axis=1,keepdims=True)+ K.epsilon())
        
#         ot1 = x1*ww1
#         # e2 = (e1 - K.min(e2, axis=1, keepdims=True))/(K.max(e2, axis=1, keepdims=True)-K.min(e2, axis=1, keepdims=True)+ K.epsilon())
#         eij2 = K.tanh(K.dot(qp,self.w2)+K.dot(qq,self.w3)+K.dot(x2,self.w1))*self.sr
#         ai2 = K.exp(eij2)
#         if mask!=None:
#             ms2=tf.cast(mask[1],'float32')
#             ms2=K.reshape(ms2,(-1,input_shape[1],1))
#             ai2 = ai2*ms2
#         ww2 = ai2/(K.sum(ai2, axis=1,keepdims=True)+ K.epsilon())
#         ot2 = x2*ww2
#         # return weighted_input.sum(axis=1)
#         # oot1=K.sum(ot1,axis=1)
#         # oot2=K.sum(ot2,axis=1)
        
#         oot1=K.sum(ot1,axis=1)
#         oot2=K.sum(ot2,axis=1)
#         # return [ww1,ww2]
#         return [ww1,ww2,ot1,ot2,oot1,oot2]
#     def compute_output_shape(self, input_shape):
#         assert isinstance(input_shape, list)
#         shape_a, shape_b = input_shape
#         return [(shape_a[0],shape_a[1],1), (shape_b[0],shape_b[1],1),(shape_a[0],shape_a[1],shape_a[2]), (shape_b[0],shape_b[1],shape_b[2]),(shape_a[0],shape_a[2]), (shape_b[0],shape_b[2])]
#         # return [(shape_a[0],shape_a[1],1), (shape_b[0],shape_b[1],1)]


class WKS(Layer):
    def __init__(self,bias=True,sr=10,unit=16,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        super(WKS, self).__init__(**kwargs)
        self.bias=bias
        self.sr=sr
        self.unit=unit
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重
        self.w1 =self.add_weight(name='w1',
                                      shape=(input_shape[0][-1],self.unit),
                                      initializer='glorot_normal',
                                      regularizer=l2(0.000001), 
                                       trainable=True)

        self.w2 =self.add_weight(name='w2',
                                      shape=(input_shape[0][1],self.unit),
                                      initializer='glorot_normal',
                                      regularizer=l2(0.000001), 
                                       trainable=True)

        self.w3 =self.add_weight(name='w3',
                                      shape=(input_shape[0][1],self.unit),
                                      initializer='glorot_normal',
                                      regularizer=l2(0.000001), 
                                       trainable=True)
        self.we =self.add_weight(name='we',
                                      shape=(self.unit,1),
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
    def call(self,x,mask=None):
        assert isinstance(x, list)
        x1,x2=x
        input_shape = K.shape(x1)
        pq=K.batch_dot(x1,x2,axes=2)
        pp=K.batch_dot(x1,x1,axes=2)
        qp=K.batch_dot(x2,x1,axes=2)
        qq=K.batch_dot(x2,x2,axes=2)
        eij1 = K.dot(K.tanh(K.dot(pq,self.w2)+K.dot(pp,self.w3)+K.dot(x1,self.w1)),self.we)*self.sr
        # k = K.expand_dims(K.dot(x2, self.w2)+self.b2, 1)
        ai1 = K.exp(eij1)
        if mask!=None:
            ms1=tf.cast(mask[0],'float32')
            ms1=K.reshape(ms1,(-1,input_shape[1],1))
            ai1 = ai1*ms1
        
        ww1 = ai1/(K.sum(ai1, axis=1,keepdims=True)+ K.epsilon())
        
        ot1 = x1*ww1
        # e2 = (e1 - K.min(e2, axis=1, keepdims=True))/(K.max(e2, axis=1, keepdims=True)-K.min(e2, axis=1, keepdims=True)+ K.epsilon())
        eij2 = K.dot(K.tanh(K.dot(qp,self.w2)+K.dot(qq,self.w3)+K.dot(x2,self.w1)),self.we)*self.sr
        # eij2 = *self.sr
        ai2 = K.exp(eij2)
        if mask!=None:
            ms2=tf.cast(mask[1],'float32')
            ms2=K.reshape(ms2,(-1,input_shape[1],1))
            ai2 = ai2*ms2
        ww2 = ai2/(K.sum(ai2, axis=1,keepdims=True)+ K.epsilon())
        ot2 = x2*ww2
        # return weighted_input.sum(axis=1)
        # oot1=K.sum(ot1,axis=1)
        # oot2=K.sum(ot2,axis=1)
        
        oot1=K.sum(ot1,axis=1)
        oot2=K.sum(ot2,axis=1)
        # return [ww1,ww2]
        return [ww1,ww2,ot1,ot2,oot1,oot2]
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0],shape_a[1],1), (shape_b[0],shape_b[1],1),(shape_a[0],shape_a[1],shape_a[2]), (shape_b[0],shape_b[1],shape_b[2]),(shape_a[0],shape_a[2]), (shape_b[0],shape_b[2])]
        # return [(shape_a[0],shape_a[1],1), (shape_b[0],shape_b[1],1)]


class WPE7_ks3(Layer):
    def __init__(self,bias=True,sr=10,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        super(WPE7_ks3, self).__init__(**kwargs)
        self.bias=bias
        self.sr=sr
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重
        self.w1 =self.add_weight(name='w1',
                                      shape=(input_shape[0][-1],1),
                                      initializer='glorot_normal',
                                       trainable=True)

        self.w2 =self.add_weight(name='w2',
                                      shape=(input_shape[0][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)

        self.w3 =self.add_weight(name='w3',
                                      shape=(input_shape[0][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        if self.bias:
            self.b1 = self.add_weight(name='b1',shape=None,
                                     initializer='zeros',
                                      trainable=True)
            # self.bh = self.add_weight(name='bh',shape=None,
            #                          initializer='zeros',
            #                           trainable=True)
        super(WPE7_ks3, self).build(input_shape)  # 一定要在最后调用它
    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None
    def call(self,x,mask=None):
        assert isinstance(x, list)
        x1,x2=x
        input_shape = K.shape(x1)
        pq=K.batch_dot(x1,x2,axes=2)
        pp=K.batch_dot(x1,x1,axes=2)
        qp=K.batch_dot(x2,x1,axes=2)
        qq=K.batch_dot(x2,x2,axes=2)
        eij1 = K.tanh(K.dot(pq,self.w2)+K.dot(pp,self.w3)+K.dot(x1,self.w1)+self.b1)*self.sr
        # k = K.expand_dims(K.dot(x2, self.w2)+self.b2, 1)
        ai1 = K.exp(eij1)
        if mask!=None:
            ms1=tf.cast(mask[0],'float32')
            ms1=K.reshape(ms1,(-1,input_shape[1],1))
            ai1 = ai1*ms1
        
        ww1 = ai1/(K.sum(ai1, axis=1,keepdims=True)+ K.epsilon())
        
        ot1 = x1*ww1
        # e2 = (e1 - K.min(e2, axis=1, keepdims=True))/(K.max(e2, axis=1, keepdims=True)-K.min(e2, axis=1, keepdims=True)+ K.epsilon())
        eij2 = K.tanh(K.dot(qp,self.w2)+K.dot(qq,self.w3)+K.dot(x2,self.w1)+self.b1)*self.sr
        ai2 = K.exp(eij2)
        if mask!=None:
            ms2=tf.cast(mask[1],'float32')
            ms2=K.reshape(ms2,(-1,input_shape[1],1))
            ai2 = ai2*ms2
        ww2 = ai2/(K.sum(ai2, axis=1,keepdims=True)+ K.epsilon())
        ot2 = x2*ww2
        # return weighted_input.sum(axis=1)
        # oot1=K.sum(ot1,axis=1)
        # oot2=K.sum(ot2,axis=1)
        
        oot1=K.sum(ot1,axis=1)
        oot2=K.sum(ot2,axis=1)
        # return [ww1,ww2]
        return [ww1,ww2,ot1,ot2,oot1,oot2]
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0],shape_a[1],1), (shape_b[0],shape_b[1],1),(shape_a[0],shape_a[1],shape_a[2]), (shape_b[0],shape_b[1],shape_b[2]),(shape_a[0],shape_a[2]), (shape_b[0],shape_b[2])]
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


class WPE7_ks2(Layer):
    def __init__(self,bias=True,sr=10,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        super(WPE7_ks2, self).__init__(**kwargs)
        self.bias=bias
        self.sr=sr
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重
        self.w1 =self.add_weight(name='w1',
                                      shape=(input_shape[0][-1],1),
                                      initializer='glorot_normal',
                                       trainable=True)

        self.w2 =self.add_weight(name='w2',
                                      shape=(input_shape[0][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)

        self.w3 =self.add_weight(name='w3',
                                      shape=(input_shape[0][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        if self.bias:
            self.b1 = self.add_weight(name='b1',shape=None,
                                     initializer='zeros',
                                      trainable=True)
            # self.bh = self.add_weight(name='bh',shape=None,
            #                          initializer='zeros',
            #                           trainable=True)
        super(WPE7_ks2, self).build(input_shape)  # 一定要在最后调用它
    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None
    def call(self,x,mask=None):
        assert isinstance(x, list)
        x1,x2=x
        input_shape = K.shape(x1)
        
            # x1=x1*ms1
            # x2=x2*ms2
        # input_len = input_shape[1]
        # q = K.expand_dims(K.dot(x1, self.w1)+self.b1, 2)
        pq=K.batch_dot(x1,x2,axes=2)
        pp=K.batch_dot(x1,x1,axes=2)
        qp=K.batch_dot(x2,x1,axes=2)
        qq=K.batch_dot(x2,x2,axes=2)
        eij1 = K.tanh(K.dot(pq,self.w2)+K.dot(pp,self.w3)+K.dot(x1,self.w1)+self.b1)*self.sr
        # k = K.expand_dims(K.dot(x2, self.w2)+self.b2, 1)
        ai1 = K.exp(eij1)
        if mask!=None:
            ms1=tf.cast(mask[0],'float32')
            ms1=K.reshape(ms1,(-1,input_shape[1],1))
            ai1 = ai1*ms1
        
        ww1 = ai1/(K.sum(ai1, axis=1,keepdims=True)+ K.epsilon())
        
        ot1 = x1*ww1
        # e2 = (e1 - K.min(e2, axis=1, keepdims=True))/(K.max(e2, axis=1, keepdims=True)-K.min(e2, axis=1, keepdims=True)+ K.epsilon())
        eij2 = K.tanh(K.dot(qp,self.w2)+K.dot(qq,self.w3)+K.dot(x2,self.w1)+self.b1)*self.sr
        ai2 = K.exp(eij2)
        ww2 = ai2/(K.sum(ai2, axis=1,keepdims=True)+ K.epsilon())
        if mask!=None:
            ms2=tf.cast(mask[1],'float32')
            ms2=K.reshape(ms2,(-1,input_shape[1],1))
            ai2 = ai2*ms2
        
        ot2 = x2*ww2
        # return weighted_input.sum(axis=1)
        # oot1=K.sum(ot1,axis=1)
        # oot2=K.sum(ot2,axis=1)
        
        oot1=K.sum(ot1,axis=1)
        oot2=K.sum(ot2,axis=1)
        # return [ww1,ww2]
        return [ww1,ww2,ot1,ot2,oot1,oot2]
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0],shape_a[1],1), (shape_b[0],shape_b[1],1),(shape_a[0],shape_a[1],shape_a[2]), (shape_b[0],shape_b[1],shape_b[2]),(shape_a[0],shape_a[2]), (shape_b[0],shape_b[2])]
        # return [(shape_a[0],shape_a[1],1), (shape_b[0],shape_b[1],1)]

class WPE7_ks1(Layer):
    def __init__(self,bias=True,sr=10,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        super(WPE7_ks1, self).__init__(**kwargs)
        self.bias=bias
        self.sr=sr
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重
        self.w1 =self.add_weight(name='w1',
                                      shape=(input_shape[0][-1],1),
                                      initializer='glorot_normal',
                                       trainable=True)

        self.w2 =self.add_weight(name='w2',
                                      shape=(input_shape[0][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)

        self.w3 =self.add_weight(name='w3',
                                      shape=(input_shape[0][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        if self.bias:
            self.b1 = self.add_weight(name='b1',shape=None,
                                     initializer='zeros',
                                      trainable=True)
            # self.bh = self.add_weight(name='bh',shape=None,
            #                          initializer='zeros',
            #                           trainable=True)
        super(WPE7_ks1, self).build(input_shape)  # 一定要在最后调用它
    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None
    def call(self,x,mask=None):
        assert isinstance(x, list)
        x1,x2=x
        input_shape = K.shape(x1)
        if mask!=None:
            ms1=tf.cast(mask[0],'float32')
            ms1=K.reshape(ms1,(-1,input_shape[1],1))
            ms2=tf.cast(mask[1],'float32')
            ms2=K.reshape(ms2,(-1,input_shape[1],1))
            x1=x1*ms1
            x2=x2*ms2
        # input_len = input_shape[1]
        # q = K.expand_dims(K.dot(x1, self.w1)+self.b1, 2)
        pq=K.batch_dot(x1,x2,axes=2)
        pp=K.batch_dot(x1,x1,axes=2)
        qp=K.batch_dot(x2,x1,axes=2)
        qq=K.batch_dot(x2,x2,axes=2)
        eij1 = K.tanh(K.dot(pq,self.w2)+K.dot(pp,self.w3)+K.dot(x1,self.w1)+self.b1)*self.sr
        # k = K.expand_dims(K.dot(x2, self.w2)+self.b2, 1)
        ai1 = K.exp(eij1)
        ww1 = ai1/(K.sum(ai1, axis=1,keepdims=True)+ K.epsilon())
        
        ot1 = x1*ww1
        # e2 = (e1 - K.min(e2, axis=1, keepdims=True))/(K.max(e2, axis=1, keepdims=True)-K.min(e2, axis=1, keepdims=True)+ K.epsilon())
        eij2 = K.tanh(K.dot(qp,self.w2)+K.dot(qq,self.w3)+K.dot(x2,self.w1)+self.b1)*self.sr
        ai2 = K.exp(eij2)
        ww2 = ai2/(K.sum(ai2, axis=1,keepdims=True)+ K.epsilon())
        ot2 = x2*ww2
        # return weighted_input.sum(axis=1)
        # oot1=K.sum(ot1,axis=1)
        # oot2=K.sum(ot2,axis=1)
        
        oot1=K.sum(ot1,axis=1)
        oot2=K.sum(ot2,axis=1)
        # return [ww1,ww2]
        return [ww1,ww2,ot1,ot2,oot1,oot2]
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0],shape_a[1],1), (shape_b[0],shape_b[1],1),(shape_a[0],shape_a[1],shape_a[2]), (shape_b[0],shape_b[1],shape_b[2]),(shape_a[0],shape_a[2]), (shape_b[0],shape_b[2])]
        # return [(shape_a[0],shape_a[1],1), (shape_b[0],shape_b[1],1)]

class WPE7_ks(Layer):
    def __init__(self,bias=True,sr=10,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        super(WPE7_ks, self).__init__(**kwargs)
        self.bias=bias
        self.sr=sr
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重
        self.w1 =self.add_weight(name='w1',
                                      shape=(input_shape[0][-1],1),
                                      initializer='glorot_normal',
                                       trainable=True)

        self.w2 =self.add_weight(name='w2',
                                      shape=(input_shape[0][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)

        self.w3 =self.add_weight(name='w3',
                                      shape=(input_shape[0][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        if self.bias:
            self.b1 = self.add_weight(name='b1',shape=None,
                                     initializer='zeros',
                                      trainable=True)
            # self.bh = self.add_weight(name='bh',shape=None,
            #                          initializer='zeros',
            #                           trainable=True)
        super(WPE7_ks, self).build(input_shape)  # 一定要在最后调用它
    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None
    def call(self,x,mask=None):
        assert isinstance(x, list)
        x1,x2=x
        input_shape = K.shape(x1)
        # input_len = input_shape[1]
        # q = K.expand_dims(K.dot(x1, self.w1)+self.b1, 2)
        pq=K.batch_dot(x1,x2,axes=2)
        pp=K.batch_dot(x1,x1,axes=2)
        qp=K.batch_dot(x2,x1,axes=2)
        qq=K.batch_dot(x2,x2,axes=2)
        eij1 = K.tanh(K.dot(pq,self.w2)+K.dot(pp,self.w3)+K.dot(x1,self.w1)+self.b1)*self.sr
        # k = K.expand_dims(K.dot(x2, self.w2)+self.b2, 1)
        ai1 = K.exp(eij1)
        ww1 = ai1/(K.sum(ai1, axis=1,keepdims=True)+ K.epsilon())
        if mask!=None:
            ms1=tf.cast(mask[0],'float32')
            ms1=K.reshape(ms1,(-1,input_shape[1],1))
            ww1=ww1*ms1
        ot1 = x1*ww1
        # e2 = (e1 - K.min(e2, axis=1, keepdims=True))/(K.max(e2, axis=1, keepdims=True)-K.min(e2, axis=1, keepdims=True)+ K.epsilon())
        eij2 = K.tanh(K.dot(qp,self.w2)+K.dot(qq,self.w3)+K.dot(x2,self.w1)+self.b1)*self.sr
        ai2 = K.exp(eij2)
        ww2 = ai2/(K.sum(ai2, axis=1,keepdims=True)+ K.epsilon())
        if mask!=None:
            ms2=tf.cast(mask[1],'float32')
            ms2=K.reshape(ms2,(-1,input_shape[1],1))
            ww2=ww1*ms2
        ot2 = x2*ww2
        # return weighted_input.sum(axis=1)
        # oot1=K.sum(ot1,axis=1)
        # oot2=K.sum(ot2,axis=1)
        
        oot1=K.sum(ot1,axis=1)
        oot2=K.sum(ot2,axis=1)
        # return [ww1,ww2]
        return [ww1,ww2,ot1,ot2,oot1,oot2]
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0],shape_a[1],1), (shape_b[0],shape_b[1],1),(shape_a[0],shape_a[1],shape_a[2]), (shape_b[0],shape_b[1],shape_b[2]),(shape_a[0],shape_a[2]), (shape_b[0],shape_b[2])]
        # return [(shape_a[0],shape_a[1],1), (shape_b[0],shape_b[1],1)]


# class FindWordPair_1(Layer):
 
#     def __init__(self,bias=True,**kwargs):
#         # self.output_dim = output_dim
#         self.supports_masking = True
#         super(FindWordPair_1, self).__init__(**kwargs)
#         self.bias=bias
#     def build(self, input_shape):
#         assert isinstance(input_shape, list)
#         # 为该层创建一个可训练的权重

#         self.w1 =self.add_weight(name='w1',
#                                       shape=(input_shape[0][-1],1),
#                                       initializer='glorot_normal',
#                                       trainable=True)
                                      
#         self.w2 =self.add_weight(name='w2',
#                                       shape=(input_shape[1][-1],1),
#                                       initializer='glorot_normal',
#                                       trainable=True)
#         self.w3 =self.add_weight(name='w3',
#                                       shape=(input_shape[0][-1],1),
#                                       initializer='glorot_normal',
#                                       trainable=True)
                                      
#         self.w4 =self.add_weight(name='w4',
#                                       shape=(input_shape[1][-1],1),
#                                       initializer='glorot_normal',
#                                       trainable=True)
#         if self.bias:
#             self.b1 = self.add_weight(name='b1',shape=None,
#                                      initializer='zeros',
#                                       trainable=True)
#             self.b2 = self.add_weight(name='b2',shape=None,
#                                      initializer='zeros',
#                                       trainable=True)
#         super(FindWordPair_1, self).build(input_shape)  # 一定要在最后调用它
#     def compute_mask(self, input, input_mask=None):
#         # need not to pass the mask to next layers
#         return None
#     def call(self,x,mask=None):
#         assert isinstance(x, list)
#         x1,x2=x
#         eij1 = K.tanh(K.dot(x1*x2,self.w1)+K.dot(x1,self.w3)+self.b1)*10
#         ai1 = K.exp(eij1)
#         ww1 = ai1/(K.sum(ai1, axis=1,keepdims=True)+ K.epsilon())
#         ot1 = x1*ww1

#         eij2 = K.tanh(K.dot(x2*x1,self.w2)+K.dot(x2,self.w4)+self.b2)*10
#         ai2 = K.exp(eij2)
#         ww2 = ai2/(K.sum(ai2, axis=1,keepdims=True)+ K.epsilon())
#         ot2 = x2*ww2

#         oot1=K.sum(ot1,axis=1)
#         oot2=K.sum(ot2,axis=1)
#         # return [ww1,ww2]
#         return [ot1,ot2,oot1,oot2]
#     def compute_output_shape(self, input_shape):
#         assert isinstance(input_shape, list)
#         shape_a, shape_b = input_shape
#         return [(shape_a[0],shape_a[1],shape_a[2]), (shape_b[0],shape_b[1],shape_b[2]),(shape_a[0],shape_a[2]), (shape_b[0],shape_b[2])]
#         # return [(shape_a[0],shape_a[1],1), (shape_b[0],shape_b[1],1)]

class WPE7(Layer):
    def __init__(self,bias=True,sr=10,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        super(WPE7, self).__init__(**kwargs)
        self.bias=bias
        self.sr=sr
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重
        self.w1 =self.add_weight(name='w1',
                                      shape=(input_shape[0][-1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
                                      
        self.w2 =self.add_weight(name='w2',
                                      shape=(input_shape[1][-1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        self.wp =self.add_weight(name='wp',
                                      shape=(input_shape[0][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
                                      
        self.wq =self.add_weight(name='wq',
                                      shape=(input_shape[1][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        self.wpq =self.add_weight(name='wpq',
                                      shape=(input_shape[0][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        self.wqp =self.add_weight(name='wqp',
                                      shape=(input_shape[1][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        if self.bias:
            self.b1 = self.add_weight(name='b1',shape=None,
                                     initializer='zeros',
                                      trainable=True)
            self.b2 = self.add_weight(name='b2',shape=None,
                                     initializer='zeros',
                                      trainable=True)
            # self.bh = self.add_weight(name='bh',shape=None,
            #                          initializer='zeros',
            #                           trainable=True)
        super(WPE7, self).build(input_shape)  # 一定要在最后调用它
    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None
    def call(self,x,mask=None):
        assert isinstance(x, list)
        x1,x2=x
        input_shape = K.shape(x1)
        # input_len = input_shape[1]
        # q = K.expand_dims(K.dot(x1, self.w1)+self.b1, 2)
        pq=K.batch_dot(x1,x2,axes=2)
        pp=K.batch_dot(x1,x1,axes=2)
        qp=K.batch_dot(x2,x1,axes=2)
        qq=K.batch_dot(x2,x2,axes=2)
        eij1 = K.tanh(K.dot(pq,self.wpq)+K.dot(pp,self.wp)+K.dot(x1,self.w1)+self.b1)*self.sr
        # k = K.expand_dims(K.dot(x2, self.w2)+self.b2, 1)
        ai1 = K.exp(eij1)
        ww1 = ai1/(K.sum(ai1, axis=1,keepdims=True)+ K.epsilon())
        ot1 = x1*ww1

        # e2 = (e1 - K.min(e2, axis=1, keepdims=True))/(K.max(e2, axis=1, keepdims=True)-K.min(e2, axis=1, keepdims=True)+ K.epsilon())
        eij2 = K.tanh(K.dot(qp,self.wqp)+K.dot(qq,self.wq)+K.dot(x2,self.w2)+self.b2)*self.sr
        ai2 = K.exp(eij2)
        ww2 = ai2/(K.sum(ai2, axis=1,keepdims=True)+ K.epsilon())
        ot2 = x2*ww2
        # return weighted_input.sum(axis=1)
        # oot1=K.sum(ot1,axis=1)
        # oot2=K.sum(ot2,axis=1)
        oot1=K.sum(ot1,axis=1)
        oot2=K.sum(ot2,axis=1)
        # return [ww1,ww2]
        return [ot1,ot2,oot1,oot2]
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0],shape_a[1],shape_a[2]), (shape_b[0],shape_b[1],shape_b[2]),(shape_a[0],shape_a[2]), (shape_b[0],shape_b[2])]
        # return [(shape_a[0],shape_a[1],1), (shape_b[0],shape_b[1],1)]


class WPE7_s1(Layer):
    def __init__(self,bias=True,sr=10,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        super(WPE7_s1, self).__init__(**kwargs)
        self.bias=bias
        self.sr=sr
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重
        self.w1 =self.add_weight(name='w1',
                                      shape=(input_shape[0][-1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        self.w2 =self.add_weight(name='w2',
                                      shape=(input_shape[0][-1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        self.w =self.add_weight(name='w',
                                      shape=(input_shape[0][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)


        self.we =self.add_weight(name='we',
                                      shape=(input_shape[1][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        if self.bias:
            self.b1 = self.add_weight(name='b1',shape=None,
                                     initializer='zeros',
                                      trainable=True)
            self.b2 = self.add_weight(name='b2',shape=None,
                                     initializer='zeros',
                                      trainable=True)
            # self.bh = self.add_weight(name='bh',shape=None,
            #                          initializer='zeros',
            #                           trainable=True)
        super(WPE7_s1, self).build(input_shape)  # 一定要在最后调用它
    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None
    def call(self,x,mask=None):
        assert isinstance(x, list)
        x1,x2=x
        input_shape = K.shape(x1)
        # input_len = input_shape[1]
        # q = K.expand_dims(K.dot(x1, self.w1)+self.b1, 2)
        pq=K.batch_dot(x1,x2,axes=2)
        pp=K.batch_dot(x1,x1,axes=2)
        qp=K.batch_dot(x2,x1,axes=2)
        qq=K.batch_dot(x2,x2,axes=2)
        eij1 = K.tanh(K.dot(pq,self.we)+K.dot(pp,self.w)+K.dot(x1,self.w1)+self.b1)*self.sr
        # k = K.expand_dims(K.dot(x2, self.w2)+self.b2, 1)
        ai1 = K.exp(eij1)
        ww1 = ai1/(K.sum(ai1, axis=1,keepdims=True)+ K.epsilon())
        ot1 = x1*ww1

        # e2 = (e1 - K.min(e2, axis=1, keepdims=True))/(K.max(e2, axis=1, keepdims=True)-K.min(e2, axis=1, keepdims=True)+ K.epsilon())
        eij2 = K.tanh(K.dot(qp,self.we)+K.dot(qq,self.w)+K.dot(x2,self.w2)+self.b2)*self.sr
        ai2 = K.exp(eij2)
        ww2 = ai2/(K.sum(ai2, axis=1,keepdims=True)+ K.epsilon())
        ot2 = x2*ww2
        # return weighted_input.sum(axis=1)
        # oot1=K.sum(ot1,axis=1)
        # oot2=K.sum(ot2,axis=1)
        oot1=K.sum(ot1,axis=1)
        oot2=K.sum(ot2,axis=1)
        # return [ww1,ww2]
        return [ot1,ot2,oot1,oot2]
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0],shape_a[1],shape_a[2]), (shape_b[0],shape_b[1],shape_b[2]),(shape_a[0],shape_a[2]), (shape_b[0],shape_b[2])]
        # return [(shape_a[0],shape_a[1],1), (shape_b[0],shape_b[1],1)]


class WPE7_s(Layer):
    def __init__(self,bias=True,sr=10,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        super(WPE7_s, self).__init__(**kwargs)
        self.bias=bias
        self.sr=sr
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重
        self.w1 =self.add_weight(name='w1',
                                      shape=(input_shape[0][-1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        self.w =self.add_weight(name='w',
                                      shape=(input_shape[0][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)


        self.we =self.add_weight(name='we',
                                      shape=(input_shape[1][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        if self.bias:
            self.b1 = self.add_weight(name='b1',shape=None,
                                     initializer='zeros',
                                      trainable=True)
            # self.bh = self.add_weight(name='bh',shape=None,
            #                          initializer='zeros',
            #                           trainable=True)
        super(WPE7_s, self).build(input_shape)  # 一定要在最后调用它
    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None
    def call(self,x,mask=None):
        assert isinstance(x, list)
        x1,x2=x
        input_shape = K.shape(x1)
        # input_len = input_shape[1]
        # q = K.expand_dims(K.dot(x1, self.w1)+self.b1, 2)
        pq=K.batch_dot(x1,x2,axes=2)
        pp=K.batch_dot(x1,x1,axes=2)
        qp=K.batch_dot(x2,x1,axes=2)
        qq=K.batch_dot(x2,x2,axes=2)
        eij1 = K.tanh(K.dot(pq,self.we)+K.dot(pp,self.w)+K.dot(x1,self.w1)+self.b1)*self.sr
        # k = K.expand_dims(K.dot(x2, self.w2)+self.b2, 1)
        ai1 = K.exp(eij1)
        ww1 = ai1/(K.sum(ai1, axis=1,keepdims=True)+ K.epsilon())
        ot1 = x1*ww1

        # e2 = (e1 - K.min(e2, axis=1, keepdims=True))/(K.max(e2, axis=1, keepdims=True)-K.min(e2, axis=1, keepdims=True)+ K.epsilon())
        eij2 = K.tanh(K.dot(qp,self.we)+K.dot(qq,self.w)+K.dot(x2,self.w1)+self.b1)*self.sr
        ai2 = K.exp(eij2)
        ww2 = ai2/(K.sum(ai2, axis=1,keepdims=True)+ K.epsilon())
        ot2 = x2*ww2
        # return weighted_input.sum(axis=1)
        # oot1=K.sum(ot1,axis=1)
        # oot2=K.sum(ot2,axis=1)
        oot1=K.sum(ot1,axis=1)
        oot2=K.sum(ot2,axis=1)
        # return [ww1,ww2]
        return [ot1,ot2,oot1,oot2]
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0],shape_a[1],shape_a[2]), (shape_b[0],shape_b[1],shape_b[2]),(shape_a[0],shape_a[2]), (shape_b[0],shape_b[2])]
        # return [(shape_a[0],shape_a[1],1), (shape_b[0],shape_b[1],1)]



class WPE6(Layer):
 
    def __init__(self,bias=True,sr=10,unit=10,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        super(WPE6, self).__init__(**kwargs)
        self.bias=bias       
        self.sr=sr
        self.unit=unit
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重

        self.w1 =self.add_weight(name='w1',
                                      shape=(input_shape[0][-1],self.unit),
                                      initializer='glorot_normal',
                                       trainable=True)
                                      
        self.w2 =self.add_weight(name='w2',
                                      shape=(input_shape[1][-1],self.unit),
                                      initializer='glorot_normal',
                                       trainable=True)
        self.w3 =self.add_weight(name='w3',
                                      shape=(input_shape[0][-1],self.unit),
                                      initializer='glorot_normal',
                                       trainable=True)
                                      
        self.w4 =self.add_weight(name='w4',
                                      shape=(input_shape[1][-1],self.unit),
                                      initializer='glorot_normal',
                                       trainable=True)
        self.we1 =self.add_weight(name='we1',
                                      shape=(self.unit,1),
                                      initializer='glorot_normal',
                                       trainable=True)
        self.we2 =self.add_weight(name='we1',
                                      shape=(self.unit,1),
                                      initializer='glorot_normal',
                                       trainable=True)
        if self.bias:
            self.b1 = self.add_weight(name='b1',shape=None,
                                     initializer='zeros',
                                      trainable=True)
            self.b2 = self.add_weight(name='b2',shape=None,
                                     initializer='zeros',
                                      trainable=True)
        super(WPE6, self).build(input_shape)  # 一定要在最后调用它
    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None
    def call(self,x,mask=None):
        assert isinstance(x, list)
        x1,x2=x
        eij1 = K.dot(K.tanh(K.dot(x1,self.w1)+K.dot(x2,self.w2)+self.b1),self.we1)*self.sr
        ai1 = K.exp(eij1)
        ww1 = ai1/(K.sum(ai1, axis=1,keepdims=True)+ K.epsilon())
        ot1 = x1*ww1

        eij2 = K.dot(K.tanh(K.dot(x1,self.w3)+K.dot(x2,self.w4)+self.b2),self.we2)*self.sr
        ai2 = K.exp(eij2)
        ww2 = ai2/(K.sum(ai2, axis=1,keepdims=True)+ K.epsilon())
        ot2 = x2*ww2

        oot1=K.sum(ot1,axis=1)
        oot2=K.sum(ot2,axis=1)
        # return [ww1,ww2]
        return [ot1,ot2,oot1,oot2]
 
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0],shape_a[1],shape_a[2]), (shape_b[0],shape_b[1],shape_b[2]),(shape_a[0],shape_a[2]), (shape_b[0],shape_b[2])]
        # return [(shape_a[0],shape_a[1],shape_a[1]), (shape_b[0],shape_b[1],1)]

class WPE5(Layer):
    def __init__(self,bias=True,sr=10,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        super(WPE5, self).__init__(**kwargs)
        self.bias=bias
        self.sr=sr
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重

        # self.ws =self.add_weight(name='ws',
        #                               shape=(input_shape[0][-1],input_shape[0][-1]),
        #                               initializer='glorot_normal',
        #                                trainable=True)
        self.w1 =self.add_weight(name='w1',
                                      shape=(input_shape[0][-1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
                                      
        self.w2 =self.add_weight(name='w2',
                                      shape=(input_shape[1][-1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        self.we1 =self.add_weight(name='we1',
                                      shape=(input_shape[0][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        self.we2 =self.add_weight(name='we2',
                                      shape=(input_shape[1][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        if self.bias:
            self.b1 = self.add_weight(name='b1',shape=None,
                                     initializer='zeros',
                                      trainable=True)
            self.b2 = self.add_weight(name='b2',shape=None,
                                     initializer='zeros',
                                      trainable=True)
            # self.bh = self.add_weight(name='bh',shape=None,
            #                          initializer='zeros',
            #                           trainable=True)
        super(WPE5, self).build(input_shape)  # 一定要在最后调用它
    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None
    def call(self,x,mask=None):
        assert isinstance(x, list)
        x1,x2=x
        input_shape = K.shape(x1)
        # input_len = input_shape[1]
        # q = K.expand_dims(K.dot(x1, self.w1)+self.b1, 2)
        pq=K.batch_dot(x1,x2,axes=2)
        qp=K.permute_dimensions(pq,(0,2,1))
        eij1 = K.tanh(K.dot(pq,self.we1)+K.dot(x1,self.w1)+self.b1)*self.sr
        # k = K.expand_dims(K.dot(x2, self.w2)+self.b2, 1)
        ai1 = K.exp(eij1)
        ww1 = ai1/(K.sum(ai1, axis=1,keepdims=True)+ K.epsilon())
        ot1 = x1*ww1

        # e2 = (e1 - K.min(e2, axis=1, keepdims=True))/(K.max(e2, axis=1, keepdims=True)-K.min(e2, axis=1, keepdims=True)+ K.epsilon())
        eij2 = K.tanh(K.dot(qp,self.we2)+K.dot(x2,self.w2)+self.b2)*self.sr
        ai2 = K.exp(eij2)
        ww2 = ai2/(K.sum(ai2, axis=1,keepdims=True)+ K.epsilon())
        ot2 = x2*ww2
        # return weighted_input.sum(axis=1)
        # oot1=K.sum(ot1,axis=1)
        # oot2=K.sum(ot2,axis=1)
        oot1=K.sum(ot1,axis=1)
        oot2=K.sum(ot2,axis=1)
        # return [ww1,ww2]
        return [ot1,ot2,oot1,oot2]
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0],shape_a[1],shape_a[2]), (shape_b[0],shape_b[1],shape_b[2]),(shape_a[0],shape_a[2]), (shape_b[0],shape_b[2])]
        # return [(shape_a[0],shape_a[1],1), (shape_b[0],shape_b[1],1)]


class WPE4_h(Layer):
    def __init__(self,bias=True,sr=10,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        super(WPE4_h, self).__init__(**kwargs)
        self.bias=bias
        self.sr=sr
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重
        self.ws =self.add_weight(name='ws',
                                      shape=(input_shape[0][-1],input_shape[0][-1]),
                                      initializer='glorot_normal',
                                       trainable=True)
        # self.w1 =self.add_weight(name='w1',
        #                               shape=(input_shape[0][1],1),
        #                               initializer='glorot_normal',
        #                                trainable=True)
                                      
        # self.w2 =self.add_weight(name='w2',
        #                               shape=(input_shape[1][1],1),
        #                               initializer='glorot_normal',
        #                                trainable=True)
        self.we1 =self.add_weight(name='we1',
                                      shape=(input_shape[0][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        self.we2 =self.add_weight(name='we2',
                                      shape=(input_shape[1][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        super(WPE4_h, self).build(input_shape)  # 一定要在最后调用它
    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None
    def call(self,x,mask=None):
        assert isinstance(x, list)
        x1,x2=x
        input_shape = K.shape(x1)
        x1=K.dot(x1,self.ws)
        # q = K.expand_dims(K.dot(x1, self.w1)+self.b1, 2)
        pq=K.batch_dot(x1,x2,axes=2)
        pp=K.batch_dot(x1,x1,axes=2)
        qp=K.batch_dot(x2,x1,axes=2)
        qq=K.batch_dot(x2,x2,axes=2)
        eij1 =K.dot(K.tanh(pq+pp),self.we1)*self.sr
        # k = K.expand_dims(K.dot(x2, self.w2)+self.b2, 1)
        ai1 = K.exp(eij1)
        ww1 = ai1/(K.sum(ai1, axis=1,keepdims=True)+ K.epsilon())
        ot1 = x1*ww1

        # e2 = (e1 - K.min(e2, axis=1, keepdims=True))/(K.max(e2, axis=1, keepdims=True)-K.min(e2, axis=1, keepdims=True)+ K.epsilon())
        eij2 =K.dot(K.tanh(qp+qq),self.we2)*self.sr
        ai2 = K.exp(eij2)
        ww2 = ai2/(K.sum(ai2, axis=1,keepdims=True)+ K.epsilon())
        ot2 = x2*ww2
        # return weighted_input.sum(axis=1)
        # oot1=K.sum(ot1,axis=1)
        # oot2=K.sum(ot2,axis=1)
        oot1=K.sum(ot1,axis=1)
        oot2=K.sum(ot2,axis=1)
        # return [ww1,ww2]
        return [ot1,ot2,oot1,oot2]
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0],shape_a[1],shape_a[2]), (shape_b[0],shape_b[1],shape_b[2]),(shape_a[0],shape_a[2]), (shape_b[0],shape_b[2])]
        # return [(shape_a[0],shape_a[1],1), (shape_b[0],shape_b[1],1)]


class WPE4_s(Layer):
    def __init__(self,bias=True,sr=10,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        super(WPE4_s, self).__init__(**kwargs)
        self.bias=bias
        self.sr=sr
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重
        self.w1 =self.add_weight(name='w1',
                                      shape=(input_shape[0][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
                                      
        # self.w2 =self.add_weight(name='w2',
        #                               shape=(input_shape[1][1],1),
        #                               initializer='glorot_normal',
        #                                trainable=True)
        self.we1 =self.add_weight(name='we1',
                                      shape=(input_shape[0][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        # self.we2 =self.add_weight(name='we2',
        #                               shape=(input_shape[1][1],1),
        #                               initializer='glorot_normal',
        #                                trainable=True)
        if self.bias:
            self.b1 = self.add_weight(name='b1',shape=None,
                                     initializer='zeros',
                                      trainable=True)
            # self.b2 = self.add_weight(name='b2',shape=None,
            #                          initializer='zeros',
            #                           trainable=True)
            # self.bh = self.add_weight(name='bh',shape=None,
            #                          initializer='zeros',
            #                           trainable=True)
        super(WPE4_s, self).build(input_shape)  # 一定要在最后调用它
    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None
    def call(self,x,mask=None):
        assert isinstance(x, list)
        x1,x2=x
        input_shape = K.shape(x1)
        input_len = input_shape[1]
        # q = K.expand_dims(K.dot(x1, self.w1)+self.b1, 2)
        pq=K.batch_dot(x1,x2,axes=2)
        pp=K.batch_dot(x1,x1,axes=2)
        qp=K.batch_dot(x2,x1,axes=2)
        qq=K.batch_dot(x2,x2,axes=2)
        eij1 = K.tanh(K.dot(pq,self.we1)+K.dot(pp,self.w1)+self.b1)*self.sr
        # k = K.expand_dims(K.dot(x2, self.w2)+self.b2, 1)
        ai1 = K.exp(eij1)
        ww1 = ai1/(K.sum(ai1, axis=1,keepdims=True)+ K.epsilon())
        ot1 = x1*ww1

        # e2 = (e1 - K.min(e2, axis=1, keepdims=True))/(K.max(e2, axis=1, keepdims=True)-K.min(e2, axis=1, keepdims=True)+ K.epsilon())
        eij2 = K.tanh(K.dot(qp,self.we1)+K.dot(qq,self.w1)+self.b1)*self.sr
        ai2 = K.exp(eij2)
        ww2 = ai2/(K.sum(ai2, axis=1,keepdims=True)+ K.epsilon())
        ot2 = x2*ww2
        # return weighted_input.sum(axis=1)
        # oot1=K.sum(ot1,axis=1)
        # oot2=K.sum(ot2,axis=1)
        oot1=K.sum(ot1,axis=1)
        oot2=K.sum(ot2,axis=1)
        # return [ww1,ww2]
        return [ot1,ot2,oot1,oot2]
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0],shape_a[1],shape_a[2]), (shape_b[0],shape_b[1],shape_b[2]),(shape_a[0],shape_a[2]), (shape_b[0],shape_b[2])]
        # return [(shape_a[0],shape_a[1],1), (shape_b[0],shape_b[1],1)]

class WPE4(Layer):
    def __init__(self,bias=True,sr=10,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        super(WPE4, self).__init__(**kwargs)
        self.bias=bias
        self.sr=sr
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重
        self.w1 =self.add_weight(name='w1',
                                      shape=(input_shape[0][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
                                      
        self.w2 =self.add_weight(name='w2',
                                      shape=(input_shape[1][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        self.we1 =self.add_weight(name='we1',
                                      shape=(input_shape[0][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        self.we2 =self.add_weight(name='we2',
                                      shape=(input_shape[1][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        if self.bias:
            self.b1 = self.add_weight(name='b1',shape=None,
                                     initializer='zeros',
                                      trainable=True)
            self.b2 = self.add_weight(name='b2',shape=None,
                                     initializer='zeros',
                                      trainable=True)
            # self.bh = self.add_weight(name='bh',shape=None,
            #                          initializer='zeros',
            #                           trainable=True)
        super(WPE4, self).build(input_shape)  # 一定要在最后调用它
    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None
    def call(self,x,mask=None):
        assert isinstance(x, list)
        x1,x2=x
        input_shape = K.shape(x1)
        input_len = input_shape[1]
        # q = K.expand_dims(K.dot(x1, self.w1)+self.b1, 2)
        pq=K.batch_dot(x1,x2,axes=2)
        pp=K.batch_dot(x1,x1,axes=2)
        qp=K.batch_dot(x2,x1,axes=2)
        qq=K.batch_dot(x2,x2,axes=2)
        eij1 = K.tanh(K.dot(pq,self.we1)+K.dot(pp,self.w1)+self.b1)*self.sr
        # k = K.expand_dims(K.dot(x2, self.w2)+self.b2, 1)
        ai1 = K.exp(eij1)
        ww1 = ai1/(K.sum(ai1, axis=1,keepdims=True)+ K.epsilon())
        ot1 = x1*ww1

        # e2 = (e1 - K.min(e2, axis=1, keepdims=True))/(K.max(e2, axis=1, keepdims=True)-K.min(e2, axis=1, keepdims=True)+ K.epsilon())
        eij2 = K.tanh(K.dot(qp,self.we2)+K.dot(qq,self.w2)+self.b2)*self.sr
        ai2 = K.exp(eij2)
        ww2 = ai2/(K.sum(ai2, axis=1,keepdims=True)+ K.epsilon())
        ot2 = x2*ww2
        # return weighted_input.sum(axis=1)
        # oot1=K.sum(ot1,axis=1)
        # oot2=K.sum(ot2,axis=1)
        oot1=K.sum(ot1,axis=1)
        oot2=K.sum(ot2,axis=1)
        # return [ww1,ww2]
        return [ot1,ot2,oot1,oot2]
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0],shape_a[1],shape_a[2]), (shape_b[0],shape_b[1],shape_b[2]),(shape_a[0],shape_a[2]), (shape_b[0],shape_b[2])]
        # return [(shape_a[0],shape_a[1],1), (shape_b[0],shape_b[1],1)]


class WPE3(Layer):
    def __init__(self,bias=True,sr=10,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        super(WPE3, self).__init__(**kwargs)
        self.bias=bias
        self.sr=sr
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重
        self.w1 =self.add_weight(name='w1',
                                      shape=(input_shape[0][-1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
                                      
        self.w2 =self.add_weight(name='w2',
                                      shape=(input_shape[1][-1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        self.we1 =self.add_weight(name='we1',
                                      shape=(input_shape[0][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        self.we2 =self.add_weight(name='we2',
                                      shape=(input_shape[1][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        if self.bias:
            self.b1 = self.add_weight(name='b1',shape=None,
                                     initializer='zeros',
                                      trainable=True)
            self.b2 = self.add_weight(name='b2',shape=None,
                                     initializer='zeros',
                                      trainable=True)
            # self.bh = self.add_weight(name='bh',shape=None,
            #                          initializer='zeros',
            #                           trainable=True)
        super(WPE3, self).build(input_shape)  # 一定要在最后调用它
    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None
    def call(self,x,mask=None):
        assert isinstance(x, list)
        x1,x2=x
        input_shape = K.shape(x1)
        input_len = input_shape[1]
        # q = K.expand_dims(K.dot(x1, self.w1)+self.b1, 2)
        pq=K.batch_dot(x1,x2,axes=2)
        qp=K.batch_dot(x2,x1,axes=2)
        eij1 = K.tanh(K.dot(pq,self.we1)+K.dot(x1,self.w1)+self.b1)*self.sr
        # k = K.expand_dims(K.dot(x2, self.w2)+self.b2, 1)
        ai1 = K.exp(eij1)
        ww1 = ai1/(K.sum(ai1, axis=1,keepdims=True)+ K.epsilon())
        ot1 = x1*ww1

        # e2 = (e1 - K.min(e2, axis=1, keepdims=True))/(K.max(e2, axis=1, keepdims=True)-K.min(e2, axis=1, keepdims=True)+ K.epsilon())
        eij2 = K.tanh(K.dot(qp,self.we2)+K.dot(x2,self.w2)+self.b2)*self.sr
        ai2 = K.exp(eij2)
        ww2 = ai2/(K.sum(ai2, axis=1,keepdims=True)+ K.epsilon())
        ot2 = x2*ww2
        # return weighted_input.sum(axis=1)
        # oot1=K.sum(ot1,axis=1)
        # oot2=K.sum(ot2,axis=1)
        oot1=K.sum(ot1,axis=1)
        oot2=K.sum(ot2,axis=1)
        # return [ww1,ww2]
        return [ot1,ot2,oot1,oot2]
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0],shape_a[1],shape_a[2]), (shape_b[0],shape_b[1],shape_b[2]),(shape_a[0],shape_a[2]), (shape_b[0],shape_b[2])]
        # return [(shape_a[0],shape_a[1],1), (shape_b[0],shape_b[1],1)]

class WPE3_2(Layer):
    def __init__(self,bias=True,sr=10,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        super(WPE3_2, self).__init__(**kwargs)
        self.bias=bias
        self.sr=sr
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重
        self.w1 =self.add_weight(name='w1',
                                      shape=(input_shape[0][-1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
                                      
        # self.w2 =self.add_weight(name='w2',
        #                               shape=(input_shape[1][-1],1),
        #                               initializer='glorot_normal',
        #                                trainable=True)
        self.we =self.add_weight(name='we',
                                      shape=(input_shape[0][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        if self.bias:
            self.b1 = self.add_weight(name='b1',shape=None,
                                     initializer='zeros',
                                      trainable=True)
            # self.b2 = self.add_weight(name='b2',shape=None,
            #                          initializer='zeros',
            #                           trainable=True)
            # self.bh = self.add_weight(name='bh',shape=None,
            #                          initializer='zeros',
            #                           trainable=True)
        super(WPE3_2, self).build(input_shape)  # 一定要在最后调用它
    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None
    def call(self,x,mask=None):
        assert isinstance(x, list)
        x1,x2=x
        input_shape = K.shape(x1)
        input_len = input_shape[1]
        # q = K.expand_dims(K.dot(x1, self.w1)+self.b1, 2)
        pq=K.batch_dot(x1,x2,axes=2)
        qp=K.batch_dot(x2,x1,axes=2)
        eij1 = K.tanh(K.dot(pq,self.we)+K.dot(x1,self.w1)+self.b1)*self.sr
        # k = K.expand_dims(K.dot(x2, self.w2)+self.b2, 1)
        ai1 = K.exp(eij1)
        ww1 = ai1/(K.sum(ai1, axis=1,keepdims=True)+ K.epsilon())
        ot1 = x1*ww1

        # e2 = (e1 - K.min(e2, axis=1, keepdims=True))/(K.max(e2, axis=1, keepdims=True)-K.min(e2, axis=1, keepdims=True)+ K.epsilon())
        eij2 = K.tanh(K.dot(qp,self.we)+K.dot(x2,self.w1)+self.b1)*self.sr
        ai2 = K.exp(eij2)
        ww2 = ai2/(K.sum(ai2, axis=1,keepdims=True)+ K.epsilon())
        ot2 = x2*ww2
        # return weighted_input.sum(axis=1)
        # oot1=K.sum(ot1,axis=1)
        # oot2=K.sum(ot2,axis=1)
        oot1=K.sum(ot1,axis=1)
        oot2=K.sum(ot2,axis=1)
        # return [ww1,ww2]
        return [ot1,ot2,oot1,oot2]
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0],shape_a[1],shape_a[2]), (shape_b[0],shape_b[1],shape_b[2]),(shape_a[0],shape_a[2]), (shape_b[0],shape_b[2])]
        # return [(shape_a[0],shape_a[1],1), (shape_b[0],shape_b[1],1)]


class WPE3_1(Layer):
    def __init__(self,bias=True,sr=10,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        super(WPE3_1, self).__init__(**kwargs)
        self.bias=bias
        self.sr=sr
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重
        self.w1 =self.add_weight(name='w1',
                                      shape=(input_shape[0][-1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
                                      
        self.w2 =self.add_weight(name='w2',
                                      shape=(input_shape[1][-1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        self.we =self.add_weight(name='we',
                                      shape=(input_shape[0][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        if self.bias:
            self.b1 = self.add_weight(name='b1',shape=None,
                                     initializer='zeros',
                                      trainable=True)
            self.b2 = self.add_weight(name='b2',shape=None,
                                     initializer='zeros',
                                      trainable=True)
            # self.bh = self.add_weight(name='bh',shape=None,
            #                          initializer='zeros',
            #                           trainable=True)
        super(WPE3_1, self).build(input_shape)  # 一定要在最后调用它
    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None
    def call(self,x,mask=None):
        assert isinstance(x, list)
        x1,x2=x
        input_shape = K.shape(x1)
        input_len = input_shape[1]
        # q = K.expand_dims(K.dot(x1, self.w1)+self.b1, 2)
        pq=K.batch_dot(x1,x2,axes=2)
        qp=K.batch_dot(x2,x1,axes=2)
        eij1 = K.tanh(K.dot(pq,self.we)+K.dot(x1,self.w1)+self.b1)*self.sr
        # k = K.expand_dims(K.dot(x2, self.w2)+self.b2, 1)
        ai1 = K.exp(eij1)
        ww1 = ai1/(K.sum(ai1, axis=1,keepdims=True)+ K.epsilon())
        ot1 = x1*ww1

        # e2 = (e1 - K.min(e2, axis=1, keepdims=True))/(K.max(e2, axis=1, keepdims=True)-K.min(e2, axis=1, keepdims=True)+ K.epsilon())
        eij2 = K.tanh(K.dot(qp,self.we)+K.dot(x2,self.w2)+self.b2)*self.sr
        ai2 = K.exp(eij2)
        ww2 = ai2/(K.sum(ai2, axis=1,keepdims=True)+ K.epsilon())
        ot2 = x2*ww2
        # return weighted_input.sum(axis=1)
        # oot1=K.sum(ot1,axis=1)
        # oot2=K.sum(ot2,axis=1)
        oot1=K.sum(ot1,axis=1)
        oot2=K.sum(ot2,axis=1)
        # return [ww1,ww2]
        return [ot1,ot2,oot1,oot2]
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0],shape_a[1],shape_a[2]), (shape_b[0],shape_b[1],shape_b[2]),(shape_a[0],shape_a[2]), (shape_b[0],shape_b[2])]
        # return [(shape_a[0],shape_a[1],1), (shape_b[0],shape_b[1],1)]

class WPE(Layer):
    def __init__(self,bias=True,sr=10,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        super(WPE, self).__init__(**kwargs)
        self.bias=bias
        self.sr=sr
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重
        self.w1 =self.add_weight(name='w1',
                                      shape=(input_shape[0][-1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
                                      
        self.w2 =self.add_weight(name='w2',
                                      shape=(input_shape[1][-1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        self.we1 =self.add_weight(name='we1',
                                      shape=(input_shape[0][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        self.we2 =self.add_weight(name='we2',
                                      shape=(input_shape[1][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        if self.bias:
            self.b1 = self.add_weight(name='b1',shape=None,
                                     initializer='zeros',
                                      trainable=True)
            self.b2 = self.add_weight(name='b2',shape=None,
                                     initializer='zeros',
                                      trainable=True)
            # self.bh = self.add_weight(name='bh',shape=None,
            #                          initializer='zeros',
            #                           trainable=True)
        super(WPE, self).build(input_shape)  # 一定要在最后调用它
    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None
    def call(self,x,mask=None):
        assert isinstance(x, list)
        x1,x2=x
        input_shape = K.shape(x1)
        input_len = input_shape[1]
        # q = K.expand_dims(K.dot(x1, self.w1)+self.b1, 2)
        q=K.dot(x1, self.w1)+self.b1
        k=K.dot(x2, self.w2)+self.b2
        # k = K.expand_dims(K.dot(x2, self.w2)+self.b2, 1)
        k=K.reshape(k,(-1,1,input_len))
        h = q+k
        e = K.reshape(h, (-1, input_len, input_len))
        # e1= K.sum(e, axis=2,keepdims=True)
        e1=K.dot(e,self.we1)
        # e1 = (e1 - K.min(e1, axis=1, keepdims=True))/(K.max(e1, axis=1, keepdims=True)-K.min(e1, axis=1, keepdims=True)+ K.epsilon())
        # eij1=K.tanh(e1)
        eij1=K.tanh(e1)*self.sr
        ai1 = K.exp(eij1)
        # if mask is not None:
        #     mask1 = K.cast(mask[0], K.floatx())
        #     # mask = K.expand_dims(mask)
        #     # e = K.permute_dimensions(K.permute_dimensions(ai1 * mask, (0, 2, 1)) * mask, (0, 2, 1))
        #     ai1=ai1*mask1
        ww1 = ai1/(K.sum(ai1, axis=1,keepdims=True)+ K.epsilon())
        ot1 = x1*ww1
        e2= K.permute_dimensions(e, (0, 2, 1))
        e2=K.dot(e2,self.we2)
        # e2 = (e1 - K.min(e2, axis=1, keepdims=True))/(K.max(e2, axis=1, keepdims=True)-K.min(e2, axis=1, keepdims=True)+ K.epsilon())
        eij2=K.tanh(e2)*self.sr
        ai2 = K.exp(eij2)
        # if mask is not None:
        #     mask2 = K.cast(mask[1], K.floatx())
        #     # mask = K.expand_dims(mask)
        #     # e = K.permute_dimensions(K.permute_dimensions(ai1 * mask, (0, 2, 1)) * mask, (0, 2, 1))
        #     ai2=ai2*mask2
        ww2 = ai2/(K.sum(ai2, axis=1,keepdims=True)+ K.epsilon())
        ot2 = x2*ww2
        # return weighted_input.sum(axis=1)
        # oot1=K.sum(ot1,axis=1)
        # oot2=K.sum(ot2,axis=1)
        oot1=K.sum(ot1,axis=1)
        oot2=K.sum(ot2,axis=1)
        # return [ww1,ww2]
        return [ot1,ot2,oot1,oot2]
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0],shape_a[1],shape_a[2]), (shape_b[0],shape_b[1],shape_b[2]),(shape_a[0],shape_a[2]), (shape_b[0],shape_b[2])]
        # return [(shape_a[0],shape_a[1],1), (shape_b[0],shape_b[1],1)]


class WPE2(Layer):
    def __init__(self,bias=True,sr=10,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        super(WPE2, self).__init__(**kwargs)
        self.bias=bias
        self.sr=sr
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.w1 =self.add_weight(name='w1',
                                      shape=(input_shape[0][-1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
                                      
        self.w2 =self.add_weight(name='w2',
                                      shape=(input_shape[1][-1],1),
                                      initializer='glorot_normal',
                                       trainable=True)
        self.we =self.add_weight(name='we1',
                                      shape=(input_shape[0][1],1),
                                      initializer='glorot_normal',
                                       trainable=True)

        if self.bias:
            self.b1 = self.add_weight(name='b1',shape=None,
                                     initializer='zeros',
                                      trainable=True)
            self.b2 = self.add_weight(name='b2',shape=None,
                                     initializer='zeros',
                                      trainable=True)
            # self.bh = self.add_weight(name='bh',shape=None,
            #                          initializer='zeros',
            #                           trainable=True)
        super(WPE2, self).build(input_shape)  # 一定要在最后调用它
    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None
    def call(self,x,mask=None):
        assert isinstance(x, list)
        x1,x2=x
        input_shape = K.shape(x1)
        input_len = input_shape[1]
        # q = K.expand_dims(K.dot(x1, self.w1)+self.b1, 2)
        q=K.dot(x1, self.w1)+self.b1
        k=K.dot(x2, self.w2)+self.b2
        # k = K.expand_dims(K.dot(x2, self.w2)+self.b2, 1)
        k=K.reshape(k,(-1,1,input_len))
        h = q+k
        e = K.reshape(h, (-1, input_len, input_len))
        # e1= K.sum(e, axis=2,keepdims=True)
        e1=K.dot(e,self.we)
        # e1 = (e1 - K.min(e1, axis=1, keepdims=True))/(K.max(e1, axis=1, keepdims=True)-K.min(e1, axis=1, keepdims=True)+ K.epsilon())
        # eij1=K.tanh(e1)
        eij1=K.tanh(e1)*self.sr
        ai1 = K.exp(eij1)
        # if mask is not None:
        #     mask1 = K.cast(mask[0], K.floatx())
        #     # mask = K.expand_dims(mask)
        #     # e = K.permute_dimensions(K.permute_dimensions(ai1 * mask, (0, 2, 1)) * mask, (0, 2, 1))
        #     ai1=ai1*mask1
        ww1 = ai1/(K.sum(ai1, axis=1,keepdims=True)+ K.epsilon())
        ot1 = x1*ww1
        e2= K.permute_dimensions(e, (0, 2, 1))
        e2=K.dot(e2,self.we)
        # e2 = (e1 - K.min(e2, axis=1, keepdims=True))/(K.max(e2, axis=1, keepdims=True)-K.min(e2, axis=1, keepdims=True)+ K.epsilon())
        eij2=K.tanh(e2)*self.sr
        ai2 = K.exp(eij2)
        # if mask is not None:
        #     mask2 = K.cast(mask[1], K.floatx())
        #     # mask = K.expand_dims(mask)
        #     # e = K.permute_dimensions(K.permute_dimensions(ai1 * mask, (0, 2, 1)) * mask, (0, 2, 1))
        #     ai2=ai2*mask2
        ww2 = ai2/(K.sum(ai2, axis=1,keepdims=True)+ K.epsilon())
        ot2 = x2*ww2
        # return weighted_input.sum(axis=1)
        # oot1=K.sum(ot1,axis=1)
        # oot2=K.sum(ot2,axis=1)
        oot1=K.sum(ot1,axis=1)
        oot2=K.sum(ot2,axis=1)
        # return [ww1,ww2]
        return [ot1,ot2,oot1,oot2]
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0],shape_a[1],shape_a[2]), (shape_b[0],shape_b[1],shape_b[2]),(shape_a[0],shape_a[2]), (shape_b[0],shape_b[2])]
        # return [(shape_a[0],shape_a[1],1), (shape_b[0],shape_b[1],1)]


class Co_Attention(Layer):
 
    def __init__(self,bias=True,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        super(Co_Attention, self).__init__(**kwargs)
        self.bias=bias
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重

        self.w1 =self.add_weight(name='w1',
                                      shape=(input_shape[1][1], input_shape[0][1]),
                                      initializer='glorot_normal',
                                      trainable=True)
                                      
        self.w2 =self.add_weight(name='w2',
                                      shape=(input_shape[0][1], input_shape[1][1]),
                                      initializer='glorot_normal',
                                      trainable=True)
        if self.bias:
            self.b1 = self.add_weight(name='b1',shape=(input_shape[1][1],),
                                     initializer='zeros',
                                      trainable=True)
            self.b2 = self.add_weight(name='b2',shape=(input_shape[0][1],),
                                     initializer='zeros',
                                      trainable=True)
        super(Co_Attention, self).build(input_shape)  # 一定要在最后调用它
    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None
    def call(self,x,mask=None):
        assert isinstance(x, list)
        s1,s2=x

        # s1.shape = (batch_size, s1_time_steps, dim)
        # s2.shape = (batch_size, s2_time_steps, dim)
        x1 = K.permute_dimensions(s1, (0, 2, 1))
        x2 = K.permute_dimensions(s2, (0, 2, 1))

      

        # uit = K.dot(x2, self.w1)
        # uit = K.tanh(uit)
        # a1 = K.exp(uit)
        # a1 /= K.cast(K.sum(a1, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        # uit = K.dot(x1, self.w2)
        # uit = K.tanh(uit)
        # a2 = K.exp(uit)
        # a2 /= K.cast(K.sum(a2, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        # uit1 = K.dot(x2, self.w1)
        # uit2 = K.dot(x2, self.w1)
        # uit = K.tanh(uit)
        # cos = 
        # a1 = K.exp(uit)
        # a1 /= K.cast(K.sum(a1, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        # uit = K.dot(x1, self.w2)
        # uit = K.tanh(uit)
        # a2 = K.exp(uit)
        # a2 /= K.cast(K.sum(a2, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        # K.batch_dot(x[0],K.permute_dimensions(x[1], (0, 2, 1)))#/K.batch_dot(K.sqrt(x[0]*x[0]),K.permute_dimensions(K.sqrt(x[1]*x[1]), (0, 2, 1)))
        uit = K.dot(x2, self.w1)+self.b1
        uit = K.tanh(uit)
        a1 = K.exp(uit)
        # if mask is not None:
        #     # Cast the mask to floatX to avoid float64 upcasting in theano
        #     a1 *= K.cast(mask, K.floatx())
            
        a1 /= K.cast(K.sum(a1, axis=2, keepdims=True) + K.epsilon(), K.floatx())

        uit = K.dot(x1, self.w2)+self.b2
        uit = K.tanh(uit)
        a2 = K.exp(uit)
        # if mask is not None:
        #     # Cast the mask to floatX to avoid float64 upcasting in theano
        #     a2 *= K.cast(mask, K.floatx())
        a2 /= K.cast(K.sum(a2, axis=2, keepdims=True) + K.epsilon(), K.floatx())

        # m1=K.tanh(K.dot(x2, self.w1)+self.b1)
        # m2=K.tanh(K.dot(x1, self.w2)+self.b2)


        # a1 = K.softmax(m1)
        # a2 = K.softmax(m2)


        outputs1 = K.permute_dimensions(a1*x2, (0, 2, 1))
        outputs2 = K.permute_dimensions(a2*x1, (0, 2, 1))

        # o1 = K.concatenate([x1,outputs1],axis=1)
        # o2 = K.concatenate([x2,outputs2],axis=1)
        return [outputs1,outputs2]
 
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0],shape_a[1],shape_a[2]), (shape_b[0],shape_b[1],shape_b[2])]

class FindWordPair(Layer):
 
    def __init__(self,bias=True,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        super(FindWordPair, self).__init__(**kwargs)
        self.bias=bias
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重

        self.w1 =self.add_weight(name='w1',
                                      shape=(input_shape[0][-1],1),
                                      initializer='glorot_normal',
                                      trainable=True)
                                      
        self.w2 =self.add_weight(name='w2',
                                      shape=(input_shape[1][-1],1),
                                      initializer='glorot_normal',
                                      trainable=True)
        if self.bias:
            self.b1 = self.add_weight(name='b1',shape=None,
                                     initializer='zeros',
                                      trainable=True)
            self.b2 = self.add_weight(name='b2',shape=None,
                                     initializer='zeros',
                                      trainable=True)
        super(FindWordPair, self).build(input_shape)  # 一定要在最后调用它
    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None
    def call(self,x,mask=None):
        assert isinstance(x, list)
        x1,x2=x
        eij1 = K.tanh(K.dot(x1*x2,self.w1)+self.b1)*10
        ai1 = K.exp(eij1)
        ww1 = ai1/(K.sum(ai1, axis=1,keepdims=True)+ K.epsilon())
        ot1 = x1*ww1

        eij2 = K.tanh(K.dot(x1*x2,self.w2)+self.b2)*10
        ai2 = K.exp(eij2)
        ww2 = ai2/(K.sum(ai2, axis=1,keepdims=True)+ K.epsilon())
        ot2 = x2*ww2

        # return weighted_input.sum(axis=1)
        # oot1=K.sum(ot1,axis=1)
        # oot2=K.sum(ot2,axis=1)
        oot1=K.sum(ot1,axis=1)
        oot2=K.sum(ot2,axis=1)
        # return [oo1,oo2]
        return [ot1,ot2,oot1,oot2]
 
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0],shape_a[1],shape_a[2]), (shape_b[0],shape_b[1],shape_b[2]),(shape_a[0],shape_a[2]), (shape_b[0],shape_b[2])]

# class FindWordPair_1(Layer):
 
#     def __init__(self,bias=True,**kwargs):
#         # self.output_dim = output_dim
#         self.supports_masking = True
#         super(FindWordPair_1, self).__init__(**kwargs)
#         self.bias=bias
#     def build(self, input_shape):
#         assert isinstance(input_shape, list)
#         # 为该层创建一个可训练的权重

#         self.w1 =self.add_weight(name='w1',
#                                       shape=(input_shape[0][-1],1),
#                                       initializer='glorot_normal',
#                                       trainable=True)
                                      
#         self.w2 =self.add_weight(name='w2',
#                                       shape=(input_shape[1][-1],1),
#                                       initializer='glorot_normal',
#                                       trainable=True)
#         self.w3 =self.add_weight(name='w3',
#                                       shape=(input_shape[0][-1],1),
#                                       initializer='glorot_normal',
#                                       trainable=True)
                                      
#         self.w4 =self.add_weight(name='w4',
#                                       shape=(input_shape[1][-1],1),
#                                       initializer='glorot_normal',
#                                       trainable=True)
#         if self.bias:
#             self.b1 = self.add_weight(name='b1',shape=None,
#                                      initializer='zeros',
#                                       trainable=True)
#             self.b2 = self.add_weight(name='b2',shape=None,
#                                      initializer='zeros',
#                                       trainable=True)
#         super(FindWordPair_1, self).build(input_shape)  # 一定要在最后调用它
#     def compute_mask(self, input, input_mask=None):
#         # need not to pass the mask to next layers
#         return None
#     def call(self,x,mask=None):
#         assert isinstance(x, list)
#         x1,x2=x
#         eij1 = K.tanh(K.dot(x1*x2,self.w1)+K.dot(x1,self.w3)+self.b1)*10
#         ai1 = K.exp(eij1)
#         ww1 = ai1/(K.sum(ai1, axis=1,keepdims=True)+ K.epsilon())
#         ot1 = x1*ww1

#         eij2 = K.tanh(K.dot(x2*x1,self.w2)+K.dot(x2,self.w4)+self.b2)*10
#         ai2 = K.exp(eij2)
#         ww2 = ai2/(K.sum(ai2, axis=1,keepdims=True)+ K.epsilon())
#         ot2 = x2*ww2

#         # return weighted_input.sum(axis=1)
#         # oot1=K.sum(ot1,axis=1)
#         # oot2=K.sum(ot2,axis=1)
        
#         return [ot1,ot2]
 
#     def compute_output_shape(self, input_shape):
#         assert isinstance(input_shape, list)
#         shape_a, shape_b = input_shape
#         return [(shape_a[0],shape_a[1],shape_a[2]), (shape_b[0],shape_b[1],shape_b[2])]
    
class FindWordPair_2(Layer):
 
    def __init__(self,bias=True,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        super(FindWordPair_2, self).__init__(**kwargs)
        self.bias=bias
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重

        self.w1 =self.add_weight(name='w1',
                                      shape=(input_shape[0][-1],1),
                                      initializer='glorot_normal',
                                      trainable=True)
                                      
        self.w2 =self.add_weight(name='w2',
                                      shape=(input_shape[1][-1],1),
                                      initializer='glorot_normal',
                                      trainable=True)
        if self.bias:
            self.b1 = self.add_weight(name='b1',
                                     initializer='zeros',
                                      trainable=True)
            self.b2 = self.add_weight(name='b2',
                                     initializer='zeros',
                                      trainable=True)
        super(FindWordPair_2, self).build(input_shape)  # 一定要在最后调用它
    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None
    def call(self,x,mask=None):
        assert isinstance(x, list)
        x1,x2=x
        eij1 = K.tanh(K.dot(x1*(x1-x2),self.w1)+self.b1)*10
        ai1 = K.exp(eij1)
        ww1 = ai1/(K.sum(ai1, axis=1,keepdims=True)+ K.epsilon())
        ot1 = x1*ww1

        eij2 = K.tanh(K.dot(x2*(x2-x1),self.w2)+self.b2)*10
        ai2 = K.exp(eij2)
        ww2 = ai2/(K.sum(ai2, axis=1,keepdims=True)+ K.epsilon())
        ot2 = x2*ww2

        # return weighted_input.sum(axis=1)
        # oot1=K.sum(ot1,axis=1)
        # oot2=K.sum(ot2,axis=1)
        
        return [ot1,ot2]
 
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0],shape_a[1],shape_a[2]), (shape_b[0],shape_b[1],shape_b[2])]

# class FindWordPair_3(Layer):
 
#     def __init__(self,bias=True,**kwargs):
#         # self.output_dim = output_dim
#         self.supports_masking = True
#         super(FindWordPair_3, self).__init__(**kwargs)
#         self.bias=bias
#     def build(self, input_shape):
#         assert isinstance(input_shape, list)
#         # 为该层创建一个可训练的权重

#         self.w1 =self.add_weight(name='w1',
#                                       shape=(input_shape[0][-1],1),
#                                       initializer='glorot_normal',
#                                       trainable=True)
                                      
#         self.w2 =self.add_weight(name='w2',
#                                       shape=(input_shape[1][-1],1),
#                                       initializer='glorot_normal',
#                                       trainable=True)
#         if self.bias:
#             self.b1 = self.add_weight(name='b1',
#                                      initializer='zeros',
#                                       trainable=True)
#             self.b2 = self.add_weight(name='b2',
#                                      initializer='zeros',
#                                       trainable=True)
#         super(FindWordPair_3, self).build(input_shape)  # 一定要在最后调用它
#     def compute_mask(self, input, input_mask=None):
#         # need not to pass the mask to next layers
#         return None
#     def call(self,x,mask=None):
#         assert isinstance(x, list)
#         x1,x2=x
#         eij1 = K.tanh(K.dot(x1,self.w1)+K.dot(x2,self.w2)+self.b1)*10
#         ai1 = K.exp(eij1)
#         ww1 = ai1/(K.sum(ai1, axis=1,keepdims=True)+ K.epsilon())
#         ot1 = x1*ww1

#         eij2 = K.tanh(K.dot(x2*(x2-x1),self.w2)+self.b2)*10
#         ai2 = K.exp(eij2)
#         ww2 = ai2/(K.sum(ai2, axis=1,keepdims=True)+ K.epsilon())
#         ot2 = x2*ww2

#         # return weighted_input.sum(axis=1)
#         # oot1=K.sum(ot1,axis=1)
#         # oot2=K.sum(ot2,axis=1)
        
#         return [ot1,ot2]
 
#     def compute_output_shape(self, input_shape):
#         assert isinstance(input_shape, list)
#         shape_a, shape_b = input_shape
#         return [(shape_a[0],shape_a[1],shape_a[2]), (shape_b[0],shape_b[1],shape_b[2])]
class FindWordPair_3(Layer):
 
    def __init__(self,bias=True,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        super(FindWordPair_3, self).__init__(**kwargs)
        self.bias=bias
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重

        self.w1 =self.add_weight(name='w1',
                                      shape=(input_shape[0][-1],1),
                                      initializer='glorot_normal',
                                      trainable=True)
                                      
        self.w2 =self.add_weight(name='w2',
                                      shape=(input_shape[1][-1],1),
                                      initializer='glorot_normal',
                                      trainable=True)
        self.w3 =self.add_weight(name='w3',
                                      shape=(input_shape[0][-1],1),
                                      initializer='glorot_normal',
                                      trainable=True)
                                      
        self.w4 =self.add_weight(name='w4',
                                      shape=(input_shape[1][-1],1),
                                      initializer='glorot_normal',
                                      trainable=True)
        if self.bias:
            self.b1 = self.add_weight(name='b1',shape=None,
                                     initializer='zeros',
                                      trainable=True)
            self.b2 = self.add_weight(name='b2',shape=None,
                                     initializer='zeros',
                                      trainable=True)
        super(FindWordPair_3, self).build(input_shape)  # 一定要在最后调用它
    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None
    def call(self,x,mask=None):
        assert isinstance(x, list)
        x1,x2=x
        eij1 = K.tanh(K.dot(x1,self.w1)+K.dot(x2,self.w2)+self.b1)*10
        ai1 = K.exp(eij1)
        ww1 = ai1/(K.sum(ai1, axis=1,keepdims=True)+ K.epsilon())
        ot1 = x1*ww1
    
        eij2 = K.tanh(K.dot(x1,self.w3)+K.dot(x2,self.w4)+self.b2)*10
        ai2 = K.exp(eij2)
        ww2 = ai2/(K.sum(ai2, axis=1,keepdims=True)+ K.epsilon())
        ot2 = x2*ww2

        # return weighted_input.sum(axis=1)
        # oot1=K.sum(ot1,axis=1)
        # oot2=K.sum(ot2,axis=1)
        
        return [ot1,ot2]
 
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0],shape_a[1],shape_a[2]), (shape_b[0],shape_b[1],shape_b[2])]

class FindWordPair2(Layer):
 
    def __init__(self,bias=True,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        super(FindWordPair2, self).__init__(**kwargs)
        self.bias=bias
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重

        self.w1 =self.add_weight(name='w1',
                                      shape=(input_shape[0][-1],1),
                                      initializer='glorot_normal',
                                      trainable=True)
                                      
        self.w2 =self.add_weight(name='w2',
                                      shape=(input_shape[1][-1],1),
                                      initializer='glorot_normal',
                                      trainable=True)
        if self.bias:
            self.b1 = self.add_weight(name='b1',
                                     initializer='zeros',
                                      trainable=True)
            self.b2 = self.add_weight(name='b2',
                                     initializer='zeros',
                                      trainable=True)
        super(FindWordPair2, self).build(input_shape)  # 一定要在最后调用它
    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None
    def call(self,x,mask=None):
        assert isinstance(x, list)
        x1,x2=x
        eij1 = K.tanh(K.dot(x1*x2,self.w1)+self.b1)*10
        ai1 = K.exp(eij1)
        ww1 = ai1/(K.sum(ai1, axis=1,keepdims=True)+ K.epsilon())
        ot1 = x1*ww1

        eij2 = K.tanh(K.dot(x1*x2,self.w2)+self.b2)*10
        ai2 = K.exp(eij2)
        ww2 = ai2/(K.sum(ai2, axis=1,keepdims=True)+ K.epsilon())
        ot2 = x2*ww2

        # return weighted_input.sum(axis=1)
        oot1=K.sum(ot1,axis=1)
        oot2=K.sum(ot2,axis=1)
        oo1=K.expand_dims(oot1,1)
        oo2=K.expand_dims(oot2,1)
        return [oo1,oo2]
 
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0],1,shape_a[2]), (shape_b[0],1,shape_b[2])]


class FindWordPair3(Layer):
 
    def __init__(self,bias=True,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        super(FindWordPair3, self).__init__(**kwargs)
        self.bias=bias
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重

        self.w1 =self.add_weight(name='w1',
                                      shape=(input_shape[0][-1],1),
                                      initializer='glorot_normal',
                                      trainable=True)
                                      
        self.w2 =self.add_weight(name='w2',
                                      shape=(input_shape[1][-1],1),
                                      initializer='glorot_normal',
                                      trainable=True)
        self.wp1 =self.add_weight(name='wp1',
                                      shape=(input_shape[2][-1],1),
                                      initializer='glorot_normal',
                                      trainable=True)
        self.wp2 =self.add_weight(name='wp2',
                                      shape=(input_shape[3][-1],1),
                                      initializer='glorot_normal',
                                      trainable=True)

        if self.bias:
            self.b1 = self.add_weight(name='b1',
                                     initializer='zeros',
                                      trainable=True)
            self.b2 = self.add_weight(name='b2',
                                     initializer='zeros',
                                      trainable=True)
        super(FindWordPair3, self).build(input_shape)  # 一定要在最后调用它
    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None
    def call(self,x,mask=None):
        assert isinstance(x, list)
        x1,x2,wx1,wx2=x
        wp=K.dot(wx1, self.wp1)+K.dot(wx2, self.wp2)
        # on=K.cast(tf.ones(wp1.shape[0],wp1.shape[1],x1.shape[1]), dtype=float)
        uit1 = K.cast(K.dot(x1*x2, self.w1)+wp+self.b1, K.floatx())
        # eij1 = K.tanh(uit)
        eij1 = K.tanh(uit1)*10
        ai1 = K.exp(eij1)
        ww1 = ai1/(K.sum(ai1, axis=1,keepdims=True)+ K.epsilon())
        ot1 = x1*ww1

        uit2 = K.cast(K.dot(x1*x2, self.w2)+wp+self.b2, K.floatx())
        eij2 = K.tanh(uit2)*10
        ai2 = K.exp(eij2)
        ww2 = ai2/(K.sum(ai2, axis=1,keepdims=True)+ K.epsilon())
        ot2 = x2*ww2

        # return weighted_input.sum(axis=1)
        oot1=K.sum(ot1,axis=1)
        oot2=K.sum(ot2,axis=1)
        oo1=K.expand_dims(oot1,1)
        oo2=K.expand_dims(oot2,1)
        return [oo1,oo2]
 
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b,s1,s2 = input_shape
        return [(shape_a[0],1,shape_a[2]), (shape_b[0],1,shape_b[2])]

# class FindWordPair3(Layer):
 
#     def __init__(self,bias=True,**kwargs):
#         # self.output_dim = output_dim
#         self.supports_masking = True
#         super(FindWordPair3, self).__init__(**kwargs)
#         self.bias=bias
#     def build(self, input_shape):
#         assert isinstance(input_shape, list)
#         # 为该层创建一个可训练的权重

#         self.w1 =self.add_weight(name='w1',
#                                       shape=(input_shape[0][1], input_shape[0][1]),
#                                       initializer='glorot_normal',
#                                       trainable=True)
                                      
#         self.w2 =self.add_weight(name='w2',
#                                       shape=(input_shape[1][1], input_shape[1][1]),
#                                       initializer='glorot_normal',
#                                       trainable=True)
#         self.wx1 =self.add_weight(name='wx1',
#                                       shape=(input_shape[2][1], input_shape[2][1]),
#                                       initializer='glorot_normal',
#                                       trainable=True)
                                      
#         self.wx2 =self.add_weight(name='wx2',
#                                       shape=(input_shape[3][1], input_shape[3][1]),
#                                       initializer='glorot_normal',
#                                       trainable=True)
#         self.ones =self.add_weight(name='ones',
#                                       shape=(1,input_shape[1][1]),
#                                       initializer='ones')
#         if self.bias:
#             self.b1 = self.add_weight(name='b1',shape=(input_shape[1][1],),
#                                      initializer='zeros',
#                                       trainable=True)
#             self.b2 = self.add_weight(name='b2',shape=(input_shape[0][1],),
#                                      initializer='zeros',
#                                       trainable=True)
#         super(FindWordPair3, self).build(input_shape)  # 一定要在最后调用它
#     def compute_mask(self, input, input_mask=None):
#         # need not to pass the mask to next layers
#         return None
#     def call(self,x,mask=None):
#         assert isinstance(x, list)
#         x1,x2,wp1,wp2=x
#         x1 = K.permute_dimensions(x1, (0, 2, 1))
#         x2 = K.permute_dimensions(x2, (0, 2, 1))
#         wp1 = K.permute_dimensions(wp1, (0, 2, 1))
#         wp2 = K.permute_dimensions(wp2, (0, 2, 1))
#         # ss=K.cast(np.ones(x1.shape[2],x1.shape[1]), K.floatx())
#         wp=K.dot(wp1, self.wx1)+K.dot(wp2, self.wx2)
#         wwp=K.dot(wp,self.ones)
#         # on=K.cast(tf.ones(wp1.shape[0],wp1.shape[1],x1.shape[1]), dtype=float)
#         uit = K.cast(K.dot(x1*x2, self.w1)+wwp+self.b1, K.floatx())
#         uit = K.tanh(uit)
#         uit = uit*10
#         a1 = K.exp(uit)
#         a1 /= K.cast(K.sum(a1, axis=2, keepdims=True) + K.epsilon(), K.floatx())

#         uit = K.cast(K.dot(x1*x2, self.w2)+wwp+self.b2, K.floatx())
#         uit = K.tanh(uit)
#         uit = uit*10
#         a2 = K.exp(uit)
#         a2 /= K.cast(K.sum(a2, axis=2, keepdims=True) + K.epsilon(), K.floatx())

#         outputs1 = K.permute_dimensions(a1*x1, (0, 2, 1))
#         outputs2 = K.permute_dimensions(a2*x2, (0, 2, 1))
#         o1=K.sum(outputs1, axis=1)
#         o2=K.sum(outputs2, axis=1)
#         oo1=K.expand_dims(o1,1)
#         oo2=K.expand_dims(o2,1)
#         # o1 = K.concatenate([x1,outputs1],axis=1)
#         # o2 = K.concatenate([x2,outputs2],axis=1)
#         return [oo1,oo2]
 
#     def compute_output_shape(self, input_shape):
#         assert isinstance(input_shape, list)
#         shape_a, shape_b,ss1,ss2 = input_shape
#         return [(shape_a[0],1,shape_a[2]), (shape_b[0],1,shape_b[2])]

class SelfAtt(Layer):
 
    def __init__(self,bias=True,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        super(SelfAtt, self).__init__(**kwargs)
        self.bias=bias
    def build(self, input_shape):
        assert len(input_shape) == 3
        # 为该层创建一个可训练的权重

        self.w1 =self.add_weight(name='w1',
                                      shape=(input_shape[1], input_shape[1]),
                                      initializer='glorot_normal',
                                      trainable=True)
        if self.bias:
            self.b1 = self.add_weight(name='b1',shape=(input_shape[1],),
                                     initializer='zeros',
                                      trainable=True)
        super(SelfAtt, self).build(input_shape)  # 一定要在最后调用它
    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None
    def call(self,x,mask=None):
        assert len(x.shape) == 3
        # x1 = x
        x1 = K.permute_dimensions(x, (0, 2, 1))
        uit = K.dot(x1, self.w1)+self.b1
        uit = K.tanh(uit)
        a1 = K.exp(uit)
        a1 /= K.cast(K.sum(a1, axis=2, keepdims=True) + K.epsilon(), K.floatx())
        outputs1 = K.permute_dimensions(a1*x1, (0, 2, 1))
        o1=K.sum(outputs1, axis=1)
        # oo1=K.expand_dims(o1,1)
        # o1 = K.concatenate([x1,outputs1],axis=1)
        # o2 = K.concatenate([x2,outputs2],axis=1)
        return o1
 
    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 3
        return (input_shape[0], input_shape[-1])

class WAdd(Layer):
 
    def __init__(self,bias=True,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        super(WAdd, self).__init__(**kwargs)
        self.bias=bias
    def build(self, input_shape):
        assert isinstance(input_shape, list)

        self.w1 =self.add_weight(name='w1',
                                      shape=(input_shape[0][-1], input_shape[0][-1]),
                                      initializer='glorot_normal',
                                      trainable=True)
                                      
        self.w2 =self.add_weight(name='w2',
                                      shape=(input_shape[1][-1], input_shape[1][-1]),
                                      initializer='glorot_normal',
                                      trainable=True)
        if self.bias:
            self.b1 = self.add_weight(name='b1',shape=(input_shape[0][-1],),
                                     initializer='zeros',
                                      trainable=True)
            self.b2 = self.add_weight(name='b2',shape=(input_shape[1][-1],),
                                     initializer='zeros',
                                      trainable=True)
        super(WAdd, self).build(input_shape)  # 一定要在最后调用它
    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None
    def call(self,x,mask=None):
        assert isinstance(x, list)
        x1,x2=x
        ot=K.tanh(K.dot(x1, self.w1)+self.b1)+K.tanh(K.dot(x2, self.w2)+self.b2)
        return ot
 
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b=input_shape
        return (shape_a[0],shape_a[1])


def match(vests):
    x1,x2=vests
    # add=x1+x2
    sub=x1-x2
    mult=x1*x2
    ks=K.abs(sub)
    # mu=x1*x2
    norm =K.l2_normalize(sub,axis=-1)
    out=K.concatenate([x1,x2,mult,ks,norm],axis=-1)
    # axis = len(x1._keras_shape)-1
    # dot = lambda a, b: K.batch_dot(a, b, axes=axis)
    return out

def match_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],shape1[1]*5)

from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape,BatchNormalization
class WordModel(Layer):
 
    def __init__(self,**kwargs):
        # self.output_dim = output_dim
        self.supports_masking = True
        super(WordModel, self).__init__(**kwargs)
    def build(self, input_shape):
        super(WordModel, self).build(input_shape)  # 一定要在最后调用它
    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None
    def call(self,x,mask=None):
        wordmodel=Lambda(match, output_shape=match_output_shape)(x)
        # merged = Dropout(0.5)(y)
        merged = BatchNormalization()(wordmodel)
        merged = Dense(128, activation='relu')(merged)

        # merged = Dropout(0.5)(merged)
        merged = BatchNormalization()(merged)
        merged = Dense(32, activation='relu')(merged)

        # merged = Dropout(0.5)(merged)
        merged = BatchNormalization()(merged)
        output = Dense(1, activation='sigmoid')(merged)
        return output
 
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b=input_shape
        return (shape_a[0],1)

