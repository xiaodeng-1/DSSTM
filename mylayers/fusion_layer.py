from keras import backend as K
from keras.engine.topology import Layer
import keras
from keras.regularizers import l2
import tensorflow as tf
from keras.layers import Dense


class FusionLayer(Layer):

    def __init__(self,
                 units=32,
                 fusion_activation="relu",
                 use_bias=True,
                 **kwargs):

        super(FusionLayer, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.fusion_activation = keras.activations.get(fusion_activation)
        self.use_bias = use_bias
        self._backend = keras.backend.backend()

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        feature_dim = input_shape[0]

        self.Wf = self.add_weight(shape=(feature_dim[-1]*4, self.units),
                                  name='{}_Add_Wt'.format(self.name),
                                  initializer='glorot_normal',
                                  regularizer=l2(0.000001),
                                  trainable=True)
        if self.use_bias:
            self.bh = self.add_weight(shape=(feature_dim[1],self.units),
                                      name='{}_Add_bh'.format(self.name),
                                      initializer='zeros',
                                      regularizer=l2(0.00001),
                                      trainable=True)
        super(FusionLayer, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        #inputs分为两部分
        assert isinstance(inputs, list)
        x1,x2 = inputs
        temp_f = self.fusion(x1,x2) #(?,15,512)
        f = K.dot(temp_f, self.Wf)
        f = f+self.bh

        if self.fusion_activation is not None:
            f = self.fusion_activation(f)

        return f

    def fusion(self,x1,x2):
        # add = x1 + x2
        sub = x1 - x2
        mult = x1 * x2
        # ks = K.abs(sub)
        # norm = K.l2_normalize(sub, axis=-1)
        out = K.concatenate([x1, x2, sub,mult], axis=-1)
        return out

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0]
        return (output_shape[0],output_shape[1],self.units)


