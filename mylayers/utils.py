from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Convolution1D, GlobalMaxPooling1D, GlobalAveragePooling1D, \
    Lambda, TimeDistributed, SpatialDropout1D, Reshape, RepeatVector, Bidirectional, LSTM
from keras.layers.merge import Dot, Concatenate, Multiply, Add
from keras.layers.advanced_activations import Softmax
from keras.models import Model
from keras import backend as K
from keras.initializers import RandomUniform


def max_pooling_with_mask(x, query_mask):
    # x is batch_size * |doc| * |query|
    # query_mask is batch_size * |query| (with masks as 0)
    return K.max(x, axis=1) * query_mask

def mean_pooling_with_mask(x, doc_mask, query_mask):
    # x is batch_size * |doc| * |query|
    # doc_mask is batch_size * |doc| (with masks as 0)
    # query_mask is batch_size * |query| (with masks as 0)
    ZERO_SHIFT = 0.1
    doc_mask_sum = (K.sum(doc_mask, axis=-1, keepdims=True) + ZERO_SHIFT)
    return query_mask * K.batch_dot(x, doc_mask, axes=[1, 1]) / doc_mask_sum

def add_conv_layer(input_list, layer_name, nb_filters, kernel_size, padding, dropout_rate=0.1,
                   activation='relu', strides=1, attention_level=0, conv_option="normal", prev_conv_tensors=None):
    conv_layer = Convolution1D(filters=nb_filters, kernel_size=kernel_size, padding=padding,
                               activation=activation, strides=strides, name=layer_name)
    max_pooling_layer = GlobalMaxPooling1D()
    #normlize_layer = BatchNormalization()
    dropout_layer = Dropout(dropout_rate)
    output_list, conv_output_list = [], []
    for i in range(len(input_list)):
        input = input_list[i]
        conv_tensor = conv_layer(input)
        if conv_option == "ResNet":
            conv_tensor = Add()([conv_tensor, prev_conv_tensors[i][-1]])
        #normlize_tensor = normlize_layer(conv_tensor)
        dropout_tensor = dropout_layer(conv_tensor)
        conv_pooling_tensor = max_pooling_layer(conv_tensor)
        output_list.append(dropout_tensor)
        #conv_output_list.append(conv_pooling_tensor)
        conv_output_list.append(conv_tensor)
    return output_list, conv_output_list

