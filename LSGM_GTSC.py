from __future__ import division
import six
from keras.models import Model
from keras.layers import Input, Activation, Dense, Flatten
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.layers import Lambda, Reshape, GlobalAveragePooling1D, concatenate, multiply, Dropout
from LSGM import Bi_LSGM, num_frame, feature_dim
import tensorflow as tf

dropout_rate = 0
reg = l2(1e-4)
weight_decay = 1e-4
alpha = 16
lstm_rate = 0.5


def _bn_relu(input):
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        conv = Dropout(dropout_rate)(conv)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        x = Conv2D(filters=filters, kernel_size=kernel_size,
                   strides=strides, padding=padding,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer)(activation)
        x = Dropout(dropout_rate)(x)
        return x

    return f


def _shortcut(input, residual):
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)
        shortcut = Dropout(dropout_rate)(shortcut)
    return add([shortcut, residual])


def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    def f(input):
        if is_first_block_of_first_layer:
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
            conv1 = Dropout(dropout_rate)(conv1)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=init_strides)(input)
        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return _shortcut(input, residual)

    return f


def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    def f(input):
        if is_first_block_of_first_layer:
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4))(input)
            conv_1_1 = Dropout(dropout_rate)(conv_1_1)
        else:
            conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                     strides=init_strides)(input)
        conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
        residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
        return _shortcut(input, residual)

    return f


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    ROW_AXIS = 1
    COL_AXIS = 2
    CHANNEL_AXIS = 3


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


def duplicate(x, nb_lstm):
    y = x
    for i in range(nb_lstm - 1):
        y = concatenate([y, x])
    return y


def attention(x_lstm, nb_lstm):
    '''
    Attention mechanism
    '''
    x_lstm1 = tf.transpose(x_lstm, [0, 2, 1])
    ap1 = GlobalAveragePooling1D()(x_lstm1)

    ap1 = Reshape((num_frame, 1))(ap1)
    ap2 = Lambda(duplicate, output_shape=[num_frame, nb_lstm], arguments={'nb_lstm': nb_lstm})(ap1)
    lstm1 = tf.transpose(ap2, [0, 2, 1])

    nb_fc = int(num_frame / alpha)
    fc1 = Dense(nb_fc, activation='relu')(lstm1)

    fc2 = Dense(num_frame, activation='sigmoid')(fc1)
    attention_weight = tf.transpose(fc2, [0, 2, 1])
    return attention_weight


def LSGM_attention():
    '''
    combination of attention and bi-LSGM
    '''
    global concat_axis
    concat_axis = 3
    img_input = Input(shape=(20, 20, 3), name='data')
    nb_lstm = 64  # For TARM
    lstm_f = []
    for i in range(3):
        with tf.variable_scope("lstm" + str(i + 1)):
            img_channel = Lambda(lambda x: x[:, :, :, i])(img_input)
            img_reshape = Reshape((num_frame, feature_dim))(img_channel)
            img_dense = Dense(nb_lstm)(img_reshape)
            lstm_output = Lambda(Bi_LSGM, input_shape=[num_frame, nb_lstm], output_shape=[num_frame, nb_lstm])(
                img_reshape)
            lstm_weight = Lambda(attention, arguments={'nb_lstm': nb_lstm}, output_shape=[num_frame, nb_lstm])(
                img_dense)
            lstm_output = multiply([lstm_output, lstm_weight])
            lstm = Dense(feature_dim)(lstm_output)
            lstm = add([img_reshape, lstm])
            lstm = Reshape((20, 20, 1))(lstm)
            lstm_f.append(lstm)
    x = concatenate(lstm_f, axis=-1)
    return x, img_input


def GTSC(input):
    '''
    GTSC with residual block
    '''
    repetitions = [2, 2]
    block_fn = basic_block
    block_fn = _get_block(block_fn)
    conv1 = _conv_bn_relu(filters=64, kernel_size=(5, 5), strides=(2, 2))(input)
    block = conv1
    filters = 64
    for i, r in enumerate(repetitions):
        block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
        filters *= 2
    block = _bn_relu(block)
    block_shape = K.int_shape(block)
    pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                             strides=(1, 1))(block)
    return pool2


def LSGM_GTSC_builder():
    '''
    divide into LSGM+attention model and GTSC mode
    Note that we divided RGB into three channels,
    and used 3 LSTM_GTSC models to model the data.
    '''
    num_outputs = 12
    _handle_dim_ordering()

    output, img_input = LSGM_attention()
    pool2 = GTSC(output)

    flatten1 = Flatten()(pool2)
    dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                  activation="softmax")(flatten1)
    model = Model(inputs=img_input, outputs=dense)
    return model
