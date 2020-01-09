import tensorflow as tf
from GraphConv import gcn_layer
from train import batch_size
num_frame = 20
feature_dim = 20
hidden_size = 32
embedding_size = 4
# embedding_size is the "d", which mention in our paper


def fix_dim(lstm_output, nb_lstm):
    '''
    In order to unify the dimensions,
    we forcedly transformed the output dimensions of LSGM through a fully connected layer
    '''
    lstm_output = tf.layers.dense(units=1, inputs=lstm_output, activation=None)
    lstm_output = tf.reshape(tf.squeeze(lstm_output), shape=(batch_size, num_frame, feature_dim))
    lstm_output = tf.transpose(lstm_output, [0, 2, 1])
    lstm_output = tf.layers.dense(units=nb_lstm, inputs=lstm_output, activation=None)
    return lstm_output


def merge_states(forward_states, backward_states):
    '''
    connect hideen states of forward and reverse lsgm
    '''
    states = []
    for i in range(num_frame):
        f_s = forward_states[i]
        b_s = backward_states[-1 - i]
        state = tf.add(f_s, b_s)  # add hidden states from two lsgm
        states.append(state)
    final = tf.concat([states], axis=0)  # expand dim
    final = tf.transpose(final, [1, 0, 2])  # reshape
    return final


def sg_lsgm(input, choice):  # single direction LSGM
    '''
    Simple implementation of unidirectional LSGM
    '''
    n = input.shape[2]
    T = input.shape[1]  # T=20
    embedding = input.shape[3]  # emb
    input = tf.transpose(input, [0, 2, 1, 3])  # (None, n, t, emb)
    input = tf.reshape(input, (-1, T, embedding))  # (batch_size * n, T, emb)

    lstm_state = []
    h0 = tf.zeros((batch_size * n, hidden_size), dtype=tf.float32)
    c0 = tf.zeros_like(h0, dtype=tf.float32)
    lstm = tf.nn.rnn_cell.LSTMCell(hidden_size)
    states = tf.nn.rnn_cell.LSTMStateTuple(c=c0, h=h0)
    with tf.variable_scope("LSGM"):
        for i in range(num_frame):
            k = i
            if choice == 1:  # 1:forward , 0:backward
                k = -1 - i
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            h = states.h
            h = tf.reshape(h, shape=(batch_size, n, hidden_size))
            gcn_output = gcn_layer(input=h, units=hidden_size/2, activation=tf.nn.relu, name='gcn_layer_1', choice=1)
            gcn_output = gcn_layer(input=gcn_output, units=embedding, activation=tf.nn.relu, name='gcn_layer_2', choice=1)
            output = tf.reshape(gcn_output, shape=(-1, embedding))
            current_input = tf.multiply(output, input[:, k, :])
            lstm_output, states = lstm(current_input, states)
            lstm_state.append(states.h)
    return lstm_state


def Bi_LSGM(img_reshape):  # batch_size, t, n
    nb_lstm = 64
    img_expand = tf.expand_dims(img_reshape, axis=-1)   # expand dim in the last dimension
    img_input2lstm = tf.layers.dense(units=embedding_size, inputs=img_expand)
    with tf.variable_scope('graph_forward'):            # forward lsgm
        forward_states = sg_lsgm(img_input2lstm, 1)     # img_input2lstm(batch_size, t, n=50, 1)
    with tf.variable_scope('graph_backward'):           # return states.h
        backward_states = sg_lsgm(img_input2lstm, 0)    # 1:forward , 0:backward
    lstm_output = merge_states(forward_states, backward_states)
    lstm_output = fix_dim(lstm_output, nb_lstm)         # force trans dim
    return lstm_output
