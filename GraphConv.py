import tensorflow as tf
import numpy as np
import scipy.io as sio
cn = sio.loadmat('connectM.mat')['connectM']


def load_adjacent():
    '''
    Here we load the adjacency matrix of the sysu dataset
    follow the simple traditional spatial-GCN formula
    we implement normalization
    '''
    A = np.transpose(cn) + cn + np.eye(20, dtype=np.int32)
    # 20:the number of nodes in sysu_dataset
    D = np.array(np.sum(A, axis=0))
    D_ = np.diag(D ** (-0.5))
    A_ = np.dot(np.dot(D_, A), D_)
    batch_A = A_
    batch_A = tf.convert_to_tensor(batch_A, dtype=tf.float32)
    return batch_A


def gcn_layer(input, units, activation=None, name=None, choice=0):
    EV = load_adjacent()
    '''
    Here we have implemented two gcn layers, 
    one of which(choice = 1) is convenient for our model to call directly,
    and the other(choice = 0) is a traditional GCN implementation
    '''
    '''
    Through coordinate transformation, 
    GCN layer is easier to implement with large batches of data
    '''
    if choice:  # input shape (batch_size, n, hidden_size)  output shape (batch_size, n, units)
        batch_size = input.shape[0]
        n = input.shape[1]
        hidden_size = input.shape[2]
        h = tf.transpose(input, [0, 2, 1])  # (batch_size, hidden_size, n)
        h = tf.reshape(h, shape=(-1, n))
        out = tf.matmul(h, EV)
        out = tf.reshape(out, shape=(batch_size, hidden_size, n))
        out = tf.transpose(out, [0, 2, 1])
        output = tf.layers.dense(inputs=out, units=units, activation=activation, name=name)

    else:  # input shape (batch_size, t, n, 3)
        n = input.shape[2]
        batch_size = input.shape[0]
        t = input.shape[1]
        x_trans = tf.transpose(input, [0, 1, 3, 2])
        # shape  batch_size, t, 3, n
        x_reshape = tf.reshape(x_trans, shape=(-1, n))
        temp = tf.layers.dense(units=units, inputs=tf.matmul(x_reshape, EV), activation=activation, name=name)
        temp = tf.reshape(temp, shape=(batch_size, t, 3, n))
        output = tf.transpose(temp, [0, 1, 3, 2])  # batch_size, t, n, 3
    return output
