from keras.optimizers import SGD
import numpy as np
import lmdb
import threading
import os
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
import LSGM_GTSC

out_dir_name = 'LSGM'

n_classes = 12
batch_size = 8
epochs = 100
nb = 20

'''
If you are using the full SYSU-3D dataset, 
then you just need to set SYSU_mini to False
'''
SYSU_mini = True
if SYSU_mini:
    samples_per_epoch = 24
    samples_per_validation = 24
else:
    samples_per_epoch = 240
    samples_per_validation = 240

data_root = "data_sysu1"
loss = 'categorical_crossentropy'
lr = 0.01
momentum = 0.9
activation = "relu"
optimizer = SGD(lr=lr, momentum=momentum, decay=0.0, nesterov=True)


class threadsafe_iter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


@threadsafe_generator
def train_datagen(augmentation=1):
    lmdb_file_train_x = os.path.join(data_root, 'Xtrain_lmdb')
    lmdb_file_train_y = os.path.join(data_root, 'Ytrain_lmdb')
    lmdb_env_x = lmdb.open(lmdb_file_train_x)
    lmdb_txn_x = lmdb_env_x.begin()
    lmdb_cursor_x = lmdb_txn_x.cursor()

    lmdb_env_y = lmdb.open(lmdb_file_train_y)
    lmdb_txn_y = lmdb_env_y.begin()
    lmdb_cursor_y = lmdb_txn_y.cursor()

    X = np.zeros((batch_size, nb, nb, 3))
    Y = np.zeros((batch_size, n_classes))
    batch_count = 0
    temp = 1
    while True:
        indices = list(range(0, samples_per_epoch))
        np.random.shuffle(indices)

        for index in indices:
            value = np.frombuffer(lmdb_cursor_x.get('{:0>8d}'.format(index).encode()))
            label = np.frombuffer(lmdb_cursor_y.get('{:0>8d}'.format(index).encode()), dtype=int)
            x = value.reshape((nb, nb, 3))
            X[batch_count] = np.reshape(x, [nb, nb, 3])
            Y[batch_count] = label
            batch_count += 1
            if batch_count == batch_size:
                ret_x = X
                ret_y = Y
                X = np.zeros((batch_size, nb, nb, 3))
                Y = np.zeros((batch_size, n_classes))
                temp += 1
                batch_count = 0
                yield (ret_x, ret_y)


@threadsafe_generator
def test_datagen():
    lmdb_file_test_x = os.path.join(data_root, 'Xtest_lmdb')
    lmdb_file_test_y = os.path.join(data_root, 'Ytest_lmdb')

    lmdb_env_x = lmdb.open(lmdb_file_test_x)
    lmdb_txn_x = lmdb_env_x.begin()
    lmdb_cursor_x = lmdb_txn_x.cursor()

    lmdb_env_y = lmdb.open(lmdb_file_test_y)
    lmdb_txn_y = lmdb_env_y.begin()
    lmdb_cursor_y = lmdb_txn_y.cursor()

    X = np.zeros((batch_size, nb, nb, 3))
    Y = np.zeros((batch_size, n_classes))
    batch_count = 0
    temp = 1
    while True:
        indices = list(range(0, samples_per_validation))
        np.random.shuffle(indices)
        batch_count = 0
        for index in indices:
            value = np.frombuffer(lmdb_cursor_x.get('{:0>8d}'.format(index).encode()))
            label = np.frombuffer(lmdb_cursor_y.get('{:0>8d}'.format(index).encode()), dtype=int)
            x = value.reshape((nb, nb, 3))
            X[batch_count] = x.reshape((nb, nb, 3))
            Y[batch_count] = label

            batch_count += 1

            if batch_count == batch_size:
                ret_x = X
                ret_y = Y
                X = np.zeros((batch_size, nb, nb, 3))
                Y = np.zeros((batch_size, n_classes))
                batch_count = 0
                temp += 1
                yield (ret_x, ret_y)


def train():
    model = LSGM_GTSC.LSGM_GTSC_builder()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    if not os.path.exists('weights/' + out_dir_name):
        os.makedirs('weights/' + out_dir_name)
    weight_path = 'weights/' + out_dir_name + '/{epoch:03d}_{val_acc:0.3f}.hdf5'

    checkpoint = ModelCheckpoint(weight_path,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.2,
                                  patience=5,
                                  verbose=1,
                                  mode='auto',
                                  cooldown=3,
                                  min_lr=0.0001)
    callbacks_list = [reduce_lr, checkpoint]
    model.fit_generator(train_datagen(),
                        steps_per_epoch=samples_per_epoch / batch_size + 1,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks_list,
                        validation_data=test_datagen(),
                        validation_steps=samples_per_validation / batch_size + 1,
                        workers=1,
                        initial_epoch=0
                        )


if __name__ == "__main__":
    train()
