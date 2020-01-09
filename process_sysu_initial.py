import sys
import os
import numpy as np
import lmdb
from pathlib import Path
import scipy.io as sio
from sklearn import preprocessing


def generate_data(argv):
    matfn = u'mini_sysu.mat'
    file = sio.loadmat(matfn)
    train_data = [np.transpose(i[0], [2, 0, 1]) for i in file['train_data']]
    train_action = [int(i) for i in file['train_label'][0]]
    test_data = [np.transpose(i[0], [2, 0, 1]) for i in file['test_data']]
    test_action = [int(i) for i in file['test_label'][0]]

    nb = 20
    data_out_dir = Path("data_sysu1/")
    if not data_out_dir.exists():
        os.mkdir("data_sysu1/")

    batch_size = 1500
    lmdb_file_x = os.path.join(data_out_dir, 'Xtrain_lmdb')
    lmdb_file_y = os.path.join(data_out_dir, 'Ytrain_lmdb')

    lmdb_env_x = lmdb.open(lmdb_file_x, map_size=int(1e12))
    lmdb_env_y = lmdb.open(lmdb_file_y, map_size=int(1e12))
    lmdb_txn_x = lmdb_env_x.begin(write=True)
    lmdb_txn_y = lmdb_env_y.begin(write=True)

    lmdb_file_x1 = os.path.join(data_out_dir, 'Xtest_lmdb')
    lmdb_file_y1 = os.path.join(data_out_dir, 'Ytest_lmdb')

    lmdb_env_x1 = lmdb.open(lmdb_file_x1, map_size=int(1e12))
    lmdb_env_y1 = lmdb.open(lmdb_file_y1, map_size=int(1e12))
    lmdb_txn_x1 = lmdb_env_x1.begin(write=True)
    lmdb_txn_y1 = lmdb_env_y1.begin(write=True)
    # sk_info = {} # key: file_name, value: corresponding vid_info dict

    n_classes = 12

    feat_dim = 20

    count1 = 0
    count2 = 0
    min_max_scaler = preprocessing.MinMaxScaler()

    for m in range(len(train_data)):

        data_all = train_data[m]

        action_class = train_action[m]
        # data_all=np.array([data_all[i] for i in range(0,len(data_all),3)])
        num_frames = len(data_all)
        if num_frames == 0:
            continue

        feature1 = data_all[:, :, 0].reshape(num_frames, feat_dim)
        feature2 = data_all[:, :, 1].reshape(num_frames, feat_dim)
        feature3 = data_all[:, :, 2].reshape(num_frames, feat_dim)

        R = []
        OR = []
        feature1 = np.array([[j for j in i] for i in feature1 if np.sum(i) != 0])
        feature2 = np.array([[j for j in i] for i in feature2 if np.sum(i) != 0])
        feature3 = np.array([[j for j in i] for i in feature3 if np.sum(i) != 0])
        num_frames = np.min([len(feature1), len(feature2), len(feature3)])

        feature1_temp = np.zeros((num_frames, feat_dim))
        feature2_temp = np.zeros((num_frames, feat_dim))
        feature3_temp = np.zeros((num_frames, feat_dim))
        for n in range(0, len(feature2_temp)):
            o1 = (feature1[n][0] + feature1[n][12] + feature1[n][16]) / 3
            o2 = (feature2[n][0] + feature2[n][12] + feature2[n][16]) / 3
            o3 = (feature3[n][0] + feature3[n][12] + feature3[n][16]) / 3
            for j in range(0, feat_dim):
                jinfo = np.array((feature1[n][j] - o1, feature2[n][j] - o2, feature3[n][j] - o3))

                joint_new = jinfo

                feature1_temp[n][j] = joint_new[0]
                feature2_temp[n][j] = joint_new[1]
                feature3_temp[n][j] = joint_new[2]
        # print(file_name, R[0])

        # feature1_temp = min_max_scaler.fit_transform(feature1_temp)
        # feature2_temp = min_max_scaler.fit_transform(feature2_temp)
        # feature3_temp = min_max_scaler.fit_transform(feature3_temp)
        ''''
        feature=np.concatenate([feature1_temp,feature2_temp,feature3_temp],axis=-1)
        feature=min_max_scaler.fit_transform(feature)
        feature1_temp=feature[:,:feat_dim]
        feature2_temp = feature[:, feat_dim:2*feat_dim]
        feature3_temp = feature[:, 2*feat_dim:]
        '''
        # draw_skeleton(feature1_temp, feature2_temp, feature3_temp)
        pad1 = int((nb - num_frames) / 2)
        pad2 = nb - num_frames - pad1
        pad3 = int((nb - feat_dim) / 2)
        pad4 = nb - feat_dim - pad3
        if pad1 >= 0 and pad2 >= 0:
            feature1_new = np.lib.pad(feature1_temp, ((pad1, pad2), (pad3, pad4)), 'constant',
                                      constant_values=0).reshape(nb, nb, 1)
            feature2_new = np.lib.pad(feature2_temp, ((pad1, pad2), (pad3, pad4)), 'constant',
                                      constant_values=0).reshape(nb, nb, 1)
            feature3_new = np.lib.pad(feature3_temp, ((pad1, pad2), (pad3, pad4)), 'constant',
                                      constant_values=0).reshape(nb, nb, 1)
        else:
            feature1_index = np.linspace(0, num_frames - 1, nb, dtype=int)

            feature1_new = np.lib.pad(feature1_temp[feature1_index], ((0, 0), (pad3, pad4)), 'constant',
                                      constant_values=0).reshape(nb, nb, 1)
            feature2_new = np.lib.pad(feature2_temp[feature1_index], ((0, 0), (pad3, pad4)), 'constant',
                                      constant_values=0).reshape(nb, nb, 1)
            feature3_new = np.lib.pad(feature3_temp[feature1_index], ((0, 0), (pad3, pad4)), 'constant',
                                      constant_values=0).reshape(nb, nb, 1)

        feature = np.concatenate([feature1_new, feature2_new, feature3_new], axis=2)
        X = feature
        Y = np.zeros(n_classes, dtype=int)
        Y[action_class-1] = 1
        #Y = np_utils.to_categorical(action_class - 1, n_classes, dtype=np.int32)

        keystr = '{:0>8d}'.format(count1)
        count1 += 1
        lmdb_txn_x.put(keystr.encode(), X.tobytes())
        lmdb_txn_y.put(keystr.encode(), Y.tobytes())

        print(count1, action_class)
        if count1 % batch_size == 0:
            lmdb_txn_x.commit()
            lmdb_txn_x = lmdb_env_x.begin(write=True)
            lmdb_txn_y.commit()
            lmdb_txn_y = lmdb_env_y.begin(write=True)

    for m in range(len(test_data)):

        data_all = test_data[m]

        action_class = test_action[m]
        # data_all=np.array([data_all[i] for i in range(0,len(data_all),3)])
        num_frames = len(data_all)
        if num_frames == 0:
            continue

        feature1 = data_all[:, :, 0].reshape(num_frames, feat_dim)
        feature2 = data_all[:, :, 1].reshape(num_frames, feat_dim)
        feature3 = data_all[:, :, 2].reshape(num_frames, feat_dim)

        R = []
        OR = []
        feature1 = np.array([[j for j in i] for i in feature1 if np.sum(i) != 0])
        feature2 = np.array([[j for j in i] for i in feature2 if np.sum(i) != 0])
        feature3 = np.array([[j for j in i] for i in feature3 if np.sum(i) != 0])
        num_frames = np.min([len(feature1), len(feature2), len(feature3)])

        feature1_temp = np.zeros((num_frames, feat_dim))
        feature2_temp = np.zeros((num_frames, feat_dim))
        feature3_temp = np.zeros((num_frames, feat_dim))
        for n in range(0, len(feature2_temp)):
            o1 = (feature1[n][0] + feature1[n][12] + feature1[n][16]) / 3
            o2 = (feature2[n][0] + feature2[n][12] + feature2[n][16]) / 3
            o3 = (feature3[n][0] + feature3[n][12] + feature3[n][16]) / 3
            for j in range(0, feat_dim):
                jinfo = np.array((feature1[n][j] - o1, feature2[n][j] - o2, feature3[n][j] - o3))

                joint_new = jinfo

                feature1_temp[n][j] = joint_new[0]
                feature2_temp[n][j] = joint_new[1]
                feature3_temp[n][j] = joint_new[2]
        # print(file_name, R[0])

        # feature1_temp = min_max_scaler.fit_transform(feature1_temp)
        # feature2_temp = min_max_scaler.fit_transform(feature2_temp)
        # feature3_temp = min_max_scaler.fit_transform(feature3_temp)
        ''''
        feature=np.concatenate([feature1_temp,feature2_temp,feature3_temp],axis=-1)
        feature=min_max_scaler.fit_transform(feature)
        feature1_temp=feature[:,:feat_dim]
        feature2_temp = feature[:, feat_dim:2*feat_dim]
        feature3_temp = feature[:, 2*feat_dim:]
        '''
        # draw_skeleton(feature1_temp, feature2_temp, feature3_temp)
        pad1 = int((nb - num_frames) / 2)
        pad2 = nb - num_frames - pad1
        pad3 = int((nb - feat_dim) / 2)
        pad4 = nb - feat_dim - pad3
        if pad1 >= 0 and pad2 >= 0:
            feature1_new = np.lib.pad(feature1_temp, ((pad1, pad2), (pad3, pad4)), 'constant',
                                      constant_values=0).reshape(
                nb, nb, 1)
            feature2_new = np.lib.pad(feature2_temp, ((pad1, pad2), (pad3, pad4)), 'constant',
                                      constant_values=0).reshape(nb, nb, 1)
            feature3_new = np.lib.pad(feature3_temp, ((pad1, pad2), (pad3, pad4)), 'constant',
                                      constant_values=0).reshape(nb, nb, 1)
        else:
            feature1_index = np.linspace(0, num_frames - 1, nb, dtype=int)

            feature1_new = np.lib.pad(feature1_temp[feature1_index], ((0, 0), (pad3, pad4)), 'constant',
                                      constant_values=0).reshape(nb, nb, 1)
            feature2_new = np.lib.pad(feature2_temp[feature1_index], ((0, 0), (pad3, pad4)), 'constant',
                                      constant_values=0).reshape(nb, nb, 1)
            feature3_new = np.lib.pad(feature3_temp[feature1_index], ((0, 0), (pad3, pad4)), 'constant',
                                      constant_values=0).reshape(nb, nb, 1)
        feature = np.concatenate([feature1_new, feature2_new, feature3_new], axis=2)
        X = feature
        Y = np.zeros(n_classes, dtype=int)
        Y[action_class - 1] = 1
        #Y = np_utils.to_categorical(action_class - 1, n_classes)

        keystr = '{:0>8d}'.format(count2)
        count2 += 1
        lmdb_txn_x1.put(keystr.encode(), X.tobytes())
        lmdb_txn_y1.put(keystr.encode(), Y.tobytes())

        print(count2, action_class)
        if count2 % batch_size == 0:
            lmdb_txn_x1.commit()
            lmdb_txn_x1 = lmdb_env_x1.begin(write=True)
            lmdb_txn_y1.commit()
            lmdb_txn_y1 = lmdb_env_y1.begin(write=True)

    ## END FILE LOOP

    print("Writing out data . . . ")

    # pdb.set_trace()

    # write last batch
    if count1 % batch_size != 0:
        lmdb_txn_x.commit()
        lmdb_txn_y.commit()
        print('last batch')
    if count2 % batch_size != 0:
        lmdb_txn_x1.commit()
        lmdb_txn_y1.commit()
        print('last batch')

    print("SAMPLES: ", count1, count2)


if __name__ == "__main__":
    generate_data(sys.argv)
