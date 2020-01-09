# Long-Short-Graph-Memory-Network
Long-Short Graph Memory Network for Skeleton-based Action Recognition（WACV2020）

Current studies have shown the effectiveness of long short-term memory network (LSTM) for skeleton-based human action recognition in capturing temporal and spatial features of the skeleton sequence. Nevertheless, it still remains challenging for LSTM to extract the latent structural dependency among nodes. In this paper, we introduce a new long-short graph memory network (LSGM) to improve the capability of LSTM to model the skeleton sequence - a type of graph data. Our proposed LSGM can learn high-level temporal-spatial features end-to-end, enabling LSTM to extract the spatial information that is neglected but intrinsic to the skeleton graph data. To improve the discriminative ability of the temporal and spatial module, we use a calibration module termed as graph temporal-spatial calibration (GTSC) to calibrate the learned temporal-spatial features. By integrating the two modules into the same framework, we obtain a stronger generalization capability in processing dynamic graph data and achieve a significant performance improvement on the NTU and SYSU dataset. Experimental results have validated the effectiveness of our proposed LSGM+GTSC model in extracting temporal and spatial information from dynamic graph data.

# Install
To run this demo, you should install these dependencies:
```
tensorflow 1.12.0
keras 2.2.4
python-lmdb 0.94
python 3.6.8
```

# Run Demo
```
python process_sysu_initial.py
python train.py
```
