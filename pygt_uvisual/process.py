from __future__ import print_function
import h5py
import numpy as np
from numpy import float32, int32, uint8, dtype
import sys

# Relative path to where PyGreentea resides
pygt_path = '../../PyGreentea'


import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), pygt_path))

# Other python modules
import math

# Load PyGreentea
import PyGreentea as pygt



test_net_file = 'net_test.prototxt'
test_device = 0

pygt.caffe.set_devices((test_device,))

caffemodels = pygt.getCaffeModels('net');

test_net = pygt.init_testnet(test_net_file, trained_model=caffemodels[-1][1], test_device=test_device)


# Load the datasets
hdf5_raw_file = '../dataset_06/fibsem_medulla_7col/tstvol-520-1-h5/img_normalized.h5'
hdf5_gt_file = '../dataset_06/fibsem_medulla_7col/tstvol-520-1-h5/groundtruth_seg.h5'
hdf5_aff_file = '../dataset_06/fibsem_medulla_7col/tstvol-520-1-h5/groundtruth_aff.h5'

hdf5_raw = h5py.File(hdf5_raw_file, 'r')
hdf5_gt = h5py.File(hdf5_gt_file, 'r')
hdf5_aff = h5py.File(hdf5_aff_file, 'r')

hdf5_raw_ds = pygt.normalize(np.asarray(hdf5_raw[hdf5_raw.keys()[0]]).astype(float32), -1, 1)
hdf5_gt_ds = np.asarray(hdf5_gt[hdf5_gt.keys()[0]]).astype(float32)
hdf5_aff_ds = np.asarray(hdf5_aff[hdf5_aff.keys()[0]]).astype(float32)

datasets = []
for i in range(0,1):
    dataset = {}
    dataset['data'] = hdf5_raw_ds[None, i, :]
    datasets += [dataset]


pred_array = pygt.process(test_net, datasets)

pygt.dump_tikzgraph_maps(test_net, 'dump')

outhdf5 = h5py.File('test_out.h5', 'w')
outdset = outhdf5.create_dataset('main', np.shape(pred_array)[1:], np.float32, data=pred_array)
outhdf5.close()