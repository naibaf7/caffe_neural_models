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
test_device = 3

pygt.caffe.set_devices((test_device,))

caffemodels = pygt.getCaffeModels('net');

test_net = pygt.init_testnet(test_net_file, trained_model=caffemodels[-1][1], test_device=test_device)

hdf5_raw_file = '../dataset_06/fibsem_medulla_7col/trvol-250-1-h5/img_normalized.h5'
hdf5_raw = h5py.File(hdf5_raw_file, 'r')
hdf5_raw_ds = pygt.normalize(np.asarray(hdf5_raw[hdf5_raw.keys()[0]]).astype(float32), -1, 1)
test_dataset = {}
test_dataset['data'] = hdf5_raw_ds

pred_array = pygt.process(test_net, [test_dataset])

outhdf5 = h5py.File('test_out.h5', 'w')
outdset = outhdf5.create_dataset('main', np.shape(pred_array)[1:], np.float32, data=pred_array)
outdset.attrs['label'] = np.string_('-1,0,0;0,-1,0;0,0,-1')
outhdf5.close()