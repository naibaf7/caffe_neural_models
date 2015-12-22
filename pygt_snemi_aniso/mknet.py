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

# Create the network we want
class NetConf:
    # 10 GB total memory limit
    mem_global_limit = 10 * 1024 * 1024 * 1024
    # 4 GB single buffer memory limit
    mem_buf_limit = 4 * 1024 * 1024 * 1024
    # Desired input dimensions (will select closest possible)
    input_shape = [44,132,132]
    # Desired output dimensions (will select closest posisble)
    output_shape = [16, 44, 44]
    # Number of U-Net Pooling-Convolution downsampling/upsampling steps
    unet_depth = 3
    # Number of feature maps in the start
    fmap_start = 16
    # Number of input feature maps
    fmap_input = 1
    # Number of ouput feature maps
    fmap_output = 11
    # Feature map increase rule (downsampling)
    def unet_fmap_inc_rule(self, fmaps):
        return int(math.ceil(fmaps * 3));
    # Feature map decrease rule (upsampling)
    def unet_fmap_dec_rule(self, fmaps):
        return int(math.ceil(fmaps / 3));
    # Skewed U-Net downsampling strategy
    unet_downsampling_strategy = [[1,2,2],[1,2,2],[1,2,2]]
    # Number of SK-Net Pooling-Convolution steps
    sknet_conv_depth = 0
    # Feature map increase rule
    def sknet_fmap_inc_rule(self, fmaps):
        return int(math.ceil(fmaps * 1.5));
    # Number of 1x1 (IP) Convolution steps
    sknet_ip_depth = 0
    # Feature map increase rule from SK-Convolution to IP
    def sknet_fmap_bridge_rule(self, fmaps):
        return int(math.ceil(fmaps * 4));
    # Feature map decrease rule within IP
    def sknet_fmap_dec_rule(self, fmaps):
        return int(math.ceil(fmaps / 2.5));
    # Loss function and mode ("malis", "euclid", "softmax")
    loss_function = "euclid"


netconf = NetConf()
netconf.loss_function = "euclid"
train_net_conf_euclid, test_net_conf = pygt.netgen.create_nets(netconf)
netconf.loss_function = "malis"
train_net_conf_malis, test_net_conf = pygt.netgen.create_nets(netconf)

with open('net_train_euclid.prototxt', 'w') as f:
    print(train_net_conf_euclid, file=f)
with open('net_train_malis.prototxt', 'w') as f:
    print(train_net_conf_malis, file=f)
with open('net_test.prototxt', 'w') as f:
    print(test_net_conf, file=f)
