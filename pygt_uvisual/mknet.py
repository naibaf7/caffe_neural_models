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


netconf = pygt.netgen.NetConf()

netconf.use_batchnorm = False
netconf.dropout = 0.0

netconf.fmap_start = 4

#sk1 = pygt.netgen.SKNetConf()
#sk1.sknet_conv = [[8],[6],[4]]

#sk2 = pygt.netgen.SKNetConf()
#sk2.sknet_padding = [44,44]
#sk2.sknet_conv = [[6],[4],[2]]

netconf.u_netconfs[0].unet_depth = 3
netconf.u_netconfs[0].unet_fmap_inc_rule = lambda fmaps: int(math.ceil(float(fmaps) * 1))
netconf.u_netconfs[0].unet_fmap_dec_rule = lambda fmaps: int(math.ceil(float(fmaps) / 1))
# netconf.u_netconfs[0].sk_netconfs = [None,sk1,sk2]
# netconf.u_netconfs = [netconf.u_netconfs[0],netconf.u_netconfs[0]]


netconf.loss_function = "euclid"
inshape,outshape,fmaps = pygt.netgen.compute_valid_io_shapes(netconf,pygt.netgen.caffe_pb2.TRAIN,[30,30],[200,200],constraints=[None,lambda x: x[0], lambda x: x[1]])


# We choose the maximum that still gives us 20 fmaps:
index = [n for n, i in enumerate(fmaps) if i>=4][-1]
print("Index to use: %s" % index)
# Some patching to allow our new parameters
netconf.input_shape = inshape[index]
netconf.output_shape = outshape[index]
netconf.fmap_start = 4

netconf.loss_function = "euclid"
train_net_conf_euclid, test_net_conf, train_net_tikzgraph, test_net_tikzgraph = pygt.netgen.create_nets(netconf)
netconf.loss_function = "malis"
train_net_conf_malis, test_net_conf, train_net_tikzgraph, test_net_tikzgraph = pygt.netgen.create_nets(netconf)

with open('net_train_euclid.prototxt', 'w') as f:
    print(train_net_conf_euclid, file=f)
with open('net_train_malis.prototxt', 'w') as f:
    print(train_net_conf_malis, file=f)
with open('net_test.prototxt', 'w') as f:
    print(test_net_conf, file=f)
    
with open('trainnet.tex', 'w') as f:
    print(train_net_tikzgraph, file=f)
with open('testnet.tex', 'w') as f:
    print(test_net_tikzgraph, file=f)

