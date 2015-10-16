from __future__ import print_function
import os, sys, inspect
import h5py
import numpy as np
import matplotlib
import random
import math
import multiprocessing
from PIL import Image
from Crypto.Random.random import randint

# Load the configuration file
import config

cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
if cmd_folder not in sys.path:
    sys.path.append(cmd_folder)
    
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],config.caffe_path+"/python")))
if cmd_subfolder not in sys.path:
    sys.path.append(cmd_subfolder)

sys.path.append(config.caffe_path+"/python")

# Ensure correct compilation of Caffe and Pycaffe
if config.library_compile:
    cpus = multiprocessing.cpu_count()
    cwd = os.getcwd()
    os.chdir(config.caffe_path)
    result = os.system("make all -j %s" % cpus)
    if result != 0:
        sys.exit(result)
    result = os.system("make pycaffe -j %s" % cpus)
    if result != 0:
        sys.exit(result)
    os.chdir(cwd)

# Import pycaffe
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2

import netconf

# General variables
# Size of a float variable
fsize = 4

def data_layer(shape):
    data, label = L.MemoryData(dim=shape, ntop=2)
    return data, label

def conv_relu(bottom, num_output, kernel_size=[3], stride=[1], pad=[0], kstride=[1], group=1, weight_std=0.01):
    conv = L.Convolution(bottom, kernel_size=kernel_size, stride=stride, kstride=kstride,
                                num_output=num_output, pad=pad, group=group,
                                param=[dict(lr_mult=1),dict(lr_mult=2)],
                                weight_filler=dict(type='gaussian', std=weight_std),
                                bias_filler=dict(type='constant'))
    return conv, L.ReLU(conv, in_place=True)

def convolution(bottom, num_output, kernel_size=[3], stride=[1], pad=[0], kstride=[1], group=1, weight_std=0.01):
    return L.Convolution(bottom, kernel_size=kernel_size, stride=stride, kstride=kstride,
                                num_output=num_output, pad=pad, group=group,
                                param=[dict(lr_mult=1),dict(lr_mult=2)],
                                weight_filler=dict(type='gaussian', std=weight_std),
                                bias_filler=dict(type='constant'))

def max_pool(bottom, kernel_size=[2], stride=[2], pad=[0], kstride=[1]):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=kernel_size, stride=stride, pad=pad, kstride=kstride)

def upconv(bottom, num_output_dec, num_output_conv, weight_std=0.01):
    deconv = L.Deconvolution(bottom, convolution_param=dict(num_output=num_output_dec, kernel_size=[2], stride=[2], pad=[0], kstride=[1], group=num_output_dec,
                                                            weight_filler=dict(type='constant', value=1), bias_term=False),
                             param=dict(lr_mult=0, decay_mult=0))
    conv = L.Convolution(deconv, num_output=num_output_conv, kernel_size=[1], stride=[1], pad=[0], kstride=[1], group=1,
                            weight_filler=dict(type='gaussian', std=weight_std),
                            bias_filler=dict(type='constant'))
    return deconv, conv

def mergecrop(bottom_a, bottom_b):
    return L.Mergecrop(bottom_a, bottom_b)

def implement_usknet(net):
    # Chained blob list to construct the network (forward direction)
    blobs = []

    # All networks start with data
    blobs = blobs + [net.data]

    if netconf.unet_depth > 0:
        # U-Net downsampling; 2*Convolution+Pooling
        for i in range(0, netconf.unet_depth):
            conv, relu = conv_relu(blobs[len(blobs)-1], 8, kernel_size=[3])
            blobs = blobs + [relu]
            conv, relu = conv_relu(blobs[len(blobs)-1], 8, kernel_size=[3])
            blobs = blobs + [relu]  # This is the blob of interest for mergecrop (index 2 + 3 * i)
            pool = max_pool(blobs[len(blobs)-1])
            blobs = blobs + [pool]
    
    # If there is no SK-Net component, fill with 2 convolutions
    if (netconf.unet_depth > 0 and netconf.sknet_conv_depth == 0):
        conv, relu = conv_relu(blobs[len(blobs)-1], 8, kernel_size=[3])
        blobs = blobs + [relu]
        conv, relu = conv_relu(blobs[len(blobs)-1], 8, kernel_size=[3])
        blobs = blobs + [relu]
    # Else use the SK-Net instead
    else:
        for i in range(0, netconf.sknet_conv_depth):
            run_shape = run_shape
    
    if netconf.unet_depth > 0:
        # U-Net upsampling; Upconvolution+MergeCrop+2*Convolution
        for i in range(0, netconf.unet_depth):
            deconv, conv = upconv(blobs[len(blobs)-1], 8, 8);
            blobs = blobs + [conv]
            # Here, layer (2 + 3 * i) with reversed i (high to low) is picked
            mergec = mergecrop(blobs[len(blobs)-1], blobs[-1 + 3 * (netconf.unet_depth - i)])
            blobs = blobs + [mergec]
            conv, relu = conv_relu(blobs[len(blobs)-1], 8, kernel_size=[3])
            blobs = blobs + [relu]
            conv, relu = conv_relu(blobs[len(blobs)-1], 8, kernel_size=[3])
            blobs = blobs + [relu]
            
        conv = convolution(blobs[len(blobs)-1], 3, kernel_size=[1])
        blobs = blobs + [conv]
    
    # Return the last blob of the network (goes to error objective)
    return blobs[len(blobs)-1]

def caffenet(netmode):
    # Start Caffe proto net
    net = caffe.NetSpec()
    # Specify input data structures
    
    if netmode == caffe_pb2.TEST:
        net.data, net.datai = data_layer([1,1,122,122,122])
        net.silence = L.Silence(net.datai, ntop=0)
        last_blob = implement_usknet(net)
        net.prob = L.Softmax(last_blob, ntop=1)
    else:
        net.data, net.datai = data_layer([1,1,122,122,122])
        net.label, net.labeli = data_layer([1,1,32,32,32])
        if netconf.loss_function == 'affinity':
            net.label_affinity, net.label_affinityi = data_layer([1,3,32,32,32])
            net.affinity_edges, net.affinity_edgesi = data_layer([1,3,3])
            
        # Silence the inputs that are not useful
        if netconf.loss_function == 'affinity':
            net.silence = L.Silence(net.datai, net.labeli, net.label_affinityi, net.affinity_edgesi, ntop=0)
        else:
            net.silence = L.Silence(net.datai, net.labeli, ntop=0)
    
        # Start the actual network
        last_blob = implement_usknet(net)
        
        # Implement the loss
        if netconf.loss_function == 'affinity':
            net.prob = L.Softmax(last_blob, ntop=1)
            net.loss = L.Affinity(net.prob, net.label, net.label_affinity, net.affinity_edges, ntop=0)
        
        if netconf.loss_function == 'softmax':
            net.loss = L.SoftmaxWithLoss(last_blob, net.label)
    
    # Return the protocol buffer of the generated network
    return net.to_proto()

def make_net():
    with open('net/net.prototxt', 'w') as f:
        print(caffenet(caffe_pb2.TRAIN), file=f)


make_net()   
