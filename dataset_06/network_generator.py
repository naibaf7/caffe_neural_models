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

def compute_memory_weights(shape_arr):
    memory = 0
    for i in range(0,len(shape_arr)):
        memory += shape_arr[i][1]
    return memory
        
def compute_memory_buffers(shape_arr):
    memory = 0
    for i in range(0,len(shape_arr)):
        memory = max(memory, shape_arr[i][0])
    return memory
        
def compute_memory_blobs(shape_arr):
    memory = 0
    for i in range(0,len(shape_arr)):
        mem = fsize * shape_arr[i][2]
        for j in range(0,len(shape_arr[i][4])):
            mem *= shape_arr[i][4][j]
        memory += mem
    return memory

def update_shape(shape_arr, update):
    last_shape = shape_arr[len(shape_arr)-1]
    new_shape = [update[0](last_shape[0]), update[1](last_shape[1]), update[2](last_shape[2]),
                 [update[3][min(i,len(update[3])-1)](last_shape[3][i]) for i in range(0,len(last_shape[3]))],
                 [update[4][min(i,len(update[4])-1)](last_shape[4][i]) for i in range(0,len(last_shape[4]))]]
    shape_arr += [new_shape]
    return shape_arr

def data_layer(shape):
    data, label = L.MemoryData(dim=shape, ntop=2)
    return data, label

def conv_relu(run_shape, bottom, num_output, kernel_size=[3], stride=[1], pad=[0], kstride=[1], group=1, weight_std=0.01):
    # The convolution buffer and weight memory
    weight_mem = fsize * num_output * run_shape[-1][2]
    conv_buff = fsize * run_shape[-1][2]
    for i in range(0,len(run_shape[-1][4])):        
        conv_buff *= kernel_size[min(i,len(kernel_size)-1)]
        conv_buff *= run_shape[-1][4][i]
        weight_mem *= kernel_size[min(i,len(kernel_size)-1)]
    
    # Shape update rules
    update =  [lambda x: conv_buff, lambda x: weight_mem, lambda x: num_output]
    update += [[lambda x: x, lambda x: x, lambda x: x]]
    update += [[lambda x: x - (kernel_size[min(i,len(kernel_size)-1)] - 1) * (run_shape[-1][3][i]) for i in range(0,len(run_shape[-1][4]))]]
    update_shape(run_shape, update)
    
    conv = L.Convolution(bottom, kernel_size=kernel_size, stride=stride, kstride=kstride,
                                num_output=num_output, pad=pad, group=group,
                                param=[dict(lr_mult=1),dict(lr_mult=2)],
                                weight_filler=dict(type='gaussian', std=weight_std),
                                bias_filler=dict(type='constant'))
    return conv, L.ReLU(conv, in_place=True)

def convolution(run_shape, bottom, num_output, kernel_size=[3], stride=[1], pad=[0], kstride=[1], group=1, weight_std=0.01):
    # The convolution buffer and weight memory
    weight_mem = fsize * num_output * run_shape[-1][2]
    conv_buff = fsize * run_shape[-1][2]
    for i in range(0,len(run_shape[-1][4])):        
        conv_buff *= kernel_size[min(i,len(kernel_size)-1)]
        conv_buff *= run_shape[-1][4][i]
        weight_mem *= kernel_size[min(i,len(kernel_size)-1)]
    
    # Shape update rules
    update =  [lambda x: conv_buff, lambda x: weight_mem, lambda x: num_output]
    update += [[lambda x: x, lambda x: x, lambda x: x]]
    update += [[lambda x: x - (kernel_size[min(i,len(kernel_size)-1)] - 1) * (run_shape[-1][3][i]) for i in range(0,len(run_shape[-1][4]))]]
    update_shape(run_shape, update)
    
    return L.Convolution(bottom, kernel_size=kernel_size, stride=stride, kstride=kstride,
                                num_output=num_output, pad=pad, group=group,
                                param=[dict(lr_mult=1),dict(lr_mult=2)],
                                weight_filler=dict(type='gaussian', std=weight_std),
                                bias_filler=dict(type='constant'))

def max_pool(run_shape, bottom, kernel_size=[2], stride=[2], pad=[0], kstride=[1]): 
    # Shape update rules
    update =  [lambda x: 0, lambda x: 0, lambda x: x]
    update += [[lambda x: x * kstride[min(i,len(kstride)-1)] for i in range(0,len(run_shape[-1][4]))]]
    # Strictly speaking this update rule is not complete, but should be sufficient for USK
    if kstride[0] == 1 and kernel_size[0] == stride[0]:
        update += [[lambda x: x / (kernel_size[min(i,len(kernel_size)-1)]) for i in range(0,len(run_shape[-1][4]))]]
    else:
        update += [[lambda x: x - (kernel_size[min(i,len(kernel_size)-1)] - 1) * (run_shape[-1][3][i]) for i in range(0,len(run_shape[-1][4]))]]
    update_shape(run_shape, update)

    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=kernel_size, stride=stride, pad=pad, kstride=kstride)

def upconv(run_shape, bottom, num_output_dec, num_output_conv, weight_std=0.01):
    
    # Shape update rules
    update =  [lambda x: 0, lambda x: 0, lambda x: num_output_dec]
    update += [[lambda x: x, lambda x: x, lambda x: x]]
    update += [[lambda x: 2 * x for i in range(0,len(run_shape[-1][4]))]]
    update_shape(run_shape, update)
    
    deconv = L.Deconvolution(bottom, convolution_param=dict(num_output=num_output_dec, kernel_size=[2], stride=[2], pad=[0], kstride=[1], group=num_output_dec,
                                                            weight_filler=dict(type='constant', value=1), bias_term=False),
                             param=dict(lr_mult=0, decay_mult=0))

    # The convolution buffer and weight memory
    weight_mem = fsize * num_output_conv * num_output_dec
    conv_buff = fsize * run_shape[-1][2]
    for i in range(0,len(run_shape[-1][4])):        
        conv_buff *= 2
        conv_buff *= run_shape[-1][4][i]
    
    # Shape update rules
    update =  [lambda x: conv_buff, lambda x: weight_mem, lambda x: num_output_conv]
    update += [[lambda x: x, lambda x: x, lambda x: x]]
    update += [[lambda x: x for i in range(0,len(run_shape[-1][4]))]]
    update_shape(run_shape, update)

    conv = L.Convolution(deconv, num_output=num_output_conv, kernel_size=[1], stride=[1], pad=[0], kstride=[1], group=1,
                            param=[dict(lr_mult=1),dict(lr_mult=2)],
                            weight_filler=dict(type='gaussian', std=weight_std),
                            bias_filler=dict(type='constant'))
    return deconv, conv

def mergecrop(run_shape, bottom_a, bottom_b):

    # Shape update rules
    update =  [lambda x: 0, lambda x: 0, lambda x: 2*x]
    update += [[lambda x: x, lambda x: x, lambda x: x]]
    update += [[lambda x: x for i in range(0,len(run_shape[-1][4]))]]
    update_shape(run_shape, update)

    return L.MergeCrop(bottom_a, bottom_b)

def implement_usknet(net, run_shape, fmaps_start, fmaps_end):
    # Chained blob list to construct the network (forward direction)
    blobs = []

    # All networks start with data
    blobs = blobs + [net.data]
    
    fmaps = fmaps_start

    if netconf.unet_depth > 0:
        # U-Net downsampling; 2*Convolution+Pooling
        for i in range(0, netconf.unet_depth):
            conv, relu = conv_relu(run_shape, blobs[-1], fmaps, kernel_size=[3], weight_std=math.sqrt(2.0/float(run_shape[-1][2]*pow(3,len(run_shape[-1][4])))))
            blobs = blobs + [relu]
            conv, relu = conv_relu(run_shape, blobs[-1], fmaps, kernel_size=[3], weight_std=math.sqrt(2.0/float(run_shape[-1][2]*pow(3,len(run_shape[-1][4])))))
            blobs = blobs + [relu]  # This is the blob of interest for mergecrop (index 2 + 3 * i)
            pool = max_pool(run_shape, blobs[-1])
            blobs = blobs + [pool]
            fmaps = netconf.unet_fmap_inc_rule(fmaps)

    
    # If there is no SK-Net component, fill with 2 convolutions
    if (netconf.unet_depth > 0 and netconf.sknet_conv_depth == 0):
        conv, relu = conv_relu(run_shape, blobs[-1], fmaps, kernel_size=[3], weight_std=math.sqrt(2.0/float(run_shape[-1][2]*pow(3,len(run_shape[-1][4])))))
        blobs = blobs + [relu]
        conv, relu = conv_relu(run_shape, blobs[-1], fmaps, kernel_size=[3], weight_std=math.sqrt(2.0/float(run_shape[-1][2]*pow(3,len(run_shape[-1][4])))))
        blobs = blobs + [relu]
    # Else use the SK-Net instead
    else:
        for i in range(0, netconf.sknet_conv_depth):
            # TODO: Not implemented yet (fixme)
            run_shape = run_shape
    
    if netconf.unet_depth > 0:
        # U-Net upsampling; Upconvolution+MergeCrop+2*Convolution
        for i in range(0, netconf.unet_depth):
            deconv, conv = upconv(run_shape, blobs[-1], fmaps, netconf.unet_fmap_dec_rule(fmaps), weight_std=math.sqrt(2.0/float(run_shape[-1][2]*pow(3,len(run_shape[-1][4])))))
            blobs = blobs + [conv]
            fmaps = netconf.unet_fmap_dec_rule(fmaps)
            # Here, layer (2 + 3 * i) with reversed i (high to low) is picked
            mergec = mergecrop(run_shape, blobs[-1], blobs[-1 + 3 * (netconf.unet_depth - i)])
            blobs = blobs + [mergec]
            conv, relu = conv_relu(run_shape, blobs[-1], fmaps, kernel_size=[3], weight_std=math.sqrt(2.0/float(run_shape[-1][2]*pow(3,len(run_shape[-1][4])))))
            blobs = blobs + [relu]
            conv, relu = conv_relu(run_shape, blobs[-1], fmaps, kernel_size=[3], weight_std=math.sqrt(2.0/float(run_shape[-1][2]*pow(3,len(run_shape[-1][4])))))
            blobs = blobs + [relu]
            
        conv = convolution(run_shape, blobs[-1], fmaps_end, kernel_size=[1], weight_std=math.sqrt(2.0/float(run_shape[-1][2]*pow(3,len(run_shape[-1][4])))))
        blobs = blobs + [conv]
    
    # Return the last blob of the network (goes to error objective)
    return blobs[-1]

def caffenet(netmode):
    # Start Caffe proto net
    net = caffe.NetSpec()
    # Specify input data structures
    
    if netmode == caffe_pb2.TEST:
        if netconf.loss_function == 'malis':
            fmaps_end = 3
            
        if netconf.loss_function == 'euclid':
            fmaps_end = 3

        if netconf.loss_function == 'softmax':
            fmaps_end = 2

        net.data, net.datai = data_layer([1,1,132,132,132])
        net.silence = L.Silence(net.datai, ntop=0)
        
        # Shape specs:
        # 00.    Convolution buffer size
        # 01.    Weight memory size
        # 03.    Num. channels
        # 04.    [d] parameter running value
        # 05.    [w] parameter running value
        run_shape_in = [[0,0,1,[1,1,1],[132,132,132]]]
        run_shape_out = run_shape_in
        
        last_blob = implement_usknet(net, run_shape_out, 24, fmaps_end)

        # Implement the prediction layer
        if netconf.loss_function == 'malis':
            net.prob = L.Sigmoid(last_blob, ntop=1)
        
        if netconf.loss_function == 'euclid':
            net.prob = L.Sigmoid(last_blob, ntop=1)
            
        if netconf.loss_function == 'softmax':
            net.prob = L.Softmax(last_blob, ntop=1)

        for i in range(0,len(run_shape_out)):
            print(run_shape_out[i])
            
        print("Max. memory requirements: %s B" % (compute_memory_buffers(run_shape_out)+compute_memory_weights(run_shape_out)+compute_memory_blobs(run_shape_out)))
        print("Weight memory: %s B" % compute_memory_weights(run_shape_out))
        print("Max. conv buffer: %s B" % compute_memory_buffers(run_shape_out))
        
    else:
        if netconf.loss_function == 'malis':
            net.data, net.datai = data_layer([1,1,132,132,132])
            net.label, net.labeli = data_layer([1,1,44,44,44])
            net.label_affinity, net.label_affinityi = data_layer([1,3,44,44,44])
            net.affinity_edges, net.affinity_edgesi = data_layer([1,1,3,3])
            net.silence = L.Silence(net.datai, net.labeli, net.label_affinityi, net.affinity_edgesi, ntop=0)
            fmaps_end = 3
            
        if netconf.loss_function == 'euclid':
            net.data, net.datai = data_layer([1,1,132,132,132])
            net.label, net.labeli = data_layer([1,3,44,44,44])
            net.scale, net.scalei = data_layer([1,3,44,44,44])
            net.silence = L.Silence(net.datai, net.labeli, net.scalei, ntop=0)
            fmaps_end = 3

        if netconf.loss_function == 'softmax':
            net.data, net.datai = data_layer([1,1,132,132,132])
            # Currently only supports binary classification
            net.label, net.labeli = data_layer([1,1,44,44,44])
            net.silence = L.Silence(net.datai, net.labeli, ntop=0)
            fmaps_end = 2
    
        run_shape_in = [[0,1,1,[1,1,1],[132,132,132]]]
        run_shape_out = run_shape_in
    
        # Start the actual network
        last_blob = implement_usknet(net, run_shape_out, 24, fmaps_end)
        
        for i in range(0,len(run_shape_out)):
            print(run_shape_out[i])
            
        print("Max. memory requirements: %s B" % (compute_memory_buffers(run_shape_out)+compute_memory_weights(run_shape_out)+2*compute_memory_blobs(run_shape_out)))
        print("Weight memory: %s B" % compute_memory_weights(run_shape_out))
        print("Max. conv buffer: %s B" % compute_memory_buffers(run_shape_out))
        
        # Implement the loss
        if netconf.loss_function == 'malis':       
            last_blob = L.Sigmoid(last_blob, in_place=True)
            net.loss = L.MalisLoss(last_blob, net.label_affinity, net.label, net.affinity_edges, ntop=0)
        
        if netconf.loss_function == 'euclid':
            last_blob = L.Sigmoid(last_blob, in_place=True)
            net.loss = L.EuclideanLoss(last_blob, net.label, net.scale, ntop=0)
            
        if netconf.loss_function == 'softmax':
            net.loss = L.SoftmaxWithLoss(last_blob, net.label, ntop=0)
            
    # Return the protocol buffer of the generated network
    return net.to_proto()

def make_net():
    with open('net/net_train.prototxt', 'w') as f:
        print(caffenet(caffe_pb2.TRAIN), file=f)
    with open('net/net_test.prototxt', 'w') as f:
        print(caffenet(caffe_pb2.TEST), file=f)

def make_solver():
    with open('net/solver.prototxt', 'w') as f:
        print('train_net: \"net/net_train.prototxt\"', file=f)
        print('base_lr: 0.00005', file=f)
        print('momentum: 0.99', file=f)
        print('weight_decay: 0.000005', file=f)
        print('lr_policy: \"inv\"', file=f)
        print('gamma: 0.0001', file=f)
        print('power: 0.75', file=f)
        print('max_iter: 100000', file=f)
        print('snapshot: 2000', file=f)
        print('snapshot_prefix: \"net_\"', file=f)
        print('display: 50', file=f)

make_net()
make_solver()


