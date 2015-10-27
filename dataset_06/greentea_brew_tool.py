import os, sys, inspect
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import math
import multiprocessing
from PIL import Image
from Crypto.Random.random import randint

# Load the configuration file
import config
from numpy import float32

cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
if cmd_folder not in sys.path:
    sys.path.append(cmd_folder)
    
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], config.caffe_path + "/python")))
if cmd_subfolder not in sys.path:
    sys.path.append(cmd_subfolder)

sys.path.append(config.caffe_path + "/python")

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

# Fix up OpenCL variables. Can interfere with the
# frame buffer if the GPU is also a display driver
os.environ["GPU_MAX_ALLOC_PERCENT"] = "100"
os.environ["GPU_SINGLE_ALLOC_PERCENT"] = "100"
os.environ["GPU_MAX_HEAP_SIZE"] = "100"


# General preparations
colorsr = np.random.rand(5000)
colorsg = np.random.rand(5000)
colorsb = np.random.rand(5000)

dims = len(config.output_dims)

def normalize(dataset, newmin=-1, newmax=1):
    maxval = dataset
    while len(maxval.shape) > 0:
        maxval = maxval.max(0)
    minval = dataset
    while len(minval.shape) > 0:
        minval = minval.min(0)
    return ((dataset - minval) / (maxval - minval)) * (newmax - newmin) + newmin

def error_scale(data, factor_low, factor_high):
    scale = np.add((data >= 0.5) * factor_high, (data < 0.5) * factor_low)
    return scale

def count_affinity(dataset):
    aff_high = np.sum(dataset >= 0.5)
    aff_low = np.sum(dataset < 0.5)
    return aff_high, aff_low;


def inspect_2D_hdf5(hdf5_file):
    print 'HDF5 keys: %s' % hdf5_file.keys()
    dset = hdf5_file[hdf5_file.keys()[0]]
    print 'HDF5 shape: X: %s Y: %s' % dset.shape
    print 'HDF5 data type: %s' % dset.dtype
    print 'Max/Min: %s' % [np.asarray(dset).max(0).max(0), np.asarray(dset).min(0).min(0)]

def inspect_3D_hdf5(hdf5_file):
    print 'HDF5 keys: %s' % hdf5_file.keys()
    dset = hdf5_file[hdf5_file.keys()[0]]
    print 'HDF5 shape: X: %s Y: %s Z: %s' % dset.shape
    print 'HDF5 data type: %s' % dset.dtype
    print 'Max/Min: %s' % [np.asarray(dset).max(0).max(0).max(0), np.asarray(dset).min(0).min(0).min(0)]
    
def inspect_4D_hdf5(hdf5_file):
    print 'HDF5 keys: %s' % hdf5_file.keys()
    dset = hdf5_file[hdf5_file.keys()[0]]
    print 'HDF5 shape: T: %s X: %s Y: %s Z: %s' % dset.shape
    print 'HDF5 data type: %s' % dset.dtype
    print 'Max/Min: %s' % [np.asarray(dset).max(0).max(0).max(0).max(0), np.asarray(dset).min(0).min(0).min(0).min(0)]
    
def display_raw(raw_ds, index):
    slice = raw_ds[0:raw_ds.shape[0], 0:raw_ds.shape[1], index]
    minval = np.min(np.min(slice, axis=1), axis=0)
    maxval = np.max(np.max(slice, axis=1), axis=0)   
    img = Image.fromarray((slice - minval) / (maxval - minval) * 255)
    img.show()
    
def display_con(con_ds, index):
    slice = hdf5_gt_ds[0:con_ds.shape[0], 0:con_ds.shape[1], index]
    rgbArray = np.zeros((con_ds.shape[0], con_ds.shape[1], 3), 'uint8')
    rgbArray[..., 0] = colorsr[slice] * 256
    rgbArray[..., 1] = colorsg[slice] * 256
    rgbArray[..., 2] = colorsb[slice] * 256
    img = Image.fromarray(rgbArray, 'RGB')
    img.show()
    
def display_aff(aff_ds, index):
    sliceX = aff_ds[0, 0:520, 0:520, index]
    sliceY = aff_ds[1, 0:520, 0:520, index]
    sliceZ = aff_ds[2, 0:520, 0:520, index]
    img = Image.fromarray((sliceX & sliceY & sliceZ) * 255)
    img.show()
    
def slice_data(data, offsets, sizes):
    if (len(offsets) == 1):
        return data[offsets[0]:offsets[0] + sizes[0]]
    if (len(offsets) == 2):
        return data[offsets[0]:offsets[0] + sizes[0], offsets[1]:offsets[1] + sizes[1]]
    if (len(offsets) == 3):
        return data[offsets[0]:offsets[0] + sizes[0], offsets[1]:offsets[1] + sizes[1], offsets[2]:offsets[2] + sizes[2]]
    if (len(offsets) == 4):
        return data[offsets[0]:offsets[0] + sizes[0], offsets[1]:offsets[1] + sizes[1], offsets[2]:offsets[2] + sizes[2], offsets[3]:offsets[3] + sizes[3]]
    
def train(solver, data_arrays, label_arrays, affinity_arrays, mode='euclid'):
    plt.ion()
    plt.show()

    c1, c0 = count_affinity(hdf5_aff_ds)
    print "High/low: %s" % [c1,c0]

    losses = []
    
    net = solver.net
    
    # Loop from current iteration to last iteration
    for i in range(solver.iter, solver.max_iter):
        
        # First pick the dataset to train with
        dataset = randint(0, len(data_arrays) - 1);
        data_array = data_arrays[dataset];
        label_array = label_arrays[dataset];
        affinity_array = affinity_arrays[dataset];
        
        offsets = []
        for j in range(0, dims):
            offsets.append(randint(0, data_array.shape[j] - (config.output_dims[j] + config.input_padding[j])))
        
        dummy_slice = [0]
        
        
        # These are the raw data elements
        data_slice = slice_data(data_array, offsets, [config.output_dims[di] + config.input_padding[di] for di in range(0, dims)])
        
        # These are the labels (connected components)
        label_slice = slice_data(label_array, [offsets[di] + int(math.ceil(config.input_padding[di] / float(2))) for di in range(0, dims)], config.output_dims)
        
        # These are the affinity edge values
        aff_slice = slice_data(affinity_array, [0] + [offsets[di] + int(math.ceil(config.input_padding[di] / float(2))) for di in range(0, dims)], [len(config.output_dims)] + config.output_dims)        
        
        # These are the affinity edges
        edge_slice = np.asarray([[[[-1, 0, 0], [0, -1, 0], [0, 0, -1]]]])
        
        print (data_slice[None, None, :]).shape
        print (label_slice[None, None, :]).shape
        print (aff_slice[None, :]).shape
        print (edge_slice).shape
        
        if mode == 'malis':
            net.set_input_arrays(0, np.ascontiguousarray(data_slice[None, None, :]).astype(float32), np.ascontiguousarray(dummy_slice).astype(float32))
            net.set_input_arrays(1, np.ascontiguousarray(label_slice[None, None, :]).astype(float32), np.ascontiguousarray(dummy_slice).astype(float32))
            net.set_input_arrays(2, np.ascontiguousarray(aff_slice[None, :]).astype(float32), np.ascontiguousarray(dummy_slice).astype(float32))
            net.set_input_arrays(3, np.ascontiguousarray(edge_slice).astype(float32), np.ascontiguousarray(dummy_slice).astype(float32))
            
        # We pass the raw and affinity array only
        if mode == 'euclid':
            net.set_input_arrays(0, np.ascontiguousarray(data_slice[None, None, :]).astype(float32), np.ascontiguousarray(dummy_slice).astype(float32))
            net.set_input_arrays(1, np.ascontiguousarray(aff_slice[None, :]).astype(float32), np.ascontiguousarray(dummy_slice).astype(float32))
            net.set_input_arrays(2, np.ascontiguousarray(error_scale(aff_slice[None, :],1.0,0.045)).astype(float32), np.ascontiguousarray(dummy_slice).astype(float32))

        
        # Single step
        loss = solver.step(1)
        print("Loss: %s" % loss)
        losses += [loss]
        plt.scatter(range(0, len(losses)), losses)
        plt.draw()
        

hdf5_raw_file = 'fibsem_medulla_7col/tstvol-520-1-h5/img_normalized.h5'
hdf5_gt_file = 'fibsem_medulla_7col/tstvol-520-1-h5/groundtruth_seg.h5'
hdf5_aff_file = 'fibsem_medulla_7col/tstvol-520-1-h5/groundtruth_aff.h5'

hdf5_raw = h5py.File(hdf5_raw_file, 'r')
hdf5_gt = h5py.File(hdf5_gt_file, 'r')
hdf5_aff = h5py.File(hdf5_aff_file, 'r')

#inspect_3D_hdf5(hdf5_raw)
#inspect_3D_hdf5(hdf5_gt)
#inspect_4D_hdf5(hdf5_aff)

hdf5_raw_ds = np.asarray(hdf5_raw[hdf5_raw.keys()[0]])
hdf5_gt_ds = np.asarray(hdf5_gt[hdf5_gt.keys()[0]])
hdf5_aff_ds = np.asarray(hdf5_aff[hdf5_aff.keys()[0]])

# display_aff(hdf5_aff_ds, 1)
# display_con(hdf5_gt_ds, 0)
# display_raw(hdf5_raw_ds, 0)

#Initialize caffe
caffe.set_mode_gpu()
caffe.set_device(config.device_id)
solver = caffe.get_solver(config.solver_proto)
net = solver.net


if(config.mode == "train"):
    train(solver, [normalize(hdf5_raw_ds)], [hdf5_gt_ds], [hdf5_aff_ds])
    








