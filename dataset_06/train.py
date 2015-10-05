import h5py
import numpy as np
import matplotlib
from PIL import Image

def inspect_3D_hdf5( hdf5_file ):
    print 'HDF5 keys: %s' % hdf5_file.keys()
    dset = hdf5_file[hdf5_file.keys()[0]]
    print 'HDF5 shape: X: %s Y: %s Z: %s' % dset.shape
    print 'HDF5 data type: %s' % dset.dtype

hdf5_raw_file = 'fibsem_medulla_7col/tstvol-520-1-h5/img_normalized.h5'
hdf5_gt_file = 'fibsem_medulla_7col/tstvol-520-1-h5/groundtruth_seg.h5'
hdf5_aff_file = 'fibsem_medulla_7col/tstvol-520-1-h5/groundtruth_aff.h5'

hdf5_raw = h5py.File(hdf5_raw_file, 'r')
hdf5_gt = h5py.File(hdf5_gt_file, 'r')
hdf5_aff = h5py.File(hdf5_aff_file, 'r')

inspect_3D_hdf5(hdf5_raw)
inspect_3D_hdf5(hdf5_gt)
#inspect_3D_hdf5(hdf5_aff)

hdf5_raw_ds = hdf5_raw[hdf5_raw.keys()[0]]
hdf5_gt_ds = hdf5_gt[hdf5_gt.keys()[0]]
hdf5_aff_ds = hdf5_aff[hdf5_aff.keys()[0]]

for i in range(0,520,50):
    slice = hdf5_raw_ds[0:520,0:520,i]
    minval = np.min(np.min(slice, axis=1), axis=0)
    maxval = np.max(np.max(slice, axis=1), axis=0)   
    img = Image.fromarray((slice-minval)/(maxval-minval)*255)
    img.show()

colorsr = np.random.rand(5000)
colorsg = np.random.rand(5000)
colorsb = np.random.rand(5000)

for i in range(0,520,50):
    slice = hdf5_gt_ds[0:520,0:520,i]
    rgbArray = np.zeros((520,520,3), 'uint8')
    rgbArray[..., 0] = colorsr[slice]*256
    rgbArray[..., 1] = colorsg[slice]*256
    rgbArray[..., 2] = colorsb[slice]*256
    img = Image.fromarray(rgbArray, 'RGB')
    img.show()


# TODO: Load data into pycaffe


