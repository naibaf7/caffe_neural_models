import numpy as np
import malis as m
import h5py
np.set_printoptions(precision=4)

print "Can we make the `nhood' for an isotropic 3d dataset"
print "corresponding to a 6-connected neighborhood?"
nhood = m.mknhood3d(1)
print nhood

print "Can we make the `nhood' for an anisotropic 3d dataset"
print "corresponding to a 4-connected neighborhood in-plane"
print "and 26-connected neighborhood in the previous z-plane?"
nhood2 = m.mknhood3d_aniso(1,1.8)
print nhood2

segTrue = np.array([0, 1, 1, 1, 2, 2, 0, 5, 5, 5, 5],dtype=np.int32);
node1 = np.arange(segTrue.shape[0]-1,dtype=np.int32)
node2 = np.arange(1,segTrue.shape[0],dtype=np.int32)
nVert = segTrue.shape[0]
edgeWeight = np.array([0, 1, 2, 0, 2, 0, 0, 1, 2, 2.5],dtype=np.float32);
edgeWeight = edgeWeight/edgeWeight.max()
print segTrue
print edgeWeight

nPairPos = m.malis_loss_weights(segTrue, node1, node2, edgeWeight, 1)
nPairNeg = m.malis_loss_weights(segTrue, node1, node2, edgeWeight, 0)
print np.vstack((nPairPos,nPairNeg))
# print nPairNeg

idxkeep = (edgeWeight > 0).astype(np.int32)
cc = m.connected_components(nVert,node1,node2,idxkeep)
print cc


# node1, node2 = m.nodelist_like((2,3,4),-np.eye(3))
# print node1
# print node2

datadir = '/groups/turaga/turagalab/greentea/project_data/dataset_06/fibsem_medulla_7col/trvol-250-1-h5/'
print "Reading test volume from " + datadir
# hdf5_raw_file = datadir + 'img_normalized.h5'
hdf5_gt_file = datadir + 'groundtruth_seg.h5'
# hdf5_aff_file = datadir + 'groundtruth_aff.h5'

#hdf5_raw_file = 'zebrafish_friedrich/raw.hdf5'
#hdf5_gt_file = 'zebrafish_friedrich/labels_2.hdf5'


# hdf5_raw = h5py.File(hdf5_raw_file, 'r')
h5seg = h5py.File(hdf5_gt_file, 'r')
# hdf5_aff = h5py.File(hdf5_aff_file, 'r')

seg = np.asarray(h5seg['main']).astype(np.int32)
aff = m.seg_to_affgraph(seg,nhood)
cc,ccSizes = m.connected_components_affgraph(aff,nhood)
aff2 = m.seg_to_affgraph(cc,nhood)
cc2,ccSizes2 = m.connected_components_affgraph(aff2,nhood)

print "Comparing 'seg' and 'cc':"
frac_disagree = np.mean(seg.ravel()!=cc.ravel())
ri,fscore,prec,rec = m.rand_index(seg,cc)
V_rand,V_rand_split,V_rand_merge = m.compute_V_rand_N2(seg,cc)
print "Connected components disagree at %f%% locations" % (frac_disagree*100)
print "\tRand index: %f, fscore: %f, prec: %f, rec: %f" % (ri,fscore,prec,rec)
print "\tV_rand: %f, V_rand_split: %f, V_rand_merge: %f" % (V_rand,V_rand_split,V_rand_merge)

print "Comparing 'cc' and 'cc2':"
frac_disagree = np.mean(cc.ravel()!=cc2.ravel())
ri,fscore,prec,rec = m.rand_index(cc,cc2)
V_rand,V_rand_split,V_rand_merge = m.compute_V_rand_N2(cc,cc2)
print "Connected components disagree at %f%% locations" % (frac_disagree*100)
print "\tRand index: %f, fscore: %f, prec: %f, rec: %f" % (ri,fscore,prec,rec)
print "\tV_rand: %f, V_rand_split: %f, V_rand_merge: %f" % (V_rand,V_rand_split,V_rand_merge)
