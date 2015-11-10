import numpy as np
cimport numpy as np
from scipy.misc import comb
import scipy.sparse

cdef extern from "malis_cpp.h":
    void malis_loss_weights_cpp(const int nVert, const int* segTrue,
                   const int nEdge, const int* node1, const int* node2, const float* edgeWeight,
                   const int pos,
                   int* nPairPerEdge);
    void connected_components_cpp(const int nVert,
                   const int nEdge, const int* node1, const int* node2, const int* edgeWeight,
                   int* seg);
    void marker_watershed_cpp(const int nVert, const int* marker,
                   const int nEdge, const int* node1, const int* node2, const float* edgeWeight,
                   int* seg);

def malis_loss_weights(np.ndarray[np.int32_t,ndim=1] segTrue,
                np.ndarray[np.int32_t,ndim=1] node1,
                np.ndarray[np.int32_t,ndim=1] node2,
                np.ndarray[np.float32_t,ndim=1] edgeWeight,
                np.int pos):
    cdef int nVert = segTrue.shape[0]
    cdef int nEdge = node1.shape[0]
    segTrue = np.ascontiguousarray(segTrue)
    node1 = np.ascontiguousarray(node1)
    node2 = np.ascontiguousarray(node2)
    edgeWeight = np.ascontiguousarray(edgeWeight)
    cdef np.ndarray[np.int32_t,ndim=1] nPairPerEdge = np.zeros(edgeWeight.shape[0],dtype=np.int32)
    malis_loss_weights_cpp(nVert, <int*> &segTrue[0],
                   nEdge, <int*> &node1[0], <int*> &node2[0], <float*> &edgeWeight[0],
                   pos,
                   <int*> &nPairPerEdge[0]);
    return nPairPerEdge

def connected_components(np.int nVert,
                         np.ndarray[np.int32_t,ndim=1] node1,
                         np.ndarray[np.int32_t,ndim=1] node2,
                         np.ndarray[np.int32_t,ndim=1] edgeWeight,
                         int sizeThreshold=1):
    cdef int nEdge = node1.shape[0]
    node1 = np.ascontiguousarray(node1)
    node2 = np.ascontiguousarray(node2)
    edgeWeight = np.ascontiguousarray(edgeWeight)
    cdef np.ndarray[np.int32_t,ndim=1] seg = np.zeros(nVert,dtype=np.int32)
    connected_components_cpp(nVert,
                             nEdge, <int*> &node1[0], <int*> &node2[0], <int*> &edgeWeight[0],
                             <int*> &seg[0]);
    (seg,segSizes) = prune_and_renum(seg,sizeThreshold)
    return (seg, segSizes)


def marker_watershed(np.ndarray[np.int32_t,ndim=1] marker,
                     np.ndarray[np.int32_t,ndim=1] node1,
                     np.ndarray[np.int32_t,ndim=1] node2,
                     np.ndarray[np.float32_t,ndim=1] edgeWeight,
                     int sizeThreshold=1):
    cdef int nVert = marker.shape[0]
    cdef int nEdge = node1.shape[0]
    marker = np.ascontiguousarray(marker)
    node1 = np.ascontiguousarray(node1)
    node2 = np.ascontiguousarray(node2)
    edgeWeight = np.ascontiguousarray(edgeWeight)
    cdef np.ndarray[np.int32_t,ndim=1] seg = np.zeros(nVert,dtype=np.int32)
    marker_watershed_cpp(nVert, <int*> &marker[0],
                         nEdge, <int*> &node1[0], <int*> &node2[0], <float*> &edgeWeight[0],
                         <int*> &seg[0]);
    (seg,segSizes) = prune_and_renum(seg,sizeThreshold)
    return (seg, segSizes)



def prune_and_renum(np.ndarray[np.int32_t,ndim=1] seg,
                    int sizeThreshold=1):
    # renumber the components in descending order by size
    segId,segSizes = np.unique(seg, return_counts=True)
    descOrder = np.argsort(segSizes)[::-1]
    renum = np.zeros(segId.max()+1,dtype=np.int32)
    segId = segId[descOrder]
    segSizes = segSizes[descOrder]
    renum[segId] = np.arange(1,len(segId)+1)

    if sizeThreshold>0:
        renum[segId[segSizes<=sizeThreshold]] = 0
        segSizes = segSizes[segSizes>sizeThreshold]

    seg = renum[seg]
    return (seg, segSizes)


def bmap_to_affgraph(bmap,nhood,return_min_idx=False):
    # constructs an affinity graph from a boundary map
    # assume affinity graph is represented as:
    # shape = (e, z, y, x)
    # nhood.shape = (edges, 3)
    shape = bmap.shape
    nEdge = nhood.shape[0]
    aff = np.zeros((nEdge,)+shape,dtype=np.int32)
    minidx = np.zeros((nEdge,)+shape,dtype=np.int32)

    for e in range(nEdge):
        aff[e, \
            max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] = np.minimum( \
                        bmap[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
                            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])], \
                        bmap[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                            max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1]), \
                            max(0,nhood[e,2]):min(shape[2],shape[2]+nhood[e,2])] )
        minidx[e, \
            max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] = \
                        bmap[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
                            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] > \
                        bmap[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                            max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1]), \
                            max(0,nhood[e,2]):min(shape[2],shape[2]+nhood[e,2])]

    return aff

def seg_to_affgraph(seg,nhood):
    # constructs an affinity graph from a segmentation
    # assume affinity graph is represented as:
    # shape = (e, z, y, x)
    # nhood.shape = (edges, 3)
    shape = seg.shape
    nEdge = nhood.shape[0]
    aff = np.zeros((nEdge,)+shape,dtype=np.int32)

    for e in range(nEdge):
        aff[e, \
            max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] = \
                        (seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
                            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] == \
                         seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                            max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1]), \
                            max(0,nhood[e,2]):min(shape[2],shape[2]+nhood[e,2])] ) \
                        * ( seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
                            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] > 0 ) \
                        * ( seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                            max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1]), \
                            max(0,nhood[e,2]):min(shape[2],shape[2]+nhood[e,2])] > 0 )

    return aff

def nodelist_like(shape,nhood):
    # constructs the node lists corresponding to the edge list representation of an affinity graph
    # assume  node shape is represented as:
    # shape = (z, y, x)
    # nhood.shape = (edges, 3)
    nEdge = nhood.shape[0]
    nodes = np.arange(np.prod(shape),dtype=np.int32).reshape(shape)
    node1 = np.tile(nodes,(nEdge,1,1,1))
    node2 = np.full(node1.shape,-1,dtype=np.int32)

    for e in range(nEdge):
        node2[e, \
            max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] = \
                nodes[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                     max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1]), \
                     max(0,nhood[e,2]):min(shape[2],shape[2]+nhood[e,2])]

    return (node1, node2)


def affgraph_to_edgelist(aff,nhood):
    node1,node2 = nodelist_like(aff.shape[1:],nhood)
    return (node1.ravel(),node2.ravel(),aff.ravel())

def connected_components_affgraph(aff,nhood):
    (node1,node2,edge) = affgraph_to_edgelist(aff,nhood)
    (seg,segSizes) = connected_components(int(np.prod(aff.shape[1:])),node1,node2,edge)
    seg = seg.reshape(aff.shape[1:])
    return (seg,segSizes)

def compute_V_rand_N2(segTrue,segEst):
    segTrue = segTrue.ravel()
    segEst = segEst.ravel()
    idx = segTrue != 0
    segTrue = segTrue[idx]
    segEst = segEst[idx]

    cont_table = scipy.sparse.coo_matrix((np.ones(segTrue.shape),(segTrue,segEst))).toarray()
    P = cont_table/cont_table.sum()
    t = P.sum(axis=0)
    s = P.sum(axis=1)

    V_rand_split = (P**2).sum() / (t**2).sum()
    V_rand_merge = (P**2).sum() / (s**2).sum()
    V_rand = 2*(P**2).sum() / ((t**2).sum()+(s**2).sum())

    return (V_rand,V_rand_split,V_rand_merge)

def rand_index(segTrue,segEst):
    segTrue = segTrue.ravel()
    segEst = segEst.ravel()
    idx = segTrue != 0
    segTrue = segTrue[idx]
    segEst = segEst[idx]

    tp_plus_fp = comb(np.bincount(segTrue), 2).sum()
    tp_plus_fn = comb(np.bincount(segEst), 2).sum()
    A = np.c_[(segTrue, segEst)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(segTrue))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    ri = (tp + tn) / (tp + fp + fn + tn)
    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    fscore = 2*prec*rec/(prec+rec)
    return (ri,fscore,prec,rec)

def mknhood2d(radius=1):
    # Makes nhood structures for some most used dense graphs.

    ceilrad = np.ceil(radius)
    x = np.arange(-ceilrad,ceilrad+1,1)
    y = np.arange(-ceilrad,ceilrad+1,1)
    [i,j] = np.meshgrid(y,x)

    idxkeep = (i**2+j**2)<=radius**2
    i=i[idxkeep].ravel(); j=j[idxkeep].ravel();
    zeroIdx = np.ceil(len(i)/2).astype(np.int32);

    nhood = np.vstack((i[:zeroIdx],j[:zeroIdx])).T.astype(np.int32)
    return np.ascontiguousarray(np.flipud(nhood))

def mknhood3d(radius=1):
    # Makes nhood structures for some most used dense graphs.
    # The neighborhood reference for the dense graph representation we use
    # nhood(1,:) is a 3 vector that describe the node that conn(:,:,:,1) connects to
    # so to use it: conn(23,12,42,3) is the edge between node [23 12 42] and [23 12 42]+nhood(3,:)
    # See? It's simple! nhood is just the offset vector that the edge corresponds to.

    ceilrad = np.ceil(radius)
    x = np.arange(-ceilrad,ceilrad+1,1)
    y = np.arange(-ceilrad,ceilrad+1,1)
    z = np.arange(-ceilrad,ceilrad+1,1)
    [i,j,k] = np.meshgrid(z,y,z)

    idxkeep = (i**2+j**2+k**2)<=radius**2
    i=i[idxkeep].ravel(); j=j[idxkeep].ravel(); k=k[idxkeep].ravel();
    zeroIdx = np.ceil(len(i)/2).astype(np.int32);

    nhood = np.vstack((k[:zeroIdx],i[:zeroIdx],j[:zeroIdx])).T.astype(np.int32)
    return np.ascontiguousarray(np.flipud(nhood))

def mknhood3d_aniso(radiusxy=1,radiusxy_zminus1=1.8):
    # Makes nhood structures for some most used dense graphs.

    nhoodxyz = mknhood3d(radiusxy)
    nhoodxy_zminus1 = mknhood2d(radiusxy_zminus1)
    
    nhood = np.zeros((nhoodxyz.shape[0]+2*nhoodxy_zminus1.shape[0],3),dtype=np.int32)
    nhood[:3,:3] = nhoodxyz
    nhood[3:,0] = -1
    nhood[3:,1:] = np.vstack((nhoodxy_zminus1,-nhoodxy_zminus1))

    return np.ascontiguousarray(nhood)
