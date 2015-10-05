# FlyEM train/test dataset (FIBSEM, fly medulla 7 column dataset)
--Srini Turaga :: June 3rd, 2015

Here’s a very large set of training data assembled by the FlyEM project team
here at Janelia. This is extracted from a part of the fly visual system known
as the “medulla”, which is basically the second stage of computation starting
at the photo-receptors -> lamina -> medulla.

There are two 250^3 volumes (which they refer to as training volumes):

/nobackup/turaga/data/fibsem_medulla_7col/trvol-250-1-h5/
/nobackup/turaga/data/fibsem_medulla_7col/trvol-250-2-h5/

And there are two 520^3 volumes (which they refer to as test volumes):

/nobackup/turaga/data/fibsem_medulla_7col/tstvol-520-1-h5
/nobackup/turaga/data/fibsem_medulla_7col/tstvol-520-2-h5

Each of these volumes contains several files, but the most important ones are:

img_normalized.h5: which contains the grayscale EM images in hdf5 format
groundtruth_seg.h5: which contains the ground truth segmentation
groundtruth_aff.h5: which contains the ground truth affinity graph


### History

This dataset was assembled by reformatting a combination of files in the
directories below into a consistent file format / naming convention. They were
originally assembled by Toufiq, and I think the '-dawmr' directories were
populated by Gary Huang / Viren Jain.

/groups/flyem/data/viren_toufiq_comparison/trvol-250-1
/groups/flyem/data/viren_toufiq_comparison/trvol-250-1-dawmr
/groups/flyem/data/viren_toufiq_comparison/trvol-250-2
/groups/flyem/data/viren_toufiq_comparison/trvol-250-2-dawmr
/groups/flyem/data/viren_toufiq_comparison/tstvol-520-1
/groups/flyem/data/viren_toufiq_comparison/tstvol-520-1-dawmr
/groups/flyem/data/viren_toufiq_comparison/tstvol-520-2
/groups/flyem/data/viren_toufiq_comparison/tstvol-520-2-dawmr


