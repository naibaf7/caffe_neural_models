BASEDIR=`pwd`
(cd ../../caffe_gt/python/ && ./draw_net.py $BASEDIR/neuraltissue_net.prototxt $BASEDIR/neuraltissue_net.ps --rankdir 'TB' --margin '0, 0')
ps2pdf -g5890x6820 neuraltissue_net.ps neuraltissue_net.pdf

