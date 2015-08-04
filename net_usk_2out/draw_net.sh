BASEDIR=`pwd`
(cd ../../caffe_gt/python/ && ./draw_net.py $BASEDIR/neuraltissue_net.prototxt $BASEDIR/neuraltissue_net.ps --rankdir 'LT' --margin '0, 0' --page '5, 8' --pagesize '5, 8' --size '5, 999')
ps2pdf -g3600x5760 neuraltissue_net.ps neuraltissue_net.pdf

