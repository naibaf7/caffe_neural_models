BASEDIR=`pwd`
(cd ../../caffe_gt/python/ && ./draw_net.py $BASEDIR/net_train_malis.prototxt $BASEDIR/net.ps --rankdir 'TB' --margin '0, 0' --page '5, 8' --pagesize '5, 8' --size '5, 999')
ps2pdf -g3600x5760 net.ps net.pdf

