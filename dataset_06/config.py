# Specify the path caffe here
caffe_path = "../../caffe_gt"
# Specify wether or not to compile caffe
library_compile = True

# Specify the device to use
device_id = 1

# Specify the solver file
solver_proto = "net_sk_3D_2out/neuraltissue_solver.prototxt"

output_dims = [32, 32, 32]
input_padding = [101, 101, 101]

border_reflect = False

# Select "train" or "process"
mode = "train"