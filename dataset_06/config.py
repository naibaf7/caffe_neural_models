# Specify the path caffe here
caffe_path = "../../caffe_gt"
# Specify wether or not to compile caffe
library_compile = False

# Specify the device to use
device_id = 0

# Specify the solver file
solver_proto = "net_u_3D_2out/neuraltissue_solver.prototxt"

output_dims = [32, 32, 32]
input_padding = [180, 180, 180]

border_reflect = False

# Select "train" or "process"
mode = "train"