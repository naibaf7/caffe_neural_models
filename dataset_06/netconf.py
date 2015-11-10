import math

# 10 GB total memory limit
mem_global_limit = 10 * 1024 * 1024 * 1024

# 4 GB single buffer memory limit
mem_buf_limit = 4 * 1024 * 1024 * 1024

# Desired number of output dimensions
output_shape = [44, 44, 44]

# Number of U-Net Pooling-Convolution downsampling/upsampling steps
unet_depth = 3
# Feature map increase rule (downsampling)
def unet_fmap_inc_rule(fmaps):
    return int(math.ceil(fmaps * 4));
# Feature map decrease rule (upsampling)
def unet_fmap_dec_rule(fmaps):
    return int(math.ceil(fmaps / 4));


# Number of SK-Net Pooling-Convolution steps
sknet_conv_depth = 0
# Feature map increase rule
def sknet_fmap_inc_rule(fmaps):
    return int(math.ceil(fmaps * 1.5));
# Number of 1x1 (IP) Convolution steps
sknet_ip_depth = 0
# Feature map increase rule from SK-Convolution to IP
def sknet_fmap_bridge_rule(fmaps):
    return int(math.ceil(fmaps * 4));
# Feature map decrease rule within IP
def sknet_fmap_dec_rule(fmaps):
    return int(math.ceil(fmaps / 2.5));


# Loss function and mode
#loss_function: "malis", "euclid", "softmax"
loss_function = "euclid"
