import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# Check CUDA version
print("CUDA version:", tf.sysconfig.get_build_info().get("cuda_version", "Not found"))

# Check cuDNN version
print("cuDNN version:", tf.sysconfig.get_build_info().get("cudnn_version", "Not found"))