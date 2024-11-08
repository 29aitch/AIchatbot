import torch
print(torch.cuda.is_available())  # Should print True if a CUDA-compatible GPU is available
print(torch.cuda.get_device_name(0))
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))