import tensorflow as tf
node1 = tf.constant(3.0)
node2 = tf.constant(4.0)
print(node1, node2)
print(node1 + node2)

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

print(get_available_gpus())
