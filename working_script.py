import tensorflow as tf

print('Message 1:')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print('______')


print('Message 2:')
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))
print('______')

print('Message 3:')
tf.debugging.set_log_device_placement(True)

# Create some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)
print('______')

print('Message 4:')
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print('Please install GPU version of TF')
print('______')

print('Message 5:')
gpus = tf.config.list_physical_devices('GPU')
print(f'tf.config.list_physical_devices(GPU): {gpus}')
print('______')

print('Message 6:')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)





