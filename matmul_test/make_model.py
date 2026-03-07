import tensorflow as tf
import os

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(1024,)),
    tf.keras.layers.Dense(1024, use_bias=False)
])

model.save('matmul_large.h5')
print("✅ Created matmul_large.h5")