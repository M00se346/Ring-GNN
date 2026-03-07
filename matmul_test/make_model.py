import tensorflow as tf
import os

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(16,)),
    tf.keras.layers.Dense(16, use_bias=False)
])

model.save('matmul.h5')
print("✅ Created matmul.h5")