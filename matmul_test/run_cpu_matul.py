import tensorflow as tf
import numpy as np
import time

# 1. Load the same model we sent to the NPU
# We force TensorFlow to use the CPU
with tf.device('/CPU:0'):
    model = tf.keras.models.load_model('matmul.h5')

    # 2. Create the exact same input [1, 16]
    data = np.random.rand(1, 16).astype(np.float32)

    print("🚀 Running Matrix Multiplication on CPU...")

    # 3. Warm-up (TensorFlow is slow the first time it runs)
    _ = model(data)

    # 4. Measure Latency
    start_time = time.perf_counter()
    
    # Run the math
    result = model(data)
    
    end_time = time.perf_counter()

# 5. Calculate results
latency_ms = (end_time - start_time) * 1000
print(f"✅ SUCCESS! CPU finished.")
print(f"⏱️ CPU Latency: {latency_ms:.3f} ms")