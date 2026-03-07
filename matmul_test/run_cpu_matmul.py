import tensorflow as tf
import numpy as np
import time

# 1. Load the same model we sent to the NPU
# We force TensorFlow to use the CPU
with tf.device('/CPU:0'):
    model = tf.keras.models.load_model('matmul_large.h5')

    # 2. Create the exact same input [1, 16]
    data = np.random.rand(1, 1024).astype(np.float32)

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


# Add this to run_cpu_matmul.py
start_time = time.perf_counter()
for _ in range(1000):
    _ = model(data)
total_time = (time.perf_counter() - start_time) * 1000
print(f"⏱️ CPU Total for 1000 runs: {total_time:.2f} ms")