import numpy as np
import time
from memryx import AsyncAccl

# 1. Configuration
num_frames = 1000
model_path = 'matmul_large.dfp'
results_received = 0

# 2. Generator function: The NPU will call this to "pull" data
def data_source():
    # We yield the same matrix 1000 times
    # Note: The NPU expects the shape [1, 1024]
    data = np.random.rand(1, 1024).astype(np.float32)
    for _ in range(num_frames):
        yield data

# 3. Output Handler
def on_output(*outputs):
    global results_received
    results_received += 1

try:
    print(f"--- Initializing Async Stream (4 Chips) ---")
    accl = AsyncAccl(model_path)
    
    # 4. Connect the generator and the processor
    # We pass the FUNCTION NAME 'data_source', not 'data_source()'
    accl.connect_input(data_source)
    accl.connect_output(on_output)
    
    print(f"🚀 Streaming {num_frames} matrices...")
    start_time = time.perf_counter()
    
    # 5. Wait for the NPU to finish pulling all 1000 frames
    accl.wait()
    end_time = time.perf_counter()
    
    total_ms = (end_time - start_time) * 1000
    fps = num_frames / (end_time - start_time)
    
    print(f"\n✅ DONE!")
    print(f"⏱️ NPU Total for {num_frames} runs: {total_ms:.2f} ms")
    print(f"⚡ Throughput: {fps:.2f} inferences per second")
    print(f"📦 Results received: {results_received}")
    
    accl.stop()

except Exception as e:
    print(f"❌ Error: {e}")