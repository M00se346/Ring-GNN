import numpy as np
import time
from memryx import SyncAccl

model_path = 'matmul.dfp'

try:
    print(f"--- Opening 4-Chip MPU (Sync Mode) ---")
    # 1. Use SyncAccl for direct data passing
    accl = SyncAccl(model_path)
    
    # 2. Create the input [1, 16]
    data = np.random.rand(1, 16).astype(np.float32)
    
    print("🚀 Sending matrix and waiting for result...")
    
    # 3. Track latency
    start_time = time.perf_counter()
    
    # .run() sends data and returns the output immediately
    outputs = accl.run(data)
    
    end_time = time.perf_counter()
    
    # 4. Success!
    latency = (end_time - start_time) * 1000
    print(f"✅ SUCCESS! MPU returned data.")
    print(f"⏱️ Round-trip Latency: {latency:.3f} ms")
    
    # outputs is a list (one for each output layer)
    print(f"🔢 Result Shape: {outputs[0].shape}")

except Exception as e:
    print(f"❌ Error: {e}")