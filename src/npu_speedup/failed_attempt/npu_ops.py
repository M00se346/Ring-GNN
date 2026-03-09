import torch as th
import numpy as np
from memryx import SyncAccl

class NPU_Ops_2_to_2:
    def __init__(self, dfp_path='npu_speedup/gnn_ops_32.dfp'):
        # Initialize the accelerator hardware
        self.accl = SyncAccl(dfp_path)
        print(f"✅ NPU Accelerator Initialized: {dfp_path}")

    def __call__(self, inputs):
        # 1. Permute to Channels Last (N, m, m, D) for NPU compatibility
        device = inputs.device
        x_np = inputs.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.float32)

        # 2. Run sequential acceleration on the input data
        # Output is [N, m, m, 160] (5 ops * 32 channels)
        out_np = self.accl.run(x_np)[0]

        # 3. Convert back to PyTorch Tensor
        out_th = th.from_numpy(out_np).to(device)
        
        # 4. Split the 160 channels back into 5 separate ops (32 channels each)
        op1, op6, op7, op10, op11 = th.split(out_th, 32, dim=-1)

        # 5. Permute back to PyTorch format [N, D, m, m]
        return [
            op1.permute(0, 3, 1, 2),
            op6.permute(0, 3, 1, 2),
            op7.permute(0, 3, 1, 2),
            op10.permute(0, 3, 1, 2),
            op11.permute(0, 3, 1, 2)
        ]

    def shutdown(self):
        # Explicitly free up hardware resources
        self.accl.shutdown()