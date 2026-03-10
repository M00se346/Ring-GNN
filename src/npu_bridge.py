import torch as th
import numpy as np
from memryx import SyncAccl

class NPUStrategy:
    def __init__(self, dfp_path):
        # Initialize the MemryX driver with your compiled .dfp
        self.accl = SyncAccl(dfp_path)
        print(f"MemryX NPU initialized with {dfp_path}")

    def run(self, inputs, dim, normalization='inf'):
        """
        inputs: Torch Tensor [Batch, Channels, 32, 32]
        dim: The size of the adjacency matrix (should be 32)
        """
        # 1. PRE-PROCESS: Prepare for NPU (Numpy + NHWC format)
        # NPU expects [Batch, Height, Width, Channels]
        x_np = inputs.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.float32)

        # 2. NPU EXECUTION: Get the 5 Basis Ops
        # Returns one big array [Batch, 32, 32, 160] (5 ops * 32 channels)
        out_np = self.accl.run(x_np)[0]
        
        # 3. POST-PROCESS: Back to Torch [Batch, 160, 32, 32]
        out_th = th.from_numpy(out_np).to(inputs.device).permute(0, 3, 1, 2)
        
        # Split the 160 channels back into our 5 Core Ops (32 channels each)
        op1, op6, op7, op10, op11 = th.split(out_th, 32, dim=1)

        # 4. CPU ASSEMBLY: Derive the remaining 10 ops
        # We use the CPU for these because they are simple reshapes/broadcasts
        diag_part = th.diagonal(inputs, dim1=2, dim2=3) 
        sum_diag_part = th.sum(diag_part, dim=2, keepdim=True)
        sum_of_rows = th.sum(inputs, dim=3)
        sum_of_cols = th.sum(inputs, dim=2)
        sum_all = th.sum(sum_of_rows, dim=2)

        op2 = th.diag_embed(sum_diag_part.repeat(1, 1, dim))
        op3 = th.diag_embed(sum_of_rows)
        op4 = th.diag_embed(sum_of_cols)
        op5 = th.diag_embed(sum_all.unsqueeze(2).repeat(1, 1, dim))
        
        # These two are transposes of the NPU results
        op8 = op6.transpose(-2, -1) 
        op9 = op7.transpose(-2, -1) 

        op12 = diag_part.unsqueeze(3).repeat(1, 1, 1, dim)
        op13 = diag_part.unsqueeze(2).repeat(1, 1, dim, 1)
        op14 = sum_diag_part.unsqueeze(3).repeat(1, 1, dim, dim)
        op15 = sum_all.unsqueeze(2).unsqueeze(3).repeat(1, 1, dim, dim)

        # 5. NORMALIZATION (Same as your original code)
        if normalization == 'inf':
            f_dim = float(dim)
            op2, op3, op4, op6, op7, op8, op9, op14 = [th.div(o, f_dim) for o in [op2, op3, op4, op6, op7, op8, op9, op14]]
            op5, op15 = [th.div(o, f_dim**2) for o in [op5, op15]]

        return [op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15]