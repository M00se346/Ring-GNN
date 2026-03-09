import torch
import torch.nn as nn
import time

class NPU_Basis_Ops(nn.Module):
    def __init__(self, m=32, d=32):
        super().__init__()
        self.m = m
        self.d = d

    def forward(self, x):
        # Input x: [1, 32, 32, 32] (Batch, Channels, Height, Width)
        
        # 1. Identity
        op10 = x 
        
        # 2. Transpose
        op11 = torch.transpose(x, -2, -1)
        
        # 3. Row Sums (Op 7 Basis)
        # Summing across width, then repeating to restore shape
        op7_b = torch.sum(x, dim=-1, keepdim=True).expand(-1, -1, -1, self.m)
        
        # 4. Col Sums (Op 6 Basis)
        # Summing across height, then repeating to restore shape
        op6_b = torch.sum(x, dim=-2, keepdim=True).expand(-1, -1, self.m, -1)
        
        # 5. Diagonal (Op 1 Basis)
        # A simple mask is often more NPU-friendly than diag_embed
        mask = torch.eye(self.m).to(x.device)
        op1 = x * mask
        
        # Concatenate into one output: [1, 160, 32, 32]
        return torch.cat([op1, op6_b, op7_b, op10, op11], dim=1)

# Exporting
model = NPU_Basis_Ops()
model.eval()
dummy_input = torch.randn(1, 32, 32, 32)

print("Exporting ONNX...")
torch.onnx.export(
    model, 
    dummy_input, 
    "gnn_basis.onnx", 
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)

# Crucial: wait for filesystem to catch up
time.sleep(2)
print("Done. Now try running: mx_nc -m gnn_basis.onnx --dfp_fname gnn_ops_32.dfp")