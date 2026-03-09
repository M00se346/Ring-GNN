import os
os.environ['DGLBACKEND'] = 'pytorch'
os.environ['DGL_DISABLE_GRAPHBOLT'] = '1'

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
from dgl import DGLGraph
from dgl.utils import Index
import pickle

# --- NPU ACCELERATOR WRAPPER ---
from memryx import SyncAccl

# class NPU_Accelerator:
#     def __init__(self, dfp_path='npu_speedup/gnn_ops_32.dfp'):
#         # Initialize hardware
#         self.accl = SyncAccl(dfp_path)
        
#     def run(self, inputs):
#         # inputs: [N, D, m, m] -> [1, 32, 32, 32]
#         # 1. Permute to Channels Last for NPU
#         x_np = inputs.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.float32)
        
#         # 2. Run on MPU (Returns [1, 32, 32, 160])
#         out_np = self.accl.run(x_np)[0]
        
#         # 3. Back to Torch
#         out_th = th.from_numpy(out_np).to(inputs.device)
        
#         # 4. Split into the 5 basis ops we compiled (32 channels each)
#         op1, op6, op7, op10, op11 = th.split(out_th, 32, dim=-1)
        
#         return [
#             op1.permute(0, 3, 1, 2),
#             op6.permute(0, 3, 1, 2),
#             op7.permute(0, 3, 1, 2),
#             op10.permute(0, 3, 1, 2),
#             op11.permute(0, 3, 1, 2)
#         ]
class NPU_Accelerator:
    def __init__(self, dfp_path='npu_speedup/gnn_ops_32.dfp'):
        self.accl = SyncAccl(dfp_path)
        
    def run(self, inputs):
        # inputs shape: [N, D, H, W]
        # 1. Pad or Crop to exactly 32x32 for the MPU
        # (N, 32, 32, 32) is required by your DFP
        N, D, H, W = inputs.shape
        
        if H != 32 or W != 32:
            # Simple zero padding to 32x32
            padded = th.zeros((N, D, 32, 32), device=inputs.device)
            h_max, w_max = min(H, 32), min(W, 32)
            padded[:, :, :h_max, :w_max] = inputs[:, :, :h_max, :w_max]
            inputs = padded

        # 2. Permute to Channels Last (N, 32, 32, D)
        x_np = inputs.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.float32)
        
        # 3. Execute on MPU
        out_np = self.accl.run(x_np)[0]
        
        # 4. Back to Torch and Split
        out_th = th.from_numpy(out_np).to(inputs.device)
        op1, op6, op7, op10, op11 = th.split(out_th, 32, dim=-1)
        
        # 5. Crop back to original size if necessary
        results = [op1, op6, op7, op10, op11]
        final_ops = [op.permute(0, 3, 1, 2)[:, :, :H, :W] for op in results]
        
        return final_ops

# --- EXISTING MLP CLASS ---
class MLP(nn.Module):
    def __init__(self, feats):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(m, n) for m, n in zip(feats[:-1], feats[1:])])
    
    def forward(self, x):
        for layer in self.linears[:-1]:
            x = layer(x)
            x = F.relu(x)
        return self.linears[-1](x)

# --- UPDATED RING_GNN ---
class Ring_GNN(nn.Module):
    def __init__(self, nodeclasses, n_classes, avgnodenum = 10, hidden = 32, radius = 2, use_npu=True):
        super(Ring_GNN, self).__init__()
        self.use_npu = use_npu
        # Share one hardware instance across all layers to save memory
        self.npu_hw = NPU_Accelerator() if use_npu else None
        
        self.depth = [th.LongTensor([nodeclasses]), th.LongTensor([64]), th.LongTensor([64])]
        self.equi_modulelist = nn.ModuleList([
            equi_2_to_2(m, n, radius=radius, k2_init=0.5/avgnodenum, npu_hw=self.npu_hw) 
            for m, n in zip(self.depth[:-1], self.depth[1:])
        ])
        self.prediction = MLP([th.sum(th.stack(self.depth)).item(), hidden, n_classes])

    def forward(self, x):
        x_list = [x]
        for layer in self.equi_modulelist:
            x = F.relu(layer(x))
            x_list.append(x)
        
        x_list = [th.sum(th.sum(x, dim=3), dim=2) for x in x_list]
        x_list = th.cat(x_list, dim=1)
        score = self.prediction(x_list)
        return score

# --- UPDATED EQUI_2_TO_2 ---
class equi_2_to_2(nn.Module):
    def __init__(self, input_depth, output_depth, normalization='inf', normalization_val=1.0, radius=2, k2_init = 0.1, npu_hw=None):
        super(equi_2_to_2, self).__init__()
        basis_dimension = 15
        self.radius = radius
        self.npu_hw = npu_hw # Reference to the MPU
        
        coeffs_values = lambda i, j, k: th.randn([i, j, k]) * th.sqrt(2. / (i + j).float())
        self.diag_bias_list = nn.ParameterList([])
        for i in range(radius):
            for j in range(i+1):
                self.diag_bias_list.append(nn.Parameter(th.zeros(1, output_depth, 1, 1)))

        self.all_bias = nn.Parameter(th.zeros(1, output_depth, 1, 1))
        self.coeffs_list = nn.ParameterList([])
        for i in range(radius):
            for j in range(i+1):
                self.coeffs_list.append(nn.Parameter(coeffs_values(input_depth, output_depth, basis_dimension)))

        self.switch = nn.ParameterList([nn.Parameter(th.FloatTensor([1])), nn.Parameter(th.FloatTensor([k2_init]))])
        self.output_depth = output_depth
        self.normalization = normalization

    def forward(self, inputs):
        m = inputs.size()[3]

        if self.npu_hw is not None and m == 32:
            # HYBRID MODE: Run 5 heavy ops on NPU, 10 light ops on CPU
            # Note: This is an optimization for speed. 
            npu_ops = self.npu_hw.run(inputs)
            cpu_ops = ops_2_to_2(inputs, m, normalization=self.normalization)
            
            # Replace the CPU versions of these 5 with the NPU versions
            # Mapping: op1=0, op6=5, op7=6, op10=9, op11=10
            cpu_ops[0], cpu_ops[5], cpu_ops[6], cpu_ops[9], cpu_ops[10] = npu_ops
            ops_out = th.stack(cpu_ops, dim=2)
        else:
            # DEFAULT CPU MODE
            ops_out = ops_2_to_2(inputs, m, normalization=self.normalization)
            ops_out = th.stack(ops_out, dim=2)

        output_list = []
        for i in range(self.radius):
            for j in range(i+1):
                output_i = th.einsum('dsb,ndbij->nsij', self.coeffs_list[i*(i+1)//2 + j], ops_out)
                mat_diag_bias = th.eye(inputs.size()[3]).to(inputs.device).unsqueeze(0).unsqueeze(0) * self.diag_bias_list[i*(i+1)//2 + j]
                if j == 0:
                    output = output_i + mat_diag_bias
                else:
                    output = th.einsum('abcd,abde->abce', output_i, output)
            output_list.append(output)

        output = 0
        for i in range(self.radius):
            output += output_list[i] * self.switch[i]
        output = output + self.all_bias
        return output

    
    
#------------------------------------------------------------------

#  original ops_2_to_2 function below this for the CPU fallback ...

#------------------------------------------------------------------

def ops_2_to_2(inputs, dim, normalization='inf', normalization_val=1.0): # N x D x m x m
    # input: N x D x m x m
    diag_part = th.diagonal(inputs, dim1 = 2, dim2 = 3) # N x D x m
    sum_diag_part = th.sum(diag_part, dim=2, keepdim = True) # N x D x 1
    sum_of_rows = th.sum(inputs, dim=3) # N x D x m
    sum_of_cols = th.sum(inputs, dim=2) # N x D x m
    sum_all = th.sum(sum_of_rows, dim=2) # N x D

    # op1 - (1234) - extract diag
    op1 = th.diag_embed(diag_part) # N x D x m x m
    
    # op2 - (1234) + (12)(34) - place sum of diag on diag
    op2 = th.diag_embed(sum_diag_part.repeat(1, 1, dim))
    
    # op3 - (1234) + (123)(4) - place sum of row i on diag ii
    op3 = th.diag_embed(sum_of_rows)

    # op4 - (1234) + (124)(3) - place sum of col i on diag ii
    op4 = th.diag_embed(sum_of_cols)

    # op5 - (1234) + (124)(3) + (123)(4) + (12)(34) + (12)(3)(4) - place sum of all entries on diag
    op5 = th.diag_embed(sum_all.unsqueeze(2).repeat(1, 1, dim))
    
    # op6 - (14)(23) + (13)(24) + (24)(1)(3) + (124)(3) + (1234) - place sum of col i on row i
    op6 = sum_of_cols.unsqueeze(3).repeat(1, 1, 1, dim)
    
    # op7 - (14)(23) + (23)(1)(4) + (234)(1) + (123)(4) + (1234) - place sum of row i on row i
    op7 = sum_of_rows.unsqueeze(3).repeat(1, 1, 1, dim)

    # op8 - (14)(2)(3) + (134)(2) + (14)(23) + (124)(3) + (1234) - place sum of col i on col i
    op8 = sum_of_cols.unsqueeze(2).repeat(1, 1, dim, 1)

    # op9 - (13)(24) + (13)(2)(4) + (134)(2) + (123)(4) + (1234) - place sum of row i on col i
    op9 = sum_of_rows.unsqueeze(2).repeat(1, 1, dim, 1)

    # op10 - (1234) + (14)(23) - identity
    op10 = inputs

    # op11 - (1234) + (13)(24) - transpose
    op11 = th.transpose(inputs, -2, -1)

    # op12 - (1234) + (234)(1) - place ii element in row i
    op12 = diag_part.unsqueeze(3).repeat(1, 1, 1, dim)

    # op13 - (1234) + (134)(2) - place ii element in col i
    op13 = diag_part.unsqueeze(2).repeat(1, 1, dim, 1)

    # op14 - (34)(1)(2) + (234)(1) + (134)(2) + (1234) + (12)(34) - place sum of diag in all entries
    op14 = sum_diag_part.unsqueeze(3).repeat(1, 1, dim, dim)

    # op15 - sum of all ops - place sum of all entries in all entries
    op15 = sum_all.unsqueeze(2).unsqueeze(3).repeat(1, 1, dim, dim)

    #A_2 = th.einsum('abcd,abde->abce', inputs, inputs)
    #A_4 = th.einsum('abcd,abde->abce', A_2, A_2)
    #op16 = th.where(A_4>1, th.ones(A_4.size()), A_4)

    if normalization is not None:
        float_dim = float(dim)
        if normalization is 'inf':
            op2 = th.div(op2, float_dim)
            op3 = th.div(op3, float_dim)
            op4 = th.div(op4, float_dim)
            op5 = th.div(op5, float_dim**2)
            op6 = th.div(op6, float_dim)
            op7 = th.div(op7, float_dim)
            op8 = th.div(op8, float_dim)
            op9 = th.div(op9, float_dim)
            op14 = th.div(op14, float_dim)
            op15 = th.div(op15, float_dim**2)
    
    #return [op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16]
    '''
    l = [op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15]
    for i, ls in enumerate(l):
        print(i+1)
        print(th.sum(ls))
    print("$%^&*(*&^%$#$%^&*(*&^%$%^&*(*&^%$%^&*(")
    '''
    return [op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15]
