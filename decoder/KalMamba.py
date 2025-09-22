import torch
from mamba_ssm import Mamba


# B, L, D = 4, 128, 256  # batch, seq, model dim
# x = torch.randn(B, L, D, device="cuda")

# block = Mamba(d_model=D, d_state=16, d_conv=4, expand=2, use_fast_path=True).cuda()
# y = block(x)           # shape: (B, L, D)
# print(y.shape)

import torch.nn as nn
#x = torch.randn(batch, length, dim).to("cuda") #the thing from the repo
#so we give (shots,rounds, dets )




class MambaBackbone(nn.Module):
    def __init__(self, d_in=8, d_model=256, depth=4):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        self.layers  = nn.ModuleList([Mamba(d_model=d_model, layer_idx=i) for i in range(depth)])
        self.norm    = nn.LayerNorm(d_model)

    def forward(self, x):                # x: (B, L, d_in)
        h = self.in_proj(x)              # (B, L, d_model)
        for layer in self.layers:
            h = layer(h)                 # (B, L, d_model)
        return self.norm(h)              # (B, L, d_model)


