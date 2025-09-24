import torch
from mamba_ssm import Mamba
from decoder.decoder_helpers import StimDecoderEnv


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
    def allocate_cache(self, B):
        # one cache (conv_state, ssm_state) per layer
        return [ly.allocate_inference_cache(B, max_seqlen=1) for ly in self.layers]

    @torch.no_grad()
    def mamba_step(self, x_t, cache):          # x_t: (B, 1, d_in)
        h = self.in_proj(x_t)
        for ly, (conv_state, ssm_state) in zip(self.layers, cache):
            h, _, _ = ly.step(h, conv_state, ssm_state)    # (B, 1, d_model)
        return self.norm(h)  # (B, 1, d_model)



class PolicyValue(nn.Module):
    def __init__(self, d_model=128, n_actions=2, squash=False):
        super().__init__()
        self.pi = nn.Linear(d_model, n_actions)  # logits for your action encoding
        self.v  = nn.Linear(d_model, 1)
    def forward(self, h_t):                      # h_t: (B, d_model)
        logits = self.pi(h_t)
        value  = self.v(h_t).squeeze(-1)
        return logits, value

class DecoderAgent(nn.Module):
    def __init__(self, d_in=8, d_model=128, depth=4, n_actions=2):
        super().__init__()
        self.backbone = MambaBackbone(d_in, d_model, depth)
        self.heads    = PolicyValue(d_model, n_actions)
        self.cache = None

    def begin_episode(self, B):
        self.cache = self.backbone.allocate_cache(B)

    @torch.no_grad()
    def act(self, obs_t):                 # obs_t: (B, d_in) float32 in [0,1]
        x = obs_t[:, None, :]             # -> (B, 1, d_in)                                 #(1024, rounds, detetctors)
        h = self.backbone.step(x, self.cache).squeeze(1)  # (B, d_model)
        logits, value = self.heads(h)     # (B, A), (B,)
        # sample or argmax here depending on train/eval
        return logits, value, h

        

