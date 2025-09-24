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



#class buffer(nn.Module):

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

  #  def sample_from_logits(logits):


# --- PPO Rollout Buffer -------------------------------------------------------
from dataclasses import dataclass


@dataclass
class RolloutConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95

class RolloutBuffer:
    """
    Stores a short on-policy rollout (your episodes are length 9).
    We store raw obs (so we can re-forward with grads), actions, logp, values, rewards.
    """
    def __init__(self, max_steps: int, obs_dim: int, device: torch.device | str = "cuda"):
        self.max_steps = max_steps
        self.obs_dim = obs_dim
        self.device = torch.device(device)
        self.reset()

    def reset(self):
        self.obs      = []
        self.actions  = []
        self.logp     = []
        self.values   = []
        self.rewards  = []
        self.masks    = []  # 1.0 for not-done (you have fixed length = 9 -> all ones)
        self._finalized = False

    def add(self, obs_t: torch.Tensor, action_t: torch.Tensor,
            logp_t: torch.Tensor, value_t: torch.Tensor, reward_t: torch.Tensor,
            mask_t: torch.Tensor | None = None):
        """
        Shapes:
          obs_t:    (B, obs_dim)
          action_t: (B, ...)  (Discrete: (B,), MultiDiscrete: (B,D))
          logp_t:   (B,)
          value_t:  (B,)
          reward_t: (B,)
          mask_t:   (B,)  (default=1.0)
        """
        if mask_t is None:
            mask_t = torch.ones_like(value_t, device=value_t.device)
        self.obs.append(obs_t.detach())
        self.actions.append(action_t.detach())
        self.logp.append(logp_t.detach())
        self.values.append(value_t.detach())
        self.rewards.append(reward_t.detach())
        self.masks.append(mask_t.detach())

    @torch.no_grad()
    def finalize(self, cfg: RolloutConfig, last_value: torch.Tensor | None = None):
        """
        Compute advantages (GAE) and returns. If you pass last_value, it's V_{T+1};
        else assumed 0 (end of episode).
        """
        assert not self._finalized, "Already finalized"
        device = self.values[0].device
        gamma, lam = cfg.gamma, cfg.gae_lambda

        # Stack time-major
        values  = torch.stack(self.values,  dim=0)        # (T,B)
        rewards = torch.stack(self.rewards, dim=0)        # (T,B)
        masks   = torch.stack(self.masks,   dim=0)        # (T,B)

        T, B = rewards.shape
        if last_value is None:
            next_values = torch.zeros(B, device=device)
        else:
            next_values = last_value.detach()

        adv = torch.zeros(B, device=device)
        advs = []
        for t in reversed(range(T)):
            delta = rewards[t] + gamma * next_values * masks[t] - values[t]
            adv = delta + gamma * lam * masks[t] * adv
            advs.append(adv)
            next_values = values[t]
        advs.reverse()
        advantages = torch.stack(advs, dim=0)             # (T,B)
        returns = advantages + values                      # (T,B)

        # Cache tensors for minibatching
        self._obs     = torch.stack(self.obs,    dim=0)   # (T,B,obs_dim)
        self._actions = self._stack_actions(self.actions) # time-major
        self._logp    = torch.stack(self.logp,   dim=0)   # (T,B)
        self._values  = values
        self._advs    = advantages
        self._rets    = returns

        # Normalize advantages across the whole rollout (common PPO trick)
        flat_advs = self._advs.flatten()
        self._advs = (self._advs - flat_advs.mean()) / (flat_advs.std(unbiased=False) + 1e-8)
        self._finalized = True

    def _stack_actions(self, acts_list):
        # Supports Discrete (T*[B]) or MultiDiscrete (T*[B,D])
        a0 = acts_list[0]
        if a0.ndim == 1:
            return torch.stack(acts_list, dim=0)          # (T,B)
        elif a0.ndim == 2:
            return torch.stack(acts_list, dim=0)          # (T,B,D)
        else:
            raise ValueError("Unexpected action ndim")

    def iter_minibatches(self, batch_size: int, shuffle: bool = True):
        """
        Yields time×batch indices flattened into (T*B, ...).
        """
        assert self._finalized, "Call finalize() first"
        T, B = self._logp.shape
        N = T * B
        idx = torch.arange(N)
        if shuffle:
            idx = idx[torch.randperm(N)]
        for start in range(0, N, batch_size):
            sl = idx[start:start+batch_size]
            t = sl // B
            b = sl % B
            yield {
                "obs":     self._obs[t, b, :],        # (bs, obs_dim)
                "actions": self._gather_time_batch(self._actions, t, b),
                "logp":    self._logp[t, b],
                "values":  self._values[t, b],
                "advs":    self._advs[t, b],
                "rets":    self._rets[t, b],
            }

    def _gather_time_batch(self, tensor, t_idx, b_idx):
        if tensor.ndim == 2:   # (T,B)
            return tensor[t_idx, b_idx]
        if tensor.ndim == 3:   # (T,B,D)
            return tensor[t_idx, b_idx, :]
        raise ValueError("Unsupported action tensor rank")

    # Convenience getters (full, time-major)
    @property
    def T(self): return len(self.rewards)
    @property
    def B(self): return self.values[0].shape[0] if self.values else 0




# --- Sampling helpers ---------------------------------------------------------
import torch.nn.functional as F

def sample_from_logits(logits: torch.Tensor, mode: str):
    """
    mode:
      - 'discrete'         : logits shape (B, A)
      - 'multidiscrete'    : logits shape (B, D, C)  (per-qubit categorical)
    Returns:
      actions, logp
      - discrete: actions shape (B,)
      - multidiscrete: actions shape (B, D) and logp is sum of per-dim logp (B,)
    """
    if mode == "discrete":
        # (B,A)
        probs = F.softmax(logits, dim=-1)
        dist  = torch.distributions.Categorical(probs=probs)
        a = dist.sample()                   # (B,)
        logp = dist.log_prob(a)             # (B,)
        return a, logp

    elif mode == "multidiscrete":
        # (B,D,C)
        B, D, C = logits.shape
        probs = F.softmax(logits, dim=-1)
        # Sample per qubit independently
        a_list, lp_list = [], []
        for d in range(D):
            dist = torch.distributions.Categorical(probs=probs[:, d, :])
            a_d  = dist.sample()            # (B,)
            lp_d = dist.log_prob(a_d)       # (B,)
            a_list.append(a_d)
            lp_list.append(lp_d)
        actions = torch.stack(a_list, dim=1)         # (B,D)
        logp    = torch.stack(lp_list, dim=1).sum(-1) # sum over D -> (B,)
        return actions, logp

    else:
        raise ValueError("mode must be 'discrete' or 'multidiscrete'")
