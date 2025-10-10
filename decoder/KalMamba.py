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
    def __init__(self, d_in=8, d_model=256, depth=4):   #d_in=obs size, d_model=latent state size, depth=stacked mamba layerss
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model) #project 
        self.layers  = nn.ModuleList([Mamba(d_model=d_model, layer_idx=i) for i in range(depth)]) #process through mamba layers
        self.norm    = nn.LayerNorm(d_model) #normalize
    def allocate_cache(self, B):  #technique to create cache with size B=#shots
        # one cache (conv_state, ssm_state) per layer
        return [ly.allocate_inference_cache(B, max_seqlen=1) for ly in self.layers]

    @torch.no_grad()
    def mamba_step(self, x_t, cache):          # x_t: (B, 1, d_in) single round obs, cache is the hidden states from prevs
        device = next(self.parameters()).device
        x_t = x_t.to(device)
        h = self.in_proj(x_t)
        for ly, (conv_state, ssm_state) in zip(self.layers, cache):
            h, _, _ = ly.step(h, conv_state, ssm_state)    # (B, 1, d_model)
        return self.norm(h)  # (B, 1, d_model)



class PolicyValue(nn.Module):
    def __init__(self, d_model=128, n_actions=2, squash=False):
        super().__init__()
        self.pi = nn.Linear(d_model, n_actions)  # logits for  action encoding
        self.v  = nn.Linear(d_model, 1)
    def forward(self, h_t):                      # h_t: (B, d_model), the latent state
        logits = self.pi(h_t) #policy logitgs
        
        value  = self.v(h_t).squeeze(-1)#state val
        return logits, value



#class buffer(nn.Module):
##basically the wrapper between mamba and PPO (and later Kalman)
class DecoderAgent(nn.Module):
    def __init__(self, d_in=8, d_model=128, depth=4, n_actions=2):
        super().__init__()
        self.backbone = MambaBackbone(d_in, d_model, depth)
        self.Kalman=None    #only for now 
        self.heads    = PolicyValue(d_model, n_actions)

        self.cache = None


    def begin_episode(self, B, device=None):  #moves cache states to device
        if device is None:
            device = next(self.parameters()).device
        self.cache = self.backbone.allocate_cache(B)
        cache=self.cache
        moved = []
        for conv_state, ssm_state in cache:
            conv_state = conv_state.to(device)
            ssm_state  = ssm_state.to(device)
            moved.append((conv_state, ssm_state))
        self.cache = moved

    @torch.no_grad()
    def act(self, obs_t):                 # obs_t: (B, d_in) obs for round t, ideally float32 in [0,1]
        x = obs_t[:, None, :]             # -> (B, 1, d_in)
        device = next(self.backbone.parameters()).device
        dtype = next(self.backbone.parameters()).dtype

        x_t = x.to(device).to(dtype)
        self.cache = [(conv_state.to(device), ssm_state.to(device)) for conv_state, ssm_state in self.cache]

        h = self.backbone.mamba_step(x_t, self.cache).squeeze(1)  # (B, d_model)
        logits, value = self.heads(h)     # (B, A), (B,)
        return logits, value, h    #we will use the logits to get the action that leads to reward and the value is the estimate that we'll use for GAE 


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
    def __init__(self, obs_dim: int, device: torch.device | str = "cuda"):
        print('\ninitialized RolloutBuffer (PPO mem)!\n')
        self.obs_dim = obs_dim
        self.device = torch.device(device)
        self.reset()

    def reset(self):  #clears history
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
          obs_t:    (B, obs_dim)  current observations,,,, noo its d_in (feature vector)
          action_t: (B, ...)  (Discrete: (B,), MultiDiscrete: (B,D))   chosen actions
          logp_t:   (B,)     log-prob of chosen action
          value_t:  (B,)   critic value estimate
          reward_t: (B,)    scalar (step) reward
          mask_t:   (B,)  (default=1.0)    =1 if not done, 0 if ep ended
        """
        if mask_t is None:
            mask_t = torch.ones_like(value_t, device=value_t.device)
        self.obs.append(obs_t.detach())
        self.actions.append(action_t.detach())
        self.logp.append(logp_t.detach())
        self.values.append(value_t.detach())
        self.rewards.append(reward_t.detach())
        self.masks.append(mask_t.detach())




#the advantage tells us how much better an action is than average given the current state.
#the finalize func computes Generalized Adv Est
#adv=reward_curr + gamma*NextVal(est) - CurrVal(est) + gamma*lambda*next_adv !!!!!!
#So GAE is basically an exponentially discounted sum of TD errors.
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
        masks   = torch.stack(self.masks,   dim=0)        # (T,B)     #whether the ep is alive or not

        T, B = rewards.shape
        if last_value is None:
            next_values = torch.zeros(B, device=device)
        else:
            next_values = last_value.detach()

        adv = torch.zeros(B, device=device)
        advs = []
        for t in reversed(range(T)):
            delta = rewards[t] + gamma * next_values * masks[t] - values[t]    #δ_t= r_t + γ*V_(t+1) - V_t
            adv = delta + gamma * lam * masks[t] * adv                          #A_t=γλδ_(t+1)
            advs.append(adv)
            next_values = values[t]
        advs.reverse()
        advantages = torch.stack(advs, dim=0)             # (T,B)  A_t
        returns = advantages + values                      # (T,B)  R_t   A_t=R_t-V_t

        # Cache tensors for minibatching
        self._obs     = torch.stack(self.obs,    dim=0)   # (T,B,obs_dim)
        print(f'buffer obs of shape: {self._obs.shape}')
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







import numpy as np
# --- Action → mask bridge -----------------------------------------------------
def action_to_masks(
    actions,
    mode: str,
    data_ids: np.ndarray | list,     # list of global qubit ids for data qubits
    num_qubits: int,                 # total Q for FlipSimulator
    shots: int,                      # S (batch size)
    classes_per_qubit: int = 3      # 3 -> {I,X,Z}, 4 -> {I,X,Z,Y}
):
    """
    Returns dict {'X': xmask, 'Z': zmask} with shape (Q, S) boolean arrays.
    NOTE: We don't use Y directly; Y = X and Z bits set in same shot/qubit.
    """
    data_ids = np.asarray(list(map(int, data_ids)), dtype=int)
    Q, S = int(num_qubits), int(shots)
    xmask = np.zeros((Q, S), dtype=bool)
    zmask = np.zeros((Q, S), dtype=bool)

    if mode == "discrete":
        # actions: (S,) integers in [0, 2*D], so if for example we have value 2 then at qubit 2 -> X
        a = actions.detach().to("cpu").numpy().astype(int)  # (S,)
        D = len(data_ids)
        for s in range(S):
            ai = a[s]
            if ai == 0:
                continue  # do-nothing
            idx = ai - 1
            q_local = idx // 2
            p = idx % 2     # 0->X, 1->Z
            if 0 <= q_local < D:
                q_global = data_ids[q_local]
                if p == 0: xmask[q_global, s] = True
                else:      zmask[q_global, s] = True

    elif mode == "multidiscrete":
        # actions: (S, D) with values in {0=I,1=X,2=Z,(3=Y optional)}
        a = actions.detach().to("cpu").numpy().astype(int)  # (S,D)
        S_, D = a.shape
        assert S_ == S
        for q_local in range(D):
            q_global = data_ids[q_local]
            vals = a[:, q_local]   # (S,)
            if classes_per_qubit == 4:
                xmask[q_global, :] |= (vals == 1) | (vals == 3)
                zmask[q_global, :] |= (vals == 2) | (vals == 3)
            else:
                xmask[q_global, :] |= (vals == 1)
                zmask[q_global, :] |= (vals == 2)
    else:
        raise ValueError("mode must be 'discrete' or 'multidiscrete'")

    return {"X": xmask, "Z": zmask}





# --- Sampling helpers ---------------------------------------------------------
import torch.nn.functional as F

def sample_from_logits(logits: torch.Tensor, mode: str):
    """
    mode:
      - 'discrete'         : logits shape (B, A) 2d+1
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






import torch
import torch.nn.functional as F
from torch.distributions import Categorical

@torch.no_grad()
def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    # logits: (N, A)
    dist = Categorical(logits=logits)
    return dist.entropy()  # (N,)

def optimize_ppo(
    agent,
    buf,                              # RolloutBuffer after .finalize()
    optimizer: torch.optim.Optimizer,
    *,
    clip_eps: float = 0.2,
    epochs: int = 4,
    batch_size: int = 8192,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 1.0,
):
    """
    PPO update using time×batch flattened minibatches from your RolloutBuffer.

    Expects buf.finalize(...) already called, so buf has:
      buf._obs   : (T,B,d_in)
      buf._actions : (T,B)          # Discrete (class ids)
      buf._logp  : (T,B)
      buf._advs  : (T,B)
      buf._rets  : (T,B)
    """
    device = next(agent.parameters()).device

    # convenience views (flatten time×batch)
    def flat(x): return x.reshape(-1, *x.shape[2:]) if x.ndim > 2 else x.reshape(-1)

    obs_all     = buf._obs.to(device)        # (T,B,d_in)

    acts_all    = buf._actions.to(device)    # (T,B)
    old_logp_all= buf._logp.to(device)       # (T,B)
    advs_all    = buf._advs.to(device)       # (T,B)
    rets_all    = buf._rets.to(device)       # (T,B)
    values_all  = buf._values.to(device)     # (T,B) (only for logging if you want)

    N_total = obs_all.shape[0] * obs_all.shape[1]

    # training loop
    stats = {"loss_pi": 0.0, "loss_v": 0.0, "entropy": 0.0, "kl": 0.0, "count": 0}

    for _ in range(epochs):
        for mb in buf.iter_minibatches(batch_size=batch_size, shuffle=True):
            obs   = mb["obs"].to(device)       # (n, d_in)
            print(f'obs shape: {obs.shape}')
            acts  = mb["actions"].to(device)   # (n,)
            oldlp = mb["logp"].to(device)      # (n,)
            adv   = mb["advs"].to(device)      # (n,)
            rets  = mb["rets"].to(device)      # (n,)

            # Re-forward policy/value for the minibatch.
            # Since the backbone is step-based (Mamba), allocate a fresh cache for this minibatch.
            agent.begin_episode(B=obs.shape[0], device=device)
            logits, v_pred, _ = agent.act(obs)          # logits:(n,A), v_pred:(n,)

            # New log-prob of taken actions
            logp = Categorical(logits=logits).log_prob(acts)   # (n,)
            entropy = _entropy_from_logits(logits)             # (n,)

            # PPO ratio
            ratio = (logp - oldlp).exp()                       # (n,)

            # Clipped surrogate policy loss
            unclipped = ratio * adv
            clipped   = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            loss_pi   = -torch.min(unclipped, clipped).mean()

            # Value loss
            loss_v = F.mse_loss(v_pred, rets)

            # Entropy bonus
            ent = entropy.mean()

            loss = loss_pi + value_coef * loss_v - entropy_coef * ent

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

            # Diagnostics (approx KL = E[oldlp - newlp])
            approx_kl = (oldlp - logp).mean().clamp_min(0.0).item()
            stats["loss_pi"] += loss_pi.item()
            stats["loss_v"]  += loss_v.item()
            stats["entropy"] += ent.item()
            stats["kl"]      += approx_kl
            stats["count"]   += 1

    # average stats
    if stats["count"] > 0:
        for k in ("loss_pi","loss_v","entropy","kl"):
            stats[k] /= stats["count"]
    return stats
    