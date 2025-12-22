import torch
from mamba_ssm import Mamba
from decoder.decoder_helpers import StimDecoderEnv


# B, L, D = 4, 128, 256  # batch, seq, model dim
# x = torch.randn(B, L, D, device="cuda")

# block = Mamba(d_model=D, d_state=16, d_conv=4, expand=2, use_fast_path=True).cuda()
# y = block(x)           # shape: (B, L, D)
# print(y.shape)

import torch.nn as nn
import torch.utils.checkpoint as cp  # NEW

#x = torch.randn(batch, length, dim).to("cuda") #the thing from the repo
#so we give (shots,rounds, dets )




class MambaBackbone(nn.Module):
    def __init__(self, d_in=8, d_model=256, depth=4, use_checkpoint=True):   #d_in=obs size, d_model=latent state size, depth=stacked mamba layerss
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model) #project 
        self.layers  = nn.ModuleList([Mamba(d_model=d_model, layer_idx=i) for i in range(depth)]) #process through mamba layers
        self.norm    = nn.LayerNorm(d_model) #normalize
        self.use_checkpoint = use_checkpoint

    
    def allocate_cache(self, B):  #technique to create cache with size B=#shots
        # one cache (conv_state, ssm_state) per layer
        return [ly.allocate_inference_cache(B, max_seqlen=1) for ly in self.layers]




    def mamba_step_opt(self, x_t, cache=None):     # x_t: (B, 1, d_in)
        # cache is unused here; you can ignore it in training
        h = self.in_proj(x_t)                      # (B, 1, d_model)
        for ly in self.layers:
            h = ly(h)                              # <- differentiable forward
        return self.norm(h)  
    def mamba_forward(self, x):  # x: (B, T, d_in)
        h = self.in_proj(x)      # (B, T, d_model)
        for ly in self.layers:
            h = ly(h)            # MUST be full-sequence forward
        return self.norm(h)      # (B, T, d_model)

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
        
        #value  = self.v(h_t).squeeze(-1)#state val
        value  = self.v(h_t).squeeze(-1)        # ← remove tanh

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
    # NEW: clear caches between PPO minibatches to reduce memory pressure
    def clear_cache(self):
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

    def act_opt(self, obs_t):                 # obs_t: (B, d_in) obs for round t, ideally float32 in [0,1]
        x = obs_t[:, None, :]             # -> (B, 1, d_in)
        device = next(self.backbone.parameters()).device
        dtype = next(self.backbone.parameters()).dtype

        x_t = x.to(device).to(dtype)
        self.cache = [(conv_state.to(device), ssm_state.to(device)) for conv_state, ssm_state in self.cache]

        h = self.backbone.mamba_step_opt(x_t, self.cache).squeeze(1)  # (B, d_model)
        logits, value = self.heads(h)     # (B, A), (B,)
        return logits, value, h    #we will use the logits to get the action that leads to reward and the value is the estimate that we'll use for GAE 

    def act_seq_opt(self, obs_seq):  # obs_seq: (T, B, d_in) OR (B, T, d_in)
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype

        # expect (T,B,d)
        assert obs_seq.dim() == 3
        T, B, d = obs_seq.shape
        x = obs_seq.permute(1,0,2).to(device=device, dtype=torch.float32)  # (B,T,d)

       # x = obs_seq.to(device=device, dtype=dtype)     # (B,T,d_in)
        h = self.backbone.mamba_forward(x)             # (B,T,d_model)

        B, T, D = h.shape
        h_flat = h.reshape(B*T, D)
        logits_flat, value_flat = self.heads(h_flat)   # (B*T,A), (B*T,)
        logits = logits_flat.view(B, T, -1).permute(1, 0, 2)  # (T,B,A)
        value  = value_flat.view(B, T).permute(1, 0)          # (T,B)
        return logits, value


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




# --- PPO Rollout Buffer -------------------------------------------------------
from dataclasses import dataclass


@dataclass
class RolloutConfig:
    '''changed!!! was 0.99'''
    gamma: float = 0.99
    gae_lambda: float = 0.95

class RolloutBuffer:
    """
    Stores a short on-policy rollout (your episodes are length 9).
    We store raw obs (so we can re-forward with grads), actions, logp, values, rewards.
    """
    def __init__(self, obs_dim: int, device: torch.device | str = "cuda"):
      #  print('\ninitialized RolloutBuffer (PPO mem)!\n')
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
     #   print(f'buffer obs of shape: {self._obs.shape}')
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


    def iter_seq_minibatches(self, mb_shots: int, shuffle: bool = True):
        """
        Yields minibatches over SHOTS, keeping full sequence length T.
        Returns dict with tensors:
        obs:   (T, Mb, obs_dim)
        actions/logp/advs/rets: (T, Mb)
        """
        import torch

        # stack time-major
        obs  = self._obs
        acts = self._actions
        logp = self._logp
        advs  = self._advs                       # expect (T,B)
        rets  = self._rets                       # expect (T,B)

        T, B = logp.shape[:2]
        idx = torch.randperm(B) if shuffle else torch.arange(B)

        for i in range(0, B, mb_shots):
            b_idx = idx[i:i+mb_shots]
            yield {
                "obs": obs[:, b_idx, :],
                "actions": acts[:, b_idx],
                "logp": logp[:, b_idx],
                "advs": advs[:, b_idx],
                "rets": rets[:, b_idx],
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
from torch.distributions import Categorical

def _sanitize_logits(x: torch.Tensor) -> torch.Tensor:
    # Work in float32 for numerical stability (avoid bf16/fp16 softmax pathologies)
    x = x.float()
    # Replace NaN/±inf with large finite numbers, then clamp
    x = torch.nan_to_num(x, neginf=-1e9, posinf=1e9)
    x = torch.clamp(x, min=-1e9, max=1e9)

    # If an entire row was non-finite (now all -1e9), make it uniform (all zeros logits).
    # Detect rows that are effectively "all masked": max very negative and variance ~0
    row_bad = (x.max(dim=-1, keepdim=True).values < -1e8) | (x.var(dim=-1, keepdim=True) == 0)
    x = torch.where(row_bad, torch.zeros_like(x), x)
    return x

def sample_from_logits(logits: torch.Tensor, mode: str):
    """
    mode:
      - 'discrete'      : logits (B, A)
      - 'multidiscrete' : logits (B, D, C)
    Returns:
      actions, logp   | (B,), (B,)   or   (B, D), (B,)
    """
    if mode == "discrete":
        lg = _sanitize_logits(logits)
        dist = Categorical(logits=lg, validate_args=False)  # skip strict Real() checks after sanitization
        a    = dist.sample()
        logp = dist.log_prob(a)
        return a, logp

    elif mode == "multidiscrete":
        B, D, C = logits.shape
        lg = _sanitize_logits(logits)
        a_list, lp_list = [], []
        for d in range(D):
            dist = Categorical(logits=lg[:, d, :], validate_args=False)
            a_d  = dist.sample()
            lp_d = dist.log_prob(a_d)
            a_list.append(a_d)
            lp_list.append(lp_d)
        actions = torch.stack(a_list, dim=1)          # (B, D)
        logp    = torch.stack(lp_list, dim=1).sum(-1) # (B,)
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


def get_entropy_coef(progress, ent0,ent_final):
    warm = 0.25

    if progress < warm:
        return ent0
    
    # cosine annealing from ent0 -> ent_final
    t = (progress - warm) / (1 - warm)
    return ent_final + 0.5 * (ent0 - ent_final) * (1 + np.cos(np.pi * t))




def optimize_ppo(
    agent,
    buf,
    optimizer: torch.optim.Optimizer,
    *,
    clip_eps: float = 0.2,
    epochs: int = 6,
    batch_size: int = 1024,
    value_coef: float = 0.5,
    entropy_coef: float = 1e-3,
    max_grad_norm: float = 1.0,
    # --- optional anneal inputs ---
    update_idx: int | None = None,
    total_updates: int | None = None,
    entropy_coef_min: float = 1e-4,
    kl_stop: float = 0.03,              # NEW: early-stop threshold per epoch
    progress:float
):
    import math
    import torch
    import torch.nn.functional as F

    device = next(agent.parameters()).device

    # # cosine-annealed entropy coeff (episode/update-wise)
    # if (update_idx is not None) and (total_updates is not None) and total_updates > 0:
    #     cos_w = 0.5 * (1.0 + math.cos(math.pi * min(update_idx, total_updates) / total_updates))
    #     entropy_coef_t = entropy_coef_min + (entropy_coef - entropy_coef_min) * cos_w
    # else:
    #     entropy_coef_t = entropy_coef
    '''stage1'''
    #ent_coef=entropy_coef
    ent_coef=get_entropy_coef(progress=progress, ent0=entropy_coef, ent_final=entropy_coef_min)


    stats = {
        "loss_pi": 0.0, "loss_v": 0.0, "entropy": 0.0, "kl": 0.0,
        "clip_frac": 0.0, "ev": 0.0, "grad_norm": 0.0, "logits_std": 0.0,
        "count": 0,
    }

    # hard assert: params must be finite
    for p in agent.parameters():
        if not torch.isfinite(p).all():
            print("[fatal] non-finite param detected before PPO step")
            raise SystemExit

    for _ in range(epochs):
        epoch_stopped = False
        for mb_idx, mb in enumerate(buf.iter_minibatches(batch_size=batch_size, shuffle=True)):
            obs   = mb["obs"].to(device)      # (n, d_in)
            acts  = mb["actions"].to(device)  # (n,)
            oldlp = mb["logp"].to(device)     # (n,)
            adv   = mb["advs"].to(device)     # (n,)
            rets  = mb["rets"].to(device)     # (n,)

            # Guards: observations must be finite
            if not torch.isfinite(obs).all():
                nbad = (~torch.isfinite(obs)).sum().item()
                print(f"[warn] non-finite OBS in minibatch {mb_idx}: {nbad} elements; skipping")
                continue

            # fresh recurrent cache for this minibatch
            agent.begin_episode(B=obs.shape[0], device=device)
            logits, v_pred, _ = agent.act_opt(obs)  # (n,A), (n,)

            # Guard: logits must be finite
            if not torch.isfinite(logits).all():
                nbad = (~torch.isfinite(logits)).sum().item()
                print(f"[warn] non-finite LOGITS in minibatch {mb_idx}: {nbad}; skipping")
                if hasattr(agent, "clear_cache"): agent.clear_cache()
                continue

            # Build distribution and compute losses
            dist    = torch.distributions.Categorical(logits=logits)
            logp    = dist.log_prob(acts.long())
            entropy = dist.entropy()                           # (n,)
            ratio   = (logp - oldlp).exp()

            unclipped = ratio * adv
            clipped   = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            loss_pi   = -torch.min(unclipped, clipped).mean()

            loss_v    = F.mse_loss(v_pred, rets)

            ent       = entropy.mean()
            loss      = loss_pi + value_coef * loss_v - ent_coef * ent

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # guard grads
            for p in agent.parameters():
                if p.grad is not None:
                    torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

            grad_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

            if hasattr(agent, "clear_cache"):
                agent.clear_cache()

            # Diagnostics (compute BEFORE potential early-stop, and count this mb)
            with torch.no_grad():
                approx_kl = (oldlp - logp).mean().clamp_min(0.0).item()
                clip_frac = ((ratio > 1.0 + clip_eps) | (ratio < 1.0 - clip_eps)).float().mean().item()

                var_returns = rets.var(unbiased=False)
                ev = (1.0 - ((rets - v_pred).var(unbiased=False) / (var_returns + 1e-8))).item() if var_returns > 0 else 0.0

                stats["loss_pi"]   += loss_pi.item()
                stats["loss_v"]    += loss_v.item()
                stats["entropy"]   += ent.item()
                stats["kl"]        += approx_kl
                stats["clip_frac"] += clip_frac
                stats["ev"]        += ev
                stats["grad_norm"] += float(grad_norm)
                stats["logits_std"]+= logits.float().std().item()
                stats["count"]     += 1

            # Early-stop if KL is large (after counting this minibatch)
            if approx_kl > kl_stop:
                epoch_stopped = True
                break

        if epoch_stopped:
            break

    if stats["count"] > 0:
        for k in ("loss_pi","loss_v","entropy","kl","clip_frac","ev","grad_norm","logits_std"):
            stats[k] /= stats["count"]
    else:
        print("[warn] PPO update skipped: no valid minibatches")

    stats["entropy_coef_used"] = ent_coef
    return stats

def optimize_ppo_seq(
    agent,
    buf,
    optimizer,
    *,
    clip_eps=0.2,
    epochs=6,
    mb_shots=128,              # <-- THIS replaces batch_size
    value_coef=0.5,
    entropy_coef=1e-3,
    max_grad_norm=1.0,
    entropy_coef_min=1e-4,
    kl_stop=0.03,
    progress: float = 0.0,
):
    import torch
    import torch.nn.functional as F

    device = next(agent.parameters()).device
    ent_coef = get_entropy_coef(progress=progress, ent0=entropy_coef, ent_final=entropy_coef_min)

    stats = {k: 0.0 for k in ["loss_pi","loss_v","entropy","kl","clip_frac","ev","grad_norm","logits_std"]}
    stats["count"] = 0

    # hard assert: params finite
    for p in agent.parameters():
        if not torch.isfinite(p).all():
            print("[fatal] non-finite param detected before PPO step")
            raise SystemExit

   
    for _ in range(epochs):
        epoch_stopped = False

        for mb_idx, mb in enumerate(buf.iter_seq_minibatches(mb_shots=mb_shots, shuffle=True)):
            obs   = mb["obs"].to(device)      # (T,Mb,obs_dim)
            acts  = mb["actions"].to(device)  # (T,Mb)
            oldlp = mb["logp"].to(device)     # (T,Mb)
            adv   = mb["advs"].to(device)     # (T,Mb)
            rets  = mb["rets"].to(device)     # (T,Mb)

            if not torch.isfinite(obs).all():
                print(f"[warn] non-finite OBS in seq minibatch {mb_idx}; skipping")
                continue

            T, Mb = oldlp.shape
            T, _=adv.shape

            # reset recurrent state ONCE per sequence minibatch
            agent.begin_episode(B=Mb, device=device)

            
            # run sequentially to let Mamba use memory
            logits_seq, vpred_seq = agent.act_seq_opt(obs)  # logits: (T,Mb,A), vpred: (T,Mb)


            if not torch.isfinite(logits_seq).all():
                n_nan = torch.isnan(logits_seq).sum().item()
                n_inf = torch.isinf(logits_seq).sum().item()
                print(f"[warn] non-finite LOGITS_SEQ: nan={n_nan} inf={n_inf} (mb={mb_idx}); skipping")
                if hasattr(agent, "clear_cache"): agent.clear_cache()
                continue
            # logits_seq = torch.nan_to_num(logits_seq, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)

            logits_seq = logits_seq.float()

            dist = torch.distributions.Categorical(logits=logits_seq)
            logp_new = dist.log_prob(acts.long())           # (T,Mb)
            entropy  = dist.entropy()                       # (T,Mb)

            if logits_seq is None:
                continue

            
            ratio = (logp_new - oldlp).exp()

            unclipped = ratio * adv
            clipped   = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            loss_pi   = -torch.min(unclipped, clipped).mean()

            loss_v    = F.mse_loss(vpred_seq, rets)

            ent       = entropy.mean()
            loss      = loss_pi + value_coef * loss_v - ent_coef * ent

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            for p in agent.parameters():
                if p.grad is not None:
                    torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

            grad_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

            if hasattr(agent, "clear_cache"):
                agent.clear_cache()

            with torch.no_grad():
                approx_kl = (oldlp - logp_new).mean().clamp_min(0.0).item()
                clip_frac = ((ratio > 1.0 + clip_eps) | (ratio < 1.0 - clip_eps)).float().mean().item()

                var_returns = rets.var(unbiased=False)
                ev = (1.0 - ((rets - vpred_seq).var(unbiased=False) / (var_returns + 1e-8))).item() if var_returns > 0 else 0.0

                stats["loss_pi"]   += loss_pi.item()
                stats["loss_v"]    += loss_v.item()
                stats["entropy"]   += ent.item()
                stats["kl"]        += approx_kl
                stats["clip_frac"] += clip_frac
                stats["ev"]        += ev
                stats["grad_norm"] += float(grad_norm)
                stats["logits_std"]+= logits_seq.float().std().item()
                stats["count"]     += 1

            if approx_kl > kl_stop:
                epoch_stopped = True
                break

        if epoch_stopped:
            break

    if stats["count"] > 0:
        for k in ("loss_pi","loss_v","entropy","kl","clip_frac","ev","grad_norm","logits_std"):
            stats[k] /= stats["count"]
    else:
        print("[warn] PPO update skipped: no valid seq minibatches")

    stats["entropy_coef_used"] = ent_coef
    return stats
