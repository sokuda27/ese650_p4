import os
os.environ["MUJOCO_GL"] = "glfw"

from dm_control import suite,viewer
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

class uth_t(nn.Module):
    def __init__(s,xdim,udim,
                 hdim=32,fixed_var=True):
        super().__init__()
        s.xdim,s.udim = xdim, udim
        s.fixed_var=fixed_var

        ### TODO
        s.net = nn.Sequential(
            nn.Linear(xdim, hdim),
            nn.Tanh(),
            nn.Linear(hdim, hdim),
            nn.Tanh(),
            nn.Linear(hdim, udim)
        )

        if fixed_var:
            s.log_std = nn.Parameter(-0.5 * th.ones(udim))
        else:
            s.log_std_net = nn.Sequential(
                nn.Linear(xdim, hdim),
                nn.Tanh(),
                nn.Linear(hdim, hdim),
                nn.Tanh(),
                nn.Linear(hdim, udim)
            )
        ### END TODO

    def forward(s,x):
        ### TODO
        mu = s.net(x)
        std = th.exp(s.log_std).unsqueeze(0).expand_as(mu)
        ### END TODO
        return mu,std
    
    def sample(s, x):
        mu, std = s.forward(x)
        dist = th.distributions.Normal(mu, std)
        u = dist.sample()
        log_prob = dist.log_prob(u).sum(dim = 1)
        return u, log_prob
    
    def log_prob(s, x, u):
        mu, std = s.forward(x)
        dist = th.distributions.Normal(mu, std)
        return dist.log_prob(u).sum(dim = 1)

    def kl(s, x, old_mu, old_std):
        mu, std = s.forward(x)
        dist_new = th.distributions.Normal(mu, std)
        dist_old = th.distributions.Normal(old_mu, old_std)
        return th.distributions.kl_divergence(dist_old, dist_new).sum(dim=1)
    
class vth_th(nn.Module):
    def __init__(s, xdim, hdim = 32):
        super().__init__()
        s.net = nn.Sequential(
            nn.Linear(xdim, hdim),
            nn.Tanh(),
            nn.Linear(hdim, hdim),
            nn.Tanh(),
            nn.Linear(hdim, 1)
        )

    def forward(s, x):
        return s.net(x).squeeze(-1)

def rollout(e,uth,T=1000):
    """
    e: environment
    uth: controller
    T: time-steps
    """

    traj=[]
    t=e.reset()
    x=t.observation
    x=np.array(x['orientations'].tolist()+[x['height']]+x['velocity'].tolist(), dtype=np.float32)
    for _ in range(T):
        with th.no_grad():
            u,_=uth.sample(th.from_numpy(x).float().unsqueeze(0))
        u_np = u.squeeze(0).numpy()
        u_np = np.clip(u_np, e.action_spec().minimum, e.action_spec().maximum)
        r = e.step(u_np)
        obs=r.observation
        xp=np.array(obs['orientations'].tolist()+[obs['height']]+obs['velocity'].tolist(), dtype=np.float32)

        t=dict(xp=xp,r=r.reward,u=u,d=r.last())
        traj.append({
            'x': x.copy(),
            'u': u_np.copy(),
            'r': float(r.reward),
            'xp': xp.copy(),
            'done': r.last()
        })
        x=xp
        if r.last():
            break
    return traj

def comp_returns(traj, gamma=0.99):

    returns = []
    G = 0

    for t in reversed(traj):
        G = t['r'] + gamma * G
        returns.insert(0,G)
    return returns

def compute_gae(traj, value_fn, gamma=0.99, lam=0.95):
    states = th.tensor(np.stack([t['x'] for t in traj]), dtype=th.float32)

    with th.no_grad():
        values = value_fn(states).cpu().numpy()

    advantages = np.zeros(len(traj), dtype=np.float32)
    gae = 0.0

    for i in reversed(range(len(traj))):
        if i == len(traj) - 1:
            next_value = 0.0
        else:
            next_value = values[i + 1]

        mask = 0.0 if traj[i]['done'] else 1.0
        delta = traj[i]['r'] + gamma * next_value * mask - values[i]
        gae = delta + gamma * lam * mask * gae
        advantages[i] = gae

    advantages = th.tensor(advantages, dtype=th.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages

def collect_batch(env, policy, batch_size=4000, max_ep_len=1000):
    trajs = []
    steps = 0

    while steps < batch_size:
        traj = rollout(env, policy, T=max_ep_len)
        trajs.append(traj)
        steps += len(traj)

    flat = [t for traj in trajs for t in traj]
    ep_returns = [sum(t['r'] for t in traj) for traj in trajs]
    return flat, ep_returns, steps

def ppo_update(policy, value_fn, policy_opt, value_opt, x, u, old_logp, returns, adv, clipping = 0.2, train_pi_iterations = 20, train_v_iterations = 40):
    # with th.no_grad():
    #     v = value_fn(x)
    #     adv = returns - v
    #     adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    
    
    for _ in range(train_pi_iterations):
        logp = policy.log_prob(x, u)
        ratio = th.exp(logp - old_logp)

        unclipped = ratio * adv
        clipped = th.clamp(ratio, 1 - clipping, 1 + clipping) * adv
        loss_pi = -(th.min(unclipped, clipped).mean())

        policy_opt.zero_grad()
        loss_pi.backward()

        th.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        policy_opt.step()

    for _ in range(train_v_iterations):
        v_pred = value_fn(x)
        loss_v = F.mse_loss(v_pred, returns)

        value_opt.zero_grad()
        loss_v.backward()
        th.nn.utils.clip_grad_norm_(value_fn.parameters(), max_norm=0.5)
        value_opt.step()

        with th.no_grad():
            logp = policy.log_prob(x, u)
            ratio = th.exp(logp - old_logp)
            approx_kl = (old_logp - logp).mean().item()
            clip_frac = ((ratio > 1 + clipping) | (ratio < 1 - clipping)).float().mean().item()

    return loss_pi.item(), loss_v.item(), approx_kl, clip_frac
    
def train_ppo(env, xdim, udim,
              total_timesteps=3_000_000,
              batch_size=8000,
              max_ep_len=1000,
              gamma=0.99,
              lam=0.95,
              clipping=0.2,
              pi_lr=3e-4,
              vf_lr=1e-3):

    policy = uth_t(xdim, udim, hdim=64, fixed_var=True)
    value_fn = vth_th(xdim, hdim=64)

    policy_opt = th.optim.Adam(policy.parameters(), lr=pi_lr)
    value_opt = th.optim.Adam(value_fn.parameters(), lr=vf_lr)

    steps_collected = 0
    episode_returns = []
    avg_returns = []
    iteration = 0

    while steps_collected < total_timesteps:
        trajs = []
        batch_steps = 0

        while batch_steps < batch_size:
            traj = rollout(env, policy, T=max_ep_len)
            trajs.append(traj)
            batch_steps += len(traj)

        iteration += 1
        steps_collected += batch_steps

        flat_traj = [t for traj in trajs for t in traj]

        x = th.tensor(np.stack([t['x'] for t in flat_traj]), dtype=th.float32)
        u = th.tensor(np.stack([t['u'] for t in flat_traj]), dtype=th.float32)

        with th.no_grad():
            old_logp = policy.log_prob(x, u)

        returns_list = []
        adv_list = []

        for traj in trajs:
            # discounted returns for this trajectory
            G = 0.0
            traj_returns = []
            for t in reversed(traj):
                G = t['r'] + gamma * G
                traj_returns.insert(0, G)
            returns_list.extend(traj_returns)

            # GAE for this trajectory
            states = th.tensor(np.stack([t['x'] for t in traj]), dtype=th.float32)
            with th.no_grad():
                values = value_fn(states).cpu().numpy()

            traj_adv = np.zeros(len(traj), dtype=np.float32)
            gae = 0.0
            for i in reversed(range(len(traj))):
                if i == len(traj) - 1:
                    next_value = 0.0
                else:
                    next_value = values[i + 1]

                mask = 0.0 if traj[i]['done'] else 1.0
                delta = traj[i]['r'] + gamma * next_value * mask - values[i]
                gae = delta + gamma * lam * mask * gae
                traj_adv[i] = gae

            adv_list.extend(traj_adv.tolist())

        returns = th.tensor(returns_list, dtype=th.float32)
        adv = th.tensor(adv_list, dtype=th.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        loss_pi, loss_v, approx_kl, clip_frac = ppo_update(
            policy, value_fn, policy_opt, value_opt,
            x, u, old_logp, returns, adv,
            clipping=clipping
        )

        batch_ep_returns = [sum(t['r'] for t in traj) for traj in trajs]
        episode_returns.extend(batch_ep_returns)

        avg_ret = np.mean(episode_returns[-10:])
        avg_returns.append((steps_collected, avg_ret))

        print(f"Iter {iteration:4d} | Steps {steps_collected:7d} | "
              f"BatchAvgRet {np.mean(batch_ep_returns):8.2f} | "
              f"AvgRet {avg_ret:8.2f} | "
              f"LossPi {loss_pi:8.4f} | LossV {loss_v:8.4f} | "
              f"KL {approx_kl:8.4f} | ClipFrac {clip_frac:6.3f}")

    return policy, value_fn, avg_returns

"""
Setup walker environment
"""
r0 = np.random.RandomState(42)
e = suite.load('walker', 'walk',
                 task_kwargs={'random': r0})
U=e.action_spec();udim=U.shape[0];
X=e.observation_spec();xdim=14+1+9;


"""
#Visualize a random controller
"""
def u(dt):
    return np.random.uniform(low=U.minimum,
                             high=U.maximum,
                             size=U.shape)
# viewer.launch(e,policy=u)

policy, value_fn, avg_returns = train_ppo(e, xdim, udim)

steps = [x[0] for x in avg_returns]
rets = [x[1] for x in avg_returns]

plt.figure(figsize=(8,5))
plt.plot(steps, rets)
plt.xlabel("Timesteps")
plt.ylabel("Average Return")
plt.title("PPO Walker Training Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig("walker_training_curve.png", dpi=300)
print("Saved plot to walker_training_curve.png")