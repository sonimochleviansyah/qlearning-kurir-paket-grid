import numpy as np
from kurir_env import KurirEnv

def train(alpha=0.8, gamma=0.9, episodes=2000, max_steps=100, eps_start=1.0, eps_min=0.01,
          decay=0.005, seed=42):
    rng = np.random.default_rng(seed)
    env = KurirEnv()
    Q = np.zeros((env.n_states, env.n_actions))
    rewards, steps_log, eps_log = [], [], []
    for ep in range(episodes):
        eps = eps_min + (eps_start - eps_min) * np.exp(-decay * ep)
        s, total, n = env.reset(), 0, 0
        for _ in range(max_steps):
            a = rng.integers(env.n_actions) if rng.random() < eps else int(np.argmax(Q[s]))
            ns, r, done = env.step(a)
            target = r if done else r + gamma * np.max(Q[ns])      # state terminal: tanpa bootstrap
            Q[s, a] = (1 - alpha) * Q[s, a] + alpha * target
            s, total, n = ns, total + r, n + 1
            if done:
                break
        rewards.append(total); steps_log.append(n); eps_log.append(eps)
    return env, Q, np.array(rewards), np.array(steps_log), np.array(eps_log)

def greedy_rollout(env, Q, max_steps=50):
    s, total, path = env.reset(), 0, [env.start]
    acts = []
    for _ in range(max_steps):
        a = int(np.argmax(Q[s])); acts.append(a)
        s, r, done = env.step(a); total += r; path.append(env.pos)
        if done: break
    return total, acts, path, done

def moving_avg(x, w=100):
    import pandas as pd
    return pd.Series(x).rolling(w, min_periods=1).mean().values

if __name__ == "__main__":
    env, Q, R, S, E = train()
    tot, acts, path, done = greedy_rollout(env, Q)
    print("greedy total reward:", tot, "langkah:", len(acts), "done:", done)
    print([env.NAMA_AKSI[a] for a in acts])
    print("MA100 awal/akhir:", moving_avg(R)[99], moving_avg(R)[-1])
