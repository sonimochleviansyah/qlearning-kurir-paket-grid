"""Eksperimen hyperparameter: gamma tinggi (0.9) vs sangat rendah (0.1), 10 seed."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qlearning import train, moving_avg, greedy_rollout
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9})
EPIS, W = 2000, 100
cols = {0.9: "#1b7f3b", 0.1: "#c0392b"}
ma_all, q42 = {}, {}
for g in cols:
    ma_all[g] = np.array([moving_avg(train(gamma=g, episodes=EPIS, seed=s)[2], W) for s in range(10)])
    env, q42[g], *_ = train(gamma=g, episodes=EPIS, seed=42)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
for g, c in cols.items():
    m, sd = ma_all[g].mean(0), ma_all[g].std(0)
    ax[0].plot(m, color=c, lw=1.5, label=f"γ = {g}"); ax[0].fill_between(range(EPIS), m-sd, m+sd, color=c, alpha=.15)
ax[0].set_xlabel("Episode"); ax[0].set_ylabel("Moving average reward"); ax[0].set_title("(a) Kurva pembelajaran (10 seed)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
s0 = env.encode(env.start, 0); x = np.arange(6)
for i, (g, c) in enumerate(cols.items()):
    ax[1].bar(x + (i-0.5)*0.38, q42[g][s0], 0.38, color=c, label=f"γ = {g}")
ax[1].set_xticks(x); ax[1].set_xticklabels(env.NAMA_AKSI, fontsize=7); ax[1].axhline(0, color="k", lw=.6)
ax[1].set_ylabel("Q(S, a)"); ax[1].set_title("(b) Q-value di state awal"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, axis="y")
plt.tight_layout(); plt.savefig("fig_gamma.png", dpi=220); plt.close()
print("ok")
