import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qlearning import train, greedy_rollout, moving_avg
from kurir_env import KurirEnv

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9})
EPIS, W = 2000, 100
res = {}

# ---------- 1. Pelatihan dasar ----------
env, Q, R, S, E = train(alpha=0.8, gamma=0.9, episodes=EPIS, seed=42)
ma = moving_avg(R, W)
final = ma[-100:].mean()
conv = int(np.argmax(ma >= 0.95 * final))          # episode pertama MA mencapai 95% plateau
tot, acts, path, done = greedy_rollout(env, Q)
res["base"] = dict(final_ma=float(final), conv_episode=conv, ma_ep100=float(ma[99]),
                   first100_mean=float(R[:100].mean()), last100_mean=float(R[-100:].mean()),
                   min_first100=float(R[:100].min()), greedy_reward=int(tot), greedy_steps=len(acts),
                   greedy_actions=[env.NAMA_AKSI[a] for a in acts], path=[list(p) for p in path],
                   eps_at_conv=float(E[conv]), eps_end=float(E[-1]),
                   avg_steps_first100=float(S[:100].mean()), avg_steps_last100=float(S[-100:].mean()))

# kurva pembelajaran
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
ax[0].plot(R, color="#bbbbbb", lw=0.4, label="Reward per episode")
ax[0].plot(ma, color="#1b7f3b", lw=1.5, label=f"Moving average ({W})")
ax[0].axvline(conv, color="red", ls="--", lw=0.8, label=f"Konvergen ≈ ep. {conv}")
ax[0].set_xlabel("Episode"); ax[0].set_ylabel("Reward"); ax[0].set_title("Kurva Pembelajaran"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
ax[1].plot(E, color="#c0392b"); ax[1].set_xlabel("Episode"); ax[1].set_ylabel("Epsilon"); ax[1].set_title("Peluruhan ε"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("fig_kurva.png", dpi=220); plt.close()

# ---------- 2. Policy ----------
def draw_policy(ax, paket):
    grid = np.zeros((4, 4))
    for r in range(4):
        for c in range(4):
            s = env.encode((r, c), paket)
            grid[r, c] = np.max(Q[s])
    im = ax.imshow(grid, cmap="YlGn")
    for r in range(4):
        for c in range(4):
            s = env.encode((r, c), paket)
            lab = env.SIMBOL[int(np.argmax(Q[s]))]
            if (r, c) in env.banjir: lab = "X"
            tag = ""
            if (r, c) == env.titik_p: tag = "P"
            if (r, c) == env.titik_d: tag = "D"
            if (r, c) == env.start: tag = "S"
            ax.text(c, r, lab, ha="center", va="center", fontsize=15, color="black", fontweight="bold")
            if tag: ax.text(c-0.36, r-0.3, tag, fontsize=8, color="navy", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Paket belum diambil" if paket == 0 else "Paket sedang dibawa")
    return im
fig, ax = plt.subplots(1, 2, figsize=(6.2, 3.0))
im = draw_policy(ax[0], 0); draw_policy(ax[1], 1)
plt.colorbar(im, ax=ax, fraction=0.025, label="max Q(s,a)")
plt.savefig("fig_policy.png", dpi=220, bbox_inches="tight"); plt.close()

np.set_printoptions(precision=1, suppress=True)
res["q_table_sample"] = {"(0,0)|0": Q[env.encode((0,0),0)].round(2).tolist(),
                         "(0,3)|0": Q[env.encode((0,3),0)].round(2).tolist(),
                         "(0,3)|1": Q[env.encode((0,3),1)].round(2).tolist(),
                         "(3,0)|1": Q[env.encode((3,0),1)].round(2).tolist()}
res["policy"] = {f"{p}": [[env.NAMA_AKSI[int(np.argmax(Q[env.encode((r,c),p)]))] for c in range(4)] for r in range(4)] for p in (0,1)}

# ---------- 3. Eksperimen gamma: 0.9 vs 0.1, 10 seed ----------
SEEDS = list(range(10))
exp = {}
for g in (0.9, 0.1):
    allR, succ, gr = [], [], []
    for sd in SEEDS:
        e, q, r, s, ee = train(alpha=0.8, gamma=g, episodes=EPIS, seed=sd)
        allR.append(r)
        t, a, p, d = greedy_rollout(e, q)
        succ.append(bool(d and e.pos == e.titik_d and t > 0)); gr.append(t)
    allR = np.array(allR)
    mam = np.array([moving_avg(x, W) for x in allR])
    exp[g] = dict(all=allR, ma=mam)
    res[f"gamma_{g}"] = dict(final_ma_mean=float(mam[:, -1].mean()), final_ma_std=float(mam[:, -1].std()),
                             success_rate=float(np.mean(succ)), greedy_reward_mean=float(np.mean(gr)),
                             greedy_rewards=[int(x) for x in gr], last100_mean=float(allR[:, -100:].mean()))
    # contoh satu seed untuk policy
    e, q, r, s, ee = train(alpha=0.8, gamma=g, episodes=EPIS, seed=42)
    t, a, p, d = greedy_rollout(e, q)
    res[f"gamma_{g}"].update(seed42_greedy_reward=int(t), seed42_done=bool(d), seed42_steps=len(a),
                             seed42_actions=[e.NAMA_AKSI[i] for i in a][:15], seed42_final_pos=list(e.pos),
                             q_start=q[e.encode((0,0),0)].round(3).tolist())
    exp[g]["q"] = q

fig, ax = plt.subplots(figsize=(4.4, 2.8))
for g, col in ((0.9, "#1b7f3b"), (0.1, "#c0392b")):
    m = exp[g]["ma"].mean(0); sd = exp[g]["ma"].std(0)
    ax.plot(m, color=col, lw=1.5, label=f"γ = {g}")
    ax.fill_between(range(EPIS), m-sd, m+sd, color=col, alpha=.15)
ax.set_xlabel("Episode"); ax.set_ylabel(f"Moving average reward ({W})"); ax.grid(alpha=.3); ax.legend(fontsize=8)
ax.set_title("Perbandingan γ (rata-rata 10 seed ± SD)")
plt.tight_layout(); plt.savefig("fig_gamma.png", dpi=220); plt.close()

json.dump(res, open("hasil.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
