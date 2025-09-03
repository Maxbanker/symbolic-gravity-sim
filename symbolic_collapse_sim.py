# symbolic_collapse_sim.py  (root)
from __future__ import annotations
import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from core.psi_eff import compute_psi_eff
from core.entropy_field import generate_entropy_field
from core.metrics import curvature
from core.agents import AgentConfig, SymbolicAgent


def _sample_starts(psi, curv, n, rng, q_psi=0.60, q_curv_max=0.75, min_dist=8.0):
    H, W = psi.shape
    psi_thr = np.quantile(psi, q_psi)
    curv_thr = np.quantile(curv, q_curv_max)
    mask = (psi >= psi_thr) & (curv <= curv_thr)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        ys, xs = np.where(np.ones_like(psi, dtype=bool))
    idx = rng.permutation(len(xs))
    picks = []
    for k in idx:
        y, x = float(ys[k]), float(xs[k])
        if all((x - px)**2 + (y - py)**2 >= (min_dist**2) for py, px in picks):
            picks.append((y, x))
        if len(picks) >= n:
            break
    while len(picks) < n:
        picks.append((rng.uniform(0, H-1), rng.uniform(0, W-1)))
    return [(x, y) for (y, x) in picks]


def run(seed=42, n_agents=4, steps=300, field="hillvalley", height=120, width=160,
        orbit_bias=0.55, collapse_q=0.15, curvature_q=0.92, curv_consecutive=4,
        curv_hysteresis=0.85, stagnation_tol=0.25, stagnation_std=0.05,
        stagnation_window=12, max_age=10000, psi_p1=1.0, psi_p99=99.0,
        start_qpsi=0.60, start_qcurv=0.75, start_min_dist=10.0,
        save=None, no_show=False, overlay_curvature=False, overlay_alpha=0.25):

    rng = np.random.default_rng(seed)
    shape = (height, width)

    entropy = generate_entropy_field(
        shape=shape, method=field, mean=0.4, std=0.12, seed=seed, n_bumps=8, sigma=0.10
    )

    info = 1.0 - entropy
    psi = compute_psi_eff(
        info_density=info, entropy_density=entropy, mode="log_ratio",
        p1=psi_p1, p99=psi_p99
    )

    curv_map = curvature(psi)

    starts = _sample_starts(
        psi, curv_map, n_agents, rng, q_psi=start_qpsi, q_curv_max=start_qcurv,
        min_dist=start_min_dist
    )

    cfg = AgentConfig(
        orbit_bias=orbit_bias, collapse_q=collapse_q, curvature_q=curvature_q,
        curvature_consecutive=curv_consecutive, curvature_hysteresis=curv_hysteresis,
        stagnation_tol=stagnation_tol, stagnation_std=stagnation_std,
        stagnation_window=stagnation_window, max_age=max_age
    )
    agents = []
    for s in starts:
        ag = SymbolicAgent(psi=psi, curv_map=curv_map, start_xy=s, cfg=cfg)
        ag.prime_thresholds()
        agents.append(ag)

    for _ in range(steps):
        any_alive = False
        for ag in agents:
            if ag.alive:
                any_alive = True
                ag.step()
        if not any_alive:
            break

    for i, ag in enumerate(agents):
        status = ag.collapse_reason or ("active" if ag.alive else "unknown")
        print(f"[agent {i}] steps={ag.steps} | status={status}")

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.contourf(psi, levels=24, cmap="viridis")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("ψ_eff (0..1)")

    if overlay_curvature:
        ax.imshow(curv_map, cmap="magma", alpha=overlay_alpha, origin="upper")

    for i, ag in enumerate(agents):
        pts = np.array(ag.track)[:, [1, 0]]
        if len(pts) > 1:
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, linewidths=2.0, cmap="plasma")
            lc.set_array(np.linspace(0, 1, len(segs)))
            ax.add_collection(lc)
        ax.plot(pts[0, 0], pts[0, 1], "wo", markersize=4)
        ax.plot(pts[-1, 0], pts[-1, 1], "ko", markersize=5)
        label = f"agent {i} ({ag.collapse_reason or 'active'})"
        ax.text(pts[-1, 0] + 1, pts[-1, 1] + 1, label, fontsize=8, color="w")

    ax.set_title("Symbolic Agents in ψ_eff (time-colored)")
    ax.set_xlim(0, width - 1)
    ax.set_ylim(0, height - 1)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    fig.tight_layout()

    if save:
        base, ext = os.path.splitext(save)
        if ext.lower() not in (".png", ".svg"):
            ext = ".png"
        out_img = base + ext
        fig.savefig(out_img, dpi=150)

        report = []
        for i, ag in enumerate(agents):
            report.append({
                "id": i,
                "steps": ag.steps,
                "status": (ag.collapse_reason or ("active" if ag.alive else "unknown")),
                "start": list(map(float, ag.track[0][::-1])),
                "end": list(map(float, ag.track[-1][::-1])),
            })
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump({"agents": report}, f, indent=2)
        print(f"Saved {out_img} and {base+'.json'}")

    if not no_show:
        plt.show()
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Symbolic Collapse Simulator (improved)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--agents", type=int, default=4)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--field", type=str, default="hillvalley",
                   choices=["hillvalley", "gaussian", "uniform", "gradient"])
    p.add_argument("--height", type=int, default=120)
    p.add_argument("--width", type=int, default=160)

    p.add_argument("--orbit_bias", type=float, default=0.55)
    p.add_argument("--collapse_q", type=float, default=0.15)
    p.add_argument("--curvature_q", type=float, default=0.92)
    p.add_argument("--curv_consecutive", type=int, default=4)
    p.add_argument("--curv_hysteresis", type=float, default=0.85)
    p.add_argument("--stagnation_tol", type=float, default=0.25)
    p.add_argument("--stagnation_std", type=float, default=0.05)
    p.add_argument("--stagnation_window", type=int, default=12)
    p.add_argument("--max_age", type=int, default=10000)

    p.add_argument("--psi_p1", type=float, default=1.0)
    p.add_argument("--psi_p99", type=float, default=99.0)

    p.add_argument("--start_qpsi", type=float, default=0.60)
    p.add_argument("--start_qcurv", type=float, default=0.75)
    p.add_argument("--start_min_dist", type=float, default=10.0)

    p.add_argument("--save", type=str, default=None)
    p.add_argument("--no-show", dest="no_show", action="store_true")
    p.add_argument("--overlay_curvature", action="store_true")
    p.add_argument("--overlay_alpha", type=float, default=0.25)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(seed=args.seed, n_agents=args.agents, steps=args.steps, field=args.field,
        height=args.height, width=args.width, orbit_bias=args.orbit_bias,
        collapse_q=args.collapse_q, curvature_q=args.curvature_q,
        curv_consecutive=args.curv_consecutive, curv_hysteresis=args.curv_hysteresis,
        stagnation_tol=args.stagnation_tol, stagnation_std=args.stagnation_std,
        stagnation_window=args.stagnation_window, max_age=args.max_age,
        psi_p1=args.psi_p1, psi_p99=args.psi_p99, start_qpsi=args.start_qpsi,
        start_qcurv=args.start_qcurv, start_min_dist=args.start_min_dist,
        save=args.save, no_show=args.no_show, overlay_curvature=args.overlay_curvature,
        overlay_alpha=args.overlay_alpha)
