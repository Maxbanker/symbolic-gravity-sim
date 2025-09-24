# symbolic_gravity_sim/symbolic_collapse_sim.py (v3.0.0-11D — robust adapters incl. info_density)
from __future__ import annotations
import argparse
import json
import os
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Core (existing)
from .core.psi_eff import compute_psi_eff
from .core.entropy_field import generate_entropy_field
from .core.metrics import curvature
from .core.agents import AgentConfig, SymbolicAgent

# 11D upgrade (new)
from .core.eleven_d import (
    init_11d_fields, ElevenDThresholds, collapse_predicate_11d,
    time_dilation_dt, route_action
)

# ----------------------------
# Normalization helpers
# ----------------------------

def _normalize_to_2d_array(x) -> np.ndarray:
    """Best-effort: turn various return types into a 2D float array."""
    for attr in ("array", "data", "values"):
        if hasattr(x, attr):
            try:
                x = getattr(x, attr)
                break
            except Exception:
                pass
    if isinstance(x, (tuple, list)) and len(x) > 0:
        for elem in x:
            try:
                arr = np.asarray(elem)
                if arr.ndim >= 2:
                    x = elem
                    break
            except Exception:
                continue
        else:
            x = x[0]
    arr = np.asarray(x)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = np.mean(arr[..., :3], axis=-1)
    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array after normalization, got shape={arr.shape}")
    return arr.astype(float, copy=False)

# ----------------------------
# Robust adapters for project-local APIs
# ----------------------------

def _generate_field_robust(field: str, rng: np.random.Generator) -> np.ndarray:
    """Call user's generate_entropy_field with many signatures, normalize to 2D."""
    default_size = (256, 256)
    errors = []
    call_styles = [
        ((), {"kind": field}),
        ((), {"field": field}),
        ((field,), {}),
        ((field, default_size), {}),
        ((default_size,), {}),
        ((field, rng), {}),
        ((field, default_size, rng), {}),
        ((), {"rng": rng}),
        ((), {}),
        ((), {"size": default_size}),
        ((field,), {"size": default_size}),
        ((), {"field": field, "size": default_size}),
        ((), {"kind": field, "size": default_size}),
        ((), {"size": default_size, "rng": rng}),
        ((), {"field": field, "rng": rng}),
        ((), {"kind": field, "rng": rng}),
        ((), {"field": field, "size": default_size, "rng": rng}),
        ((), {"kind": field, "size": default_size, "rng": rng}),
    ]
    for args, kwargs in call_styles:
        try:
            out = generate_entropy_field(*args, **kwargs)
            return _normalize_to_2d_array(out)
        except Exception:
            continue
    # last-ditch
    out = generate_entropy_field()
    return _normalize_to_2d_array(out)

def _derive_info_density(base2d: np.ndarray) -> np.ndarray:
    """
    Create a plausible info-density companion from base field:
      - gradient magnitude (|∇base|)
      - min–max normalize to [0,1]
    """
    gy, gx = np.gradient(base2d.astype(float, copy=False))
    mag = np.sqrt(gx * gx + gy * gy)
    vmin = float(np.nanmin(mag))
    vmax = float(np.nanmax(mag))
    if vmax - vmin < 1e-12:
        return np.zeros_like(mag)
    return (mag - vmin) / (vmax - vmin)

def _compute_psi_eff_robust(entropy_density: np.ndarray) -> np.ndarray:
    """
    Try common signatures:
      compute_psi_eff(entropy_density, info_density)
      compute_psi_eff(entropy_density=..., info_density=...)
      compute_psi_eff(density=..., info=...)
      compute_psi_eff(field=..., info_density=...)
      Fallbacks try same array for both.
    """
    info_density = _derive_info_density(entropy_density)
    attempts = [
        ((entropy_density, info_density), {}),
        ((), {"entropy_density": entropy_density, "info_density": info_density}),
        ((), {"density": entropy_density, "info": info_density}),
        ((), {"field": entropy_density, "info_density": info_density}),
        # conservative fallback: reuse entropy as info
        ((entropy_density, entropy_density), {}),
        ((), {"entropy_density": entropy_density, "info_density": entropy_density}),
    ]
    errors = []
    for args, kwargs in attempts:
        try:
            out = compute_psi_eff(*args, **kwargs)
            return _normalize_to_2d_array(out)
        except Exception as e:
            errors.append(f"{args!r} {kwargs!r} -> {type(e).__name__}: {e}")
    # If you still land here, raise a concise error with the top attempts
    raise RuntimeError(
        "compute_psi_eff() could not be called with available adapters. "
        "Expected a signature like compute_psi_eff(entropy_density, info_density). "
        "Errors:\n" + "\n".join(errors[:6])
    )

# ----------------------------
# Sampling/plotting helpers
# ----------------------------

def _quantile_mask(a: np.ndarray, q_low: float = 0.6, q_high: float = 1.0) -> np.ndarray:
    lo = np.quantile(a, q_low); hi = np.quantile(a, q_high)
    return (a >= lo) & (a <= hi)

def _sample_starts(
    psi: np.ndarray,
    curv: np.ndarray,
    n: int,
    rng: np.random.Generator,
    q_psi: float = 0.60,
    q_curv_max: float = 0.75,
    min_dist: float = 8.0
) -> List[Tuple[int, int]]:
    H, W = psi.shape
    psi_thr = np.quantile(psi, q_psi)
    curv_thr = np.quantile(curv, q_curv_max)
    cand = np.argwhere((psi >= psi_thr) & (curv <= curv_thr))
    if cand.size == 0:
        cand = np.argwhere(np.ones_like(psi, dtype=bool))
    rng.shuffle(cand)
    picks: List[Tuple[int, int]] = []
    for y, x in cand:
        if all((y - py) ** 2 + (x - px) ** 2 >= (min_dist ** 2) for (py, px) in picks):
            picks.append((y, x))
            if len(picks) >= n:
                break
    while len(picks) < n:
        picks.append((int(rng.integers(0, H)), int(rng.integers(0, W))))
    return picks

def _build_agents(starts: List[Tuple[int, int]], rng: np.random.Generator) -> List[SymbolicAgent]:
    agents: List[SymbolicAgent] = []
    for (y, x) in starts:
        try:
            cfg = AgentConfig(seed=int(rng.integers(0, 1_000_000)))
        except TypeError:
            cfg = AgentConfig()
        agent = None
        try:
            agent = SymbolicAgent(cfg, start=(y, x))
        except Exception:
            pass
        if agent is None:
            try:
                agent = SymbolicAgent(config=cfg, yx=(y, x))
            except Exception:
                pass
        if agent is None:
            try:
                agent = SymbolicAgent(config=cfg)
                if hasattr(agent, "reset"):
                    agent.reset((y, x))
            except Exception:
                pass
        if agent is None:
            raise RuntimeError("Could not construct SymbolicAgent with known signatures.")
        agents.append(agent)
    return agents

def _collect_tracks(agents: List[SymbolicAgent]) -> List[np.ndarray]:
    tracks: List[np.ndarray] = []
    for ag in agents:
        tr = getattr(ag, "track", None)
        if tr is None:
            hist = getattr(ag, "history", [])
            tr = np.array(hist, dtype=float)
        else:
            tr = np.array(tr, dtype=float)
        if tr.ndim == 2 and tr.shape[1] == 2:
            tracks.append(tr[:, ::-1])
        else:
            tracks.append(np.asarray(tr))
    return tracks

def _plot(
    psi: np.ndarray,
    tracks: List[np.ndarray],
    curv_map: Optional[np.ndarray] = None,
    overlay_curvature: bool = False,
    overlay_alpha: float = 0.25,
    route_preview: Optional[np.ndarray] = None
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(psi, origin="upper", cmap="viridis")
    ax.set_xticks([]); ax.set_yticks([])
    if overlay_curvature and curv_map is not None:
        ax.imshow(curv_map, cmap="magma", alpha=overlay_alpha, origin="upper")
    if route_preview is not None:
        try:
            rp = np.where(np.isin(route_preview, [1, 5, 6, 7]), route_preview, np.nan)
            ax.imshow(rp, alpha=0.15, origin="upper")
        except Exception:
            pass
    segs = []
    for tr in tracks:
        if tr is None or len(tr) < 2:
            continue
        segs.append(np.stack([tr[:-1, :], tr[1:, :]], axis=1))
    if segs:
        lc = LineCollection(np.concatenate(segs, axis=0), linewidths=1.5, alpha=0.9)
        ax.add_collection(lc)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ψ_eff")
    fig.tight_layout()
    return fig

# ----------------------------
# Public API
# ----------------------------

def run(
    seed: int = 42,
    n_agents: int = 4,
    steps: int = 300,
    field: str = "hillvalley",
    overlay_curvature: bool = False,
    overlay_alpha: float = 0.25,
    save: Optional[str] = None,
    no_show: bool = False,
    dump_11d_means: bool = False,
) -> None:
    """
    Run the Symbolic Gravity simulator with 11D upgrades.
    Produces an image and a JSON report (and optional CSV of 11D means).
    """
    rng = np.random.default_rng(seed)

    # ---- Generate base field & ψ (robust)
    psi_base = _generate_field_robust(field, rng)
    psi = _compute_psi_eff_robust(psi_base)
    if not isinstance(psi, np.ndarray) or psi.ndim != 2:
        raise ValueError("compute_psi_eff() must return a 2D numpy array.")

    shape = psi.shape
    curv_map = curvature(psi)

    # ---- 11D fields
    F11 = init_11d_fields(shape, rng, psi, noise=0.05)
    thr = ElevenDThresholds()

    # ---- Sample agents & build
    starts = _sample_starts(psi, curv_map, n_agents, rng)
    agents = _build_agents(starts, rng)

    # ---- Simulate
    collapsed_mask = collapse_predicate_11d(F11, thr)
    route_map = route_action(F11, collapsed_mask, thr)
    dtp = float(np.mean(time_dilation_dt(1.0, F11, thr)))

    omega_mean = float(np.mean(F11["Omega"])) if "Omega" in F11 else 0.5

    any_alive = True
    step_count = 0
    while any_alive and step_count < steps:
        any_alive = False
        collapsed_mask = collapse_predicate_11d(F11, thr)
        route_map = route_action(F11, collapsed_mask, thr)
        dtp = float(np.mean(time_dilation_dt(1.0, F11, thr)))

        for ag in agents:
            alive = getattr(ag, "alive", True)
            if not alive:
                continue
            any_alive = True

            cfg = getattr(ag, "cfg", getattr(ag, "config", None))
            if cfg is not None and hasattr(cfg, "orbit_bias"):
                try:
                    cfg.orbit_bias = max(0.0, min(1.0, float(cfg.orbit_bias) + 0.02 * (omega_mean - 0.5)))
                except Exception:
                    pass

            stepped = False
            try:
                ag.step(dt=dtp); stepped = True
            except TypeError:
                pass
            if not stepped:
                try:
                    ag.step(); stepped = True
                except Exception:
                    if hasattr(ag, "tick"):
                        ag.tick(); stepped = True
            if not stepped:
                raise RuntimeError("Agent could not be advanced with known step/tick signatures.")

        step_count += 1

    # ---- Visualization
    tracks = _collect_tracks(agents)
    fig = _plot(
        psi=psi,
        tracks=tracks,
        curv_map=curv_map,
        overlay_curvature=overlay_curvature,
        overlay_alpha=overlay_alpha,
        route_preview=route_map
    )

    # ---- Output
    base_prefix = "run"
    if save:
        os.makedirs(os.path.dirname(save) if os.path.dirname(save) else ".", exist_ok=True)
        base_prefix = save
    out_img = f"{base_prefix}.png"
    fig.savefig(out_img, dpi=144)

    # Agent report
    report = []
    for i, ag in enumerate(agents):
        steps_done = getattr(ag, "steps", step_count)
        status = getattr(ag, "collapse_reason", None)
        if status is None:
            status = "active" if getattr(ag, "alive", False) else "unknown"
        tr = getattr(ag, "track", None)
        if tr is None or len(tr) == 0:
            start_xy = [None, None]; end_xy = [None, None]
        else:
            start_xy = list(map(float, np.array(tr[0][::-1], dtype=float)))
            end_xy = list(map(float, np.array(tr[-1][::-1], dtype=float)))
        report.append({"id": i, "steps": int(steps_done), "status": status, "start": start_xy, "end": end_xy})

    with open(base_prefix + ".json", "w", encoding="utf-8") as f:
        json.dump({"agents": report}, f, indent=2)
    print(f"Saved {out_img} and {base_prefix + '.json'}")

    # Optional: dump 11D means
    if dump_11d_means:
        try:
            import csv
            with open(base_prefix + "_11d_means.csv", "w", newline="", encoding="utf-8") as cf:
                writer = csv.writer(cf)
                writer.writerow(["field", "mean"])
                for k, v in F11.items():
                    writer.writerow([k, float(np.nanmean(v))])
            print(f"Saved {base_prefix + '_11d_means.csv'}")
        except Exception:
            pass

    # Show or close
    if not no_show:
        plt.show()
    plt.close(fig)

# ----------------------------
# Optional CLI (module entry)
# ----------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Symbolic Gravity Simulator (11D upgrade)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_agents", type=int, default=4)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--field", type=str, default="hillvalley")
    p.add_argument("--overlay_curvature", action="store_true")
    p.add_argument("--overlay_alpha", type=float, default=0.25)
    p.add_argument("--save", type=str, default="runs/run_cli")
    p.add_argument("--no_show", action="store_true")
    p.add_argument("--dump_11d_means", action="store_true")
    return p.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    run(
        seed=args.seed,
        n_agents=args.n_agents,
        steps=args.steps,
        field=args.field,
        overlay_curvature=args.overlay_curvature,
        overlay_alpha=args.overlay_alpha,
        save=args.save,
        no_show=args.no_show,
        dump_11d_means=args.dump_11d_means,
    )

if __name__ == "__main__":
    main()
