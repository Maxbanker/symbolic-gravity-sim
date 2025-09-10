# symbolic_gravity_sim/cli.py
import argparse
import yaml
import os
import subprocess
from .symbolic_collapse_sim import run

def load_config(config_path):
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def main():
    parser = argparse.ArgumentParser(description="Symbolic Gravity Simulator (v2.2)")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    run_parser = subparsers.add_parser("run", help="Run a single simulation")
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--n-agents", type=int, default=4)  # Changed from --agents to --n-agents
    run_parser.add_argument("--steps", type=int, default=300)
    run_parser.add_argument("--field", type=str, default="hillvalley",
                            choices=["hillvalley", "gaussian", "uniform", "gradient"])
    run_parser.add_argument("--height", type=int, default=120)
    run_parser.add_argument("--width", type=int, default=160)
    run_parser.add_argument("--orbit_bias", type=float, default=0.55)
    run_parser.add_argument("--collapse_q", type=float, default=0.15)
    run_parser.add_argument("--curvature_q", type=float, default=0.92)
    run_parser.add_argument("--curv_consecutive", type=int, default=4)
    run_parser.add_argument("--curv_hysteresis", type=float, default=0.85)
    run_parser.add_argument("--stagnation_tol", type=float, default=0.25)
    run_parser.add_argument("--stagnation_std", type=float, default=0.05)
    run_parser.add_argument("--stagnation_window", type=int, default=12)
    run_parser.add_argument("--max_age", type=int, default=10000)
    run_parser.add_argument("--psi_p1", type=float, default=1.0)
    run_parser.add_argument("--psi_p99", type=float, default=99.0)
    run_parser.add_argument("--start_qpsi", type=float, default=0.60)
    run_parser.add_argument("--start_qcurv", type=float, default=0.75)
    run_parser.add_argument("--start_min_dist", type=float, default=10.0)
    run_parser.add_argument("--save", type=str, default=None)
    run_parser.add_argument("--no-show", action="store_true")
    run_parser.add_argument("--overlay_curvature", action="store_true")
    run_parser.add_argument("--overlay_alpha", type=float, default=0.25)

    subparsers.add_parser("bench", help="Run parameter sweeps (TBD)")
    subparsers.add_parser("render", help="Render saved run (TBD)")

    args = parser.parse_args()
    config = load_config(args.config)
    
    run_args = vars(args).copy()
    run_args.update(config.get("run", {}))
    
    if args.command == "run":
        save_dir = run_args.get("save", None)
        if save_dir:
            os.makedirs("runs", exist_ok=True)
            run_args["save"] = os.path.join("runs", save_dir)
            metadata = {
                "seed": run_args["seed"],
                "git_sha": subprocess.getoutput("git rev-parse HEAD") if os.path.exists(".git") else "N/A",
                "args": {k: v for k, v in run_args.items() if k != "config"}
            }
            with open(os.path.join("runs", f"{save_dir}_metadata.yaml"), "w") as f:
                yaml.dump(metadata, f)
        
        run(**{k: v for k, v in run_args.items() if k != "command" and k != "config"})

if __name__ == "__main__":
    main()