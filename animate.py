"""Create an offline animated presentation from run.py outputs."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import webbrowser

from headway.animation import Walkthrough, export_html
from headway.distributions import DISTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["highd", "ngsim"], default="highd")
    parser.add_argument("--results", type=Path, default=Path("outputs"), help="Existing run.py output root")
    parser.add_argument("--model", choices=list(DISTS), default="Proposed", help="Model for detailed MCMC/posterior views")
    parser.add_argument("--out", type=Path, help="HTML destination (default: <results>/animations/<dataset>.html)")
    parser.add_argument("--demo", action="store_true", help="Run a synthetic pipeline in outputs/animation_demo; never reads raw data")
    parser.add_argument("--open", action="store_true", help="Open the finished HTML in your browser")
    parser.add_argument("--frames-per-stage", type=int, default=16, help="2..60; default 16 (each stage lasts 4 seconds)")
    args = parser.parse_args()
    if not 2 <= args.frames_per_stage <= 60:
        parser.error("--frames-per-stage must be between 2 and 60")
    if args.demo:
        args.results = Path("outputs/animation_demo").resolve()
        # A new empty directory guarantees the demo cannot discover real datasets.
        with tempfile.TemporaryDirectory(prefix="headway-animation-") as empty:
            subprocess.run([sys.executable, str(Path(__file__).with_name("run.py")),
                            "--dataset", args.dataset, "--data-dir", empty,
                            "--synthetic-size", "2000", "--workers", "1", "--out", str(args.results)], check=True)
    try:
        walkthrough = Walkthrough(args.results, args.dataset, args.model)
        destination = args.out or args.results / "animations" / f"{args.dataset}.html"
        path = export_html(walkthrough, destination, args.frames_per_stage)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Animation could not be built: {exc}\nUse complete outputs from run.py, or try --demo.\n")
    print(f"Animation saved: {path.resolve()} ({path.stat().st_size / 1024**2:.1f} MB)")
    if args.open:
        webbrowser.open(path.resolve().as_uri())


if __name__ == "__main__":
    main()
