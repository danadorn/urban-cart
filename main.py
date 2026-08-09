import argparse
import subprocess
from pathlib import Path
from shutil import which

from pipeline import run_phase2, run_phase3, run_phase4

FIG_DIR = Path("figures")


def list_figures() -> list[Path]:
    if not FIG_DIR.exists():
        return []
    return sorted(FIG_DIR.glob("*.png"))


def open_figures(fig_paths: list[Path]) -> None:
    opener = None
    if which("xdg-open"):
        opener = ["xdg-open"]
    elif which("gio"):
        opener = ["gio", "open"]

    if opener is None:
        print("No supported image opener found (xdg-open/gio). Skipping open step.")
        return

    for fig in fig_paths:
        print(f"Opening {fig}")
        subprocess.run(opener + [str(fig)], check=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full Phase 2-4 pipeline and report generated figure images."
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open generated PNG chart files after the pipeline completes.",
    )
    args = parser.parse_args()

    run_phase2()
    run_phase3()
    run_phase4()

    figures = list_figures()
    if figures:
        print(f"\nGenerated {len(figures)} figure files in {FIG_DIR}:")
        for fig in figures:
            print(f"  - {fig}")
        if args.open:
            open_figures(figures)
    else:
        print(f"\nNo figure files found in {FIG_DIR}. Make sure phase 4 completed successfully.")


if __name__ == "__main__":
    main()