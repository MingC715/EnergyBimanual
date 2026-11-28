"""Compatibility entry point for collecting Table I results."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from summarize_results import main


if __name__ == "__main__":
    raise SystemExit(main(["rlbench", *sys.argv[1:]]))
