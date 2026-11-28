"""Collect task success rates for one benchmark, method, demonstration budget and seed."""

import argparse
import json
import math
from pathlib import Path
import sys

from evaluate import RLBENCH_TASKS, ROBOTWIN_TASKS


def read_rate(path, task, benchmark):
    if benchmark == "rlbench":
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        try:
            rate = payload[task]["mean"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"expected a mean under {task!r}") from exc
        if type(rate) not in (int, float):
            raise ValueError("the mean must be a number")
    else:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            raise ValueError("empty result file")
        rate = float(lines[-1])
    if not 0 <= rate <= 1:
        raise ValueError("success fraction must be between 0 and 1")
    return rate


def task_result(folder, benchmark, task, selected):
    task_dir = folder / task
    if benchmark == "rlbench":
        return task_dir / "eval.json"
    if task in selected:
        return task_dir / selected[task] / "_result.txt"
    candidates = sorted(task_dir.glob("*/_result.txt"))
    if not candidates:
        raise ValueError("missing rollout result")
    if len(candidates) > 1:
        choices = ", ".join(p.parent.name for p in candidates)
        raise ValueError(f"multiple runs ({choices}). Choose one with --select {task}/RUN")
    return candidates[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="benchmark", required=True)
    for name, label in (("rlbench", "Table I"), ("robotwin", "Table II")):
        mode = modes.add_parser(name, help=label)
        mode.add_argument("--folder", required=True, type=Path,
                          help="seed directory containing the individual task directories")
        if name == "robotwin":
            mode.add_argument("--select", action="append", default=[], metavar="TASK/RUN",
                              help="select the result directory for a task")
    args = parser.parse_args(argv)
    folder = args.folder.expanduser()
    if not folder.is_dir():
        parser.error(f"result directory does not exist: {folder}")
    tasks = RLBENCH_TASKS if args.benchmark == "rlbench" else ROBOTWIN_TASKS
    selected = {}
    for item in getattr(args, "select", []):
        parts = item.split("/")
        if len(parts) != 2 or parts[0] not in tasks or parts[1] in ("", ".", ".."):
            parser.error(f"invalid selection {item!r}. Expected TASK/RUN")
        task, run = parts
        if task in selected:
            parser.error(f"select one run for {task}")
        selected[task] = run

    rates = []
    for task in tasks:
        try:
            path = task_result(folder, args.benchmark, task, selected)
            rate = read_rate(path, task, args.benchmark)
        except (OSError, ValueError) as exc:
            print(f"{task}: {exc}", file=sys.stderr)
            continue
        print(f"{task}: {100 * rate:.2f}%")
        rates.append(rate)
    if len(rates) != len(tasks):
        print(f"Valid task results: {len(rates)}/{len(tasks)}", file=sys.stderr)
        return 1
    print(f"Mean on {len(tasks)} tasks: {100 * math.fsum(rates) / len(tasks):.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
