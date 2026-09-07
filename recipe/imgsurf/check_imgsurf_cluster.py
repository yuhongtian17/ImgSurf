#!/usr/bin/env python3
"""Fail fast when a manually started Ray cluster cannot satisfy ImgSurf."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@ray.remote(num_cpus=0)
def run_runtime_check(runtime_check_script: str) -> tuple[str, int, str]:
    """Run the same dependency/API check in a process pinned to one node."""
    result = subprocess.run(
        [sys.executable, runtime_check_script],
        capture_output=True,
        check=False,
        text=True,
    )
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    return ray.util.get_node_ip_address(), result.returncode, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--runtime-check-script", type=Path, required=True)
    args = parser.parse_args()

    runtime_check_script = args.runtime_check_script.resolve()
    if not runtime_check_script.is_file():
        raise FileNotFoundError(f"Runtime check script does not exist: {runtime_check_script}")

    address = os.environ.get("RAY_ADDRESS", "auto")
    ray.init(address=address, ignore_reinit_error=True)
    try:
        eligible_nodes = []
        for node in ray.nodes():
            if not node.get("Alive", False):
                continue
            gpu_count = int(node.get("Resources", {}).get("GPU", 0))
            if gpu_count >= args.gpus_per_node:
                eligible_nodes.append(
                    (
                        node.get("NodeID"),
                        node.get("NodeManagerAddress", "unknown"),
                        gpu_count,
                    )
                )

        if len(eligible_nodes) < args.nnodes:
            details = ", ".join(f"{host}:{count}GPU" for _, host, count in eligible_nodes) or "none"
            raise RuntimeError(
                f"Ray has only {len(eligible_nodes)} eligible node(s), but ImgSurf requires "
                f"{args.nnodes} node(s) with at least {args.gpus_per_node} GPU(s) each. "
                f"Eligible nodes: {details}"
            )

        selected = eligible_nodes[: args.nnodes]
        checks = [
            run_runtime_check.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
            ).remote(str(runtime_check_script))
            for node_id, _, _ in selected
        ]
        failed_checks = []
        for host, returncode, output in ray.get(checks):
            if returncode != 0:
                failed_checks.append(f"{host}:\n{output}")
        if failed_checks:
            raise RuntimeError(
                "ImgSurf runtime validation failed on Ray node(s):\n" + "\n\n".join(failed_checks)
            )

        print(
            "Ray cluster check OK: "
            + ", ".join(f"{host}:{count}GPU" for _, host, count in selected)
        )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
