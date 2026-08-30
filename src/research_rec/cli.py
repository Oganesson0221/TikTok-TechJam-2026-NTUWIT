from __future__ import annotations

import argparse
import json
from typing import Any

from .config import apply_override, load_config
from .training import train_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate a configurable KuaiRand recommender")
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override a configuration value; may be supplied multiple times",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    for expression in args.set:
        apply_override(config, expression)
    config.validate()
    summary = train_experiment(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
