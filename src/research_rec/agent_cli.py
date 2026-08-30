from __future__ import annotations

import argparse
import json

from .agent import load_agent_config, run_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded autonomous recommender experiments")
    parser.add_argument("--config", default="configs/agent.yaml")
    parser.add_argument("--smoke", action="store_true", help="Limit each candidate to 100k/20k rows and two epochs")
    args = parser.parse_args()
    print(json.dumps(run_agent(load_agent_config(args.config), smoke=args.smoke), indent=2))


if __name__ == "__main__":
    main()

