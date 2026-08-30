from __future__ import annotations

import argparse
import json

from .baseline import run_popularity_baseline
from .prepare import prepare_kuairand


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare KuaiRand-Pure and run the reproducible click baseline")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="artifacts/baselines/item_popularity")
    parser.add_argument("--download", action="store_true", help="Download the official archive if needed")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--smoothing", type=float, default=20.0)
    parser.add_argument("--official-ndcg", type=float)
    parser.add_argument("--official-recall", type=float)
    args = parser.parse_args()
    manifest = prepare_kuairand(args.data_root, args.download, args.force_prepare)
    result = run_popularity_baseline(
        args.data_root,
        args.output_dir,
        args.smoothing,
        args.official_ndcg,
        args.official_recall,
    )
    print(json.dumps({"data_manifest": manifest, "baseline_result": result}, indent=2))


if __name__ == "__main__":
    main()

