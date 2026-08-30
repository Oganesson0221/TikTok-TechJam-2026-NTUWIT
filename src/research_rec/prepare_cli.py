from __future__ import annotations

import argparse
import json

from .prepare import prepare_kuairand


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare fixed KuaiRand-Pure splits")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--variant", choices=["pure", "1k"], default="pure")
    args = parser.parse_args()
    print(json.dumps(prepare_kuairand(args.data_root, args.download, args.force, args.variant), indent=2))


if __name__ == "__main__":
    main()
