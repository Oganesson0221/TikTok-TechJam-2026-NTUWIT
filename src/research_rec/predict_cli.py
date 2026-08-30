from __future__ import annotations

import argparse
import json
from pathlib import Path

from .predict import export_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ranked click predictions from a validation-best checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--column-map", help="JSON file mapping internal column names to organizer names")
    parser.add_argument(
        "--columns",
        help="Comma-separated internal columns to emit (default: user_id,video_id,score,rank)",
    )
    args = parser.parse_args()
    column_mapping = json.loads(Path(args.column_map).read_text(encoding="utf-8")) if args.column_map else None
    output_columns = [column.strip() for column in args.columns.split(",")] if args.columns else None
    summary = export_predictions(
        args.checkpoint,
        args.input_csv,
        args.output_csv,
        top_k=args.top_k,
        batch_size=args.batch_size,
        device_name=args.device,
        limit=args.limit,
        column_mapping=column_mapping,
        output_columns=output_columns,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
