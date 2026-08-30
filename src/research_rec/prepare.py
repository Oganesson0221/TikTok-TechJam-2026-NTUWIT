from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tarfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

import pandas as pd


KUAI_RAND_URL = "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz"
KUAI_RAND_MD5 = "0820331067a3784d9691136f772b35a7"
TRAIN_FILENAME = "log_standard_4_08_to_4_21_pure.csv"
LATE_FILENAME = "log_standard_4_22_to_5_08_pure.csv"
USER_FEATURES_FILENAME = "user_features_pure.csv"
ITEM_FEATURES_FILENAME = "video_features_basic_pure.csv"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    archive_name: str
    url: str
    md5: str
    train_filename: str
    late_filename: str
    user_features_filename: str
    item_features_filename: str


DATASET_SPECS = {
    "pure": DatasetSpec(
        "KuaiRand-Pure", "KuaiRand-Pure.tar.gz", KUAI_RAND_URL, KUAI_RAND_MD5,
        TRAIN_FILENAME, LATE_FILENAME, USER_FEATURES_FILENAME, ITEM_FEATURES_FILENAME,
    ),
    "1k": DatasetSpec(
        "KuaiRand-1K",
        "KuaiRand-1K.tar.gz",
        "https://zenodo.org/records/10439422/files/KuaiRand-1K.tar.gz",
        "6b0b9c8222d67fcd4c676218edca3f1f",
        "log_standard_4_08_to_4_21_1k.csv",
        "log_standard_4_22_to_5_08_1k.csv",
        "user_features_1k.csv",
        "video_features_basic_1k.csv",
    ),
}


@dataclass(frozen=True)
class SplitSummary:
    rows: int
    min_time_ms: int
    max_time_ms: int
    positives: int
    users: int
    items: int
    sha256: str


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_dataset(destination: Path, url: str = KUAI_RAND_URL, expected_md5: str = KUAI_RAND_MD5) -> Path:
    """Download the organizer archive atomically and verify its published MD5."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and file_digest(destination, "md5") == expected_md5:
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "kuairand-techjam-agent/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    actual = file_digest(temporary, "md5")
    if actual != expected_md5:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"KuaiRand archive checksum mismatch: expected {expected_md5}, got {actual}")
    os.replace(temporary, destination)
    return destination


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Unsafe path in dataset archive: {member.name}")
        bundle.extractall(destination, filter="data")


def _locate(raw_root: Path, filename: str) -> Path:
    matches = list(raw_root.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {filename} under {raw_root}, found {len(matches)}")
    return matches[0]


def _count_csv_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _copy_rows(reader: csv.reader, writer: csv.writer, count: int) -> None:
    for _ in range(count):
        try:
            writer.writerow(next(reader))
        except StopIteration as exc:
            raise ValueError("Interaction CSV ended before its expected row count") from exc


def split_late_interactions(source: Path, validation_path: Path, test_path: Path) -> tuple[int, int]:
    """Split the later standard log by row order into its fixed first/last halves."""
    total = _count_csv_rows(source)
    validation_rows = total // 2
    test_rows = total - validation_rows
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_tmp = validation_path.with_suffix(".csv.tmp")
    test_tmp = test_path.with_suffix(".csv.tmp")
    with source.open("r", encoding="utf-8", newline="") as input_handle, validation_tmp.open(
        "w", encoding="utf-8", newline=""
    ) as validation_handle, test_tmp.open("w", encoding="utf-8", newline="") as test_handle:
        reader = csv.reader(input_handle)
        header = next(reader)
        validation_writer = csv.writer(validation_handle)
        test_writer = csv.writer(test_handle)
        validation_writer.writerow(header)
        test_writer.writerow(header)
        _copy_rows(reader, validation_writer, validation_rows)
        _copy_rows(reader, test_writer, test_rows)
        if next(reader, None) is not None:
            raise ValueError("Interaction CSV contained more rows than expected")
    os.replace(validation_tmp, validation_path)
    os.replace(test_tmp, test_path)
    return validation_rows, test_rows


def summarize_split(path: Path) -> SplitSummary:
    columns = ["user_id", "video_id", "time_ms", "is_click"]
    totals = {"rows": 0, "positives": 0}
    min_time: int | None = None
    max_time: int | None = None
    users: set[int] = set()
    items: set[int] = set()
    for chunk in pd.read_csv(path, usecols=columns, chunksize=250_000):
        if chunk[columns].isna().any().any():
            raise ValueError(f"Null required values found in {path}")
        if not chunk["is_click"].isin([0, 1]).all():
            raise ValueError(f"Non-binary is_click values found in {path}")
        totals["rows"] += len(chunk)
        totals["positives"] += int(chunk["is_click"].sum())
        chunk_min = int(chunk["time_ms"].min())
        chunk_max = int(chunk["time_ms"].max())
        min_time = chunk_min if min_time is None else min(min_time, chunk_min)
        max_time = chunk_max if max_time is None else max(max_time, chunk_max)
        users.update(chunk["user_id"].astype(int).unique().tolist())
        items.update(chunk["video_id"].astype(int).unique().tolist())
    if totals["rows"] == 0 or min_time is None or max_time is None:
        raise ValueError(f"Prepared split is empty: {path}")
    return SplitSummary(
        rows=totals["rows"],
        min_time_ms=min_time,
        max_time_ms=max_time,
        positives=totals["positives"],
        users=len(users),
        items=len(items),
        sha256=file_digest(path),
    )


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def prepare_kuairand(
    data_root: str | Path = "data", download: bool = False, force: bool = False, variant: str = "pure"
) -> dict[str, object]:
    if variant not in DATASET_SPECS:
        raise ValueError(f"variant must be one of: {', '.join(DATASET_SPECS)}")
    spec = DATASET_SPECS[variant]
    root = Path(data_root)
    archive = root / "downloads" / spec.archive_name
    raw_root = root / "raw"
    prepared = root / "prepared"
    manifest_path = prepared / "manifest.json"
    if manifest_path.is_file() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset") == spec.name:
            return manifest
    if download:
        download_dataset(archive, spec.url, spec.md5)
    if not raw_root.exists() or not list(raw_root.rglob(spec.train_filename)):
        if not archive.is_file():
            raise FileNotFoundError(f"Dataset archive not found at {archive}; rerun with --download")
        if file_digest(archive, "md5") != spec.md5:
            raise ValueError("Dataset archive does not match the organizer-published MD5")
        _safe_extract(archive, raw_root)

    raw_train = _locate(raw_root, spec.train_filename)
    raw_late = _locate(raw_root, spec.late_filename)
    train_path = prepared / "train.csv"
    validation_path = prepared / "validation.csv"
    test_path = prepared / "test.csv"
    _copy_atomic(raw_train, train_path)
    validation_rows, test_rows = split_late_interactions(raw_late, validation_path, test_path)
    _copy_atomic(_locate(raw_root, spec.user_features_filename), prepared / spec.user_features_filename)
    _copy_atomic(_locate(raw_root, spec.item_features_filename), prepared / spec.item_features_filename)

    summaries = {
        "train": summarize_split(train_path),
        "validation": summarize_split(validation_path),
        "test": summarize_split(test_path),
    }
    train_validation_time_overlap = summaries["train"].max_time_ms >= summaries["validation"].min_time_ms
    if train_validation_time_overlap and variant == "pure":
        raise ValueError("Train/validation timestamp leakage detected")
    # The official instruction says row halves. Record, but do not fail, if the
    # source is not globally time-sorted and the halves overlap in timestamps.
    temporal_half_overlap = summaries["validation"].max_time_ms >= summaries["test"].min_time_ms
    manifest: dict[str, object] = {
        "dataset": spec.name,
        "source_url": spec.url,
        "archive_md5": spec.md5,
        "split_rule": "4/08-4/21 standard=train; first 50% of 4/22-5/08 standard=validation; last 50%=test",
        "validation_rows_expected": validation_rows,
        "test_rows_expected": test_rows,
        "temporal_half_overlap": temporal_half_overlap,
        "train_validation_time_overlap": train_validation_time_overlap,
        "splits": {name: asdict(summary) for name, summary in summaries.items()},
        "static_features": [spec.user_features_filename, spec.item_features_filename],
        "excluded_leakage_risk_file": f"video_features_statistic_{variant}.csv",
    }
    prepared.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
