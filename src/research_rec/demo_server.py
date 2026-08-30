from __future__ import annotations

import argparse
import csv
import json
import mimetypes
from collections import Counter, defaultdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


class DemoDataStore:
    """Lazy reader for the label-blind Pure prediction export."""

    def __init__(self, prediction_path: str | Path):
        self.prediction_path = Path(prediction_path)
        self._rankings: dict[str, list[dict[str, int | float]]] | None = None
        self._counts: Counter[str] | None = None

    def _load(self) -> None:
        if self._rankings is not None:
            return
        if not self.prediction_path.is_file():
            raise FileNotFoundError(f"Prediction export not found: {self.prediction_path}")
        rankings: dict[str, list[dict[str, int | float]]] = defaultdict(list)
        counts: Counter[str] = Counter()
        with self.prediction_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                user_id = row["user_id"]
                counts[user_id] += 1
                rankings[user_id].append(
                    {
                        "video_id": int(row["video_id"]),
                        "score": float(row["score"]),
                        "rank": int(row["rank"]),
                    }
                )
        for rows in rankings.values():
            rows.sort(key=lambda row: int(row["rank"]))
        self._rankings = dict(rankings)
        self._counts = counts

    def ranking(self, user_id: str, limit: int = 8) -> list[dict[str, int | float]]:
        self._load()
        assert self._rankings is not None
        return self._rankings.get(str(user_id), [])[:limit]

    def sample_users(self, limit: int = 8) -> list[dict[str, int | str]]:
        self._load()
        assert self._counts is not None
        return [{"user_id": user_id, "candidates": count} for user_id, count in self._counts.most_common(limit)]


def create_handler(repository_root: Path, store: DemoDataStore) -> type[BaseHTTPRequestHandler]:
    class DemoHandler(BaseHTTPRequestHandler):
        def _json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            parsed = urlparse(self.path)
            if parsed.path == "/api/health":
                self._json({"status": "ok", "task": "short-video click ranking", "fake_video_detection": False})
                return
            if parsed.path == "/api/sample-users":
                try:
                    self._json({"users": store.sample_users()})
                except FileNotFoundError as error:
                    self._json({"error": str(error)}, HTTPStatus.SERVICE_UNAVAILABLE)
                return
            if parsed.path == "/api/rank":
                query = parse_qs(parsed.query)
                user_id = query.get("user_id", [""])[0]
                try:
                    limit = min(50, max(1, int(query.get("limit", ["8"])[0])))
                except ValueError:
                    self._json({"error": "limit must be an integer"}, HTTPStatus.BAD_REQUEST)
                    return
                rows = store.ranking(user_id, limit)
                if not rows:
                    self._json({"error": "user_id not present in the exported candidate set"}, HTTPStatus.NOT_FOUND)
                    return
                self._json({"user_id": user_id, "rows": rows, "source": "validation-selected label-blind export"})
                return
            self._serve_static(parsed.path)

        def _serve_static(self, request_path: str) -> None:
            if request_path in {"/", "/index.html"}:
                relative = "frontend/index.html"
            else:
                requested = unquote(request_path.lstrip("/"))
                frontend_candidate = repository_root / "frontend" / requested
                relative = f"frontend/{requested}" if frontend_candidate.is_file() else requested
            candidate = (repository_root / relative).resolve()
            try:
                candidate.relative_to(repository_root)
            except ValueError:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            if not candidate.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content = candidate.read_bytes()
            content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8" if content_type.startswith("text/") else content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format: str, *args: object) -> None:
            return

    return DemoHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the LoopRank frontend and verified prediction explorer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--predictions", default="artifacts/submissions/kuairand_pure_scores.csv")
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    store = DemoDataStore(repository_root / args.predictions)
    server = ThreadingHTTPServer((args.host, args.port), create_handler(repository_root, store))
    print(f"LoopRank frontend: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nLoopRank frontend stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
