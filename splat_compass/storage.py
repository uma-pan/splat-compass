"""試合ごとの記録を、使う人のPCの中に貯めていく。

data/matches/ に1試合1ファイル（JSON）で保存する。動画や他のプレイヤーの名前は保存しない。
"""
import datetime as dt
import glob
import json
import os

from .tracker import ROW_HEADER

DEFAULT_DIR = os.path.join(os.path.expanduser("~"), "splat-compass-data")


class Store:
    def __init__(self, root=None):
        self.root = root or os.environ.get("SPLAT_COMPASS_DATA", DEFAULT_DIR)
        self.mdir = os.path.join(self.root, "matches")
        os.makedirs(self.mdir, exist_ok=True)

    def save_match(self, m, source, started_at=None):
        """started_at: 試合が始まった実際の日時（分からなければ今）。"""
        when = started_at or dt.datetime.now()
        name = f"{when:%Y%m%d-%H%M%S}"
        path = os.path.join(self.mdir, name + ".json")
        n = 1
        while os.path.exists(path):
            n += 1
            path = os.path.join(self.mdir, f"{name}-{n}.json")
        data = {
            "played_at": when.isoformat(timespec="seconds"),
            "source": source,
            "video_start": round(m.start, 1),
            "video_end": round(m.end, 1) if m.end is not None else None,
            "rule": m.rule,
            "result": m.result,
            "columns": ROW_HEADER,
            "events": [e.row() for e in m.events],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        return path

    def matches(self):
        out = []
        for p in sorted(glob.glob(os.path.join(self.mdir, "*.json"))):
            with open(p, encoding="utf-8") as f:
                out.append(json.load(f))
        return out

    # 録画フォルダの処理済み一覧
    def _done_path(self):
        return os.path.join(self.root, "processed_videos.json")

    def processed(self):
        try:
            with open(self._done_path(), encoding="utf-8") as f:
                return set(json.load(f))
        except (OSError, ValueError):
            return set()

    def mark_processed(self, key):
        s = self.processed()
        s.add(key)
        with open(self._done_path(), "w", encoding="utf-8") as f:
            json.dump(sorted(s), f, ensure_ascii=False, indent=0)
