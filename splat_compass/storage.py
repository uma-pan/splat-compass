"""試合ごとの記録を、使う人のPCの中に貯めていく。

data/matches/ に1試合1ファイル（JSON）で保存する。動画や他のプレイヤーの名前は保存しない。
"""
import datetime as dt
import glob
import json
import os
import tempfile

from .hud import SLOT_NAMES
from .tracker import ROW_HEADER

DEFAULT_DIR = os.path.join(os.path.expanduser("~"), "splat-compass-data")
MODES = ("ガチ", "エンジョイ")  # 本気でやる記録と、そうでない記録（練習・遊びなど）を分ける
_OLD_MODE = {"Xマッチ": "ガチ", "それ以外": "エンジョイ"}  # 前の呼び方


def mode_name(v):
    return _OLD_MODE.get(v, v)


def mode_of(m):
    """試合の種類。種類を記録する前の試合はガチとみなす。"""
    return mode_name(m.get("mode")) or "ガチ"


def _write_json(path, data):
    """集計が読み取りの途中で未完成の JSON に当たらないようにする。"""
    tmp = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=os.path.dirname(path), delete=False)
    try:
        with tmp:
            json.dump(data, tmp, ensure_ascii=False, indent=1)
        os.replace(tmp.name, path)
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


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
        stem0 = os.path.splitext(os.path.basename(path))[0]
        if getattr(m, "stage_crop", None) is not None:  # ステージの名前の部分（あとで名前を付け直せるように）
            import cv2
            sdir = os.path.join(self.root, "stages")
            os.makedirs(sdir, exist_ok=True)
            ok, buf = cv2.imencode(".png", m.stage_crop[..., ::-1])
            if ok:
                buf.tofile(os.path.join(sdir, stem0 + ".png"))
        killed_by = []
        if getattr(m, "killcams", None):
            import cv2
            kdir = os.path.join(self.root, "killcam")
            os.makedirs(kdir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(path))[0]
            for i, (t, img) in enumerate(m.killcams):
                fname = f"{stem}_{i:02d}.png"
                ok, buf = cv2.imencode(".png", img[..., ::-1])
                if ok:
                    buf.tofile(os.path.join(kdir, fname))
                    killed_by.append({"t": round(t, 1), "rel": round(t - m.start, 1), "image": fname})
        map_shots = []
        if getattr(m, "map_shots", None):
            import cv2
            mdir = os.path.join(self.root, "maps")
            os.makedirs(mdir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(path))[0]
            for i, (t, img) in enumerate(m.map_shots):
                fname = f"{stem}_{i:02d}.jpg"
                ok, buf = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    buf.tofile(os.path.join(mdir, fname))
                    map_shots.append({"t": round(t, 1), "rel": round(t - m.start, 1), "image": fname})
        result_check = None
        if getattr(m, "result_check", None):
            import cv2
            img, scores = m.result_check
            ddir = os.path.join(self.root, "results")
            os.makedirs(ddir, exist_ok=True)
            fname = os.path.splitext(os.path.basename(path))[0] + ".jpg"
            ok, buf = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                buf.tofile(os.path.join(ddir, fname))
                result_check = {"image": fname, "win": scores[0], "lose": scores[1]}
        weapon_icons = None
        icons = getattr(m, "icons", None)
        if icons and any(i is not None for i in icons):
            from . import weapons
            weapon_icons = weapons.save_strip(self, os.path.splitext(os.path.basename(path))[0], icons)
        data = {
            "played_at": when.isoformat(timespec="seconds"),
            "source": source,
            "video_start": round(m.start, 1),
            "video_end": round(m.end, 1) if m.end is not None else None,
            "rule": m.rule,
            "stage": getattr(m, "stage", None),
            "mode": getattr(m, "mode", None) or self.mode(),
            "go": getattr(m, "go", None),
            "result": m.result,
            "interrupted": m.interrupted,
            "me_slot": m.me_slot,
            "columns": ROW_HEADER,
            "events": [e.row() for e in m.events],
            "timeline_columns": ["時刻", "味方カウント", "敵カウント", "味方の箱が塗られている", "敵の箱が塗られている",
                                 "味方ペナルティ", "敵ペナルティ"],
            "timeline": m.timeline,
            "objective_columns": ["時刻", "ヤグラ・ホコの位置（-1が味方のゴール、+1が敵のゴール）", "アサリのゴールが開いている"],
            "objective": m.objective,
            "checkpoints": m.checkpoints,
            "killed_by": killed_by,
            "map_shots": map_shots,
            "inks": [[t, round(t - m.start, 1), v] for t, v in getattr(m, "inks", [])],
            "weapon_icons": weapon_icons,
            "result_check": result_check,
            "kills": [{"t": round(t, 1), "rel": round(t - m.start, 1)} for t in getattr(m, "kills", [])],
        }
        if m.rule == "ガチアサリ":
            data["clam_columns"] = ["時刻", *SLOT_NAMES]
            data["clam_timeline"] = getattr(m, "clam_timeline", [])
        _write_json(path, data)
        return path

    def matches(self, all=False):
        """試合の記録の一覧。設定の「見る記録」（ガチだけ・エンジョイだけ）で絞る。all=True なら絞らない
        （勝率モデルやブキの見本は、どの試合でも同じように使えるので全部使う）。"""
        view = None if all else self.view()
        out = []
        cache = self.__dict__.setdefault("_cache", {})
        seen = set()
        for p in sorted(glob.glob(os.path.join(self.mdir, "*.json"))):
            # 読んだことがあり、変わっていないファイルは読み直さない（まとめでは何度も一覧を取るので速くなる）
            try:
                st = os.stat(p)
            except OSError:
                continue
            key = (st.st_mtime_ns, st.st_size)
            seen.add(p)
            hit = cache.get(p)
            if hit is None or hit[0] != key:
                with open(p, encoding="utf-8") as f:
                    d = json.load(f)
                d["_file"] = os.path.basename(p)
                cache[p] = hit = (key, d)
            d = dict(hit[1])  # 呼び出し側が項目を書き換えても、覚えている分は変わらないように
            if view and mode_of(d) != view:
                continue
            out.append(d)
        for p in list(cache):
            if p not in seen:
                del cache[p]
        return out

    def view(self):
        """まとめや振り返りで見る記録の種類（"ガチ" / "エンジョイ"）。None なら全部。"""
        v = mode_name(self.load_config().get("view", "ガチ"))
        return v if v in MODES else None

    def mode(self):
        """これから保存する試合の種類（画面で切り替える）。"""
        v = mode_name(self.load_config().get("mode", "ガチ"))
        return v if v in MODES else "ガチ"

    def set_result(self, fname, result):
        """勝敗を手で直す（読み違えていたとき）。"""
        p = os.path.join(self.mdir, os.path.basename(fname))
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        d["result"] = result
        d["interrupted"] = result is None and d.get("interrupted", False)
        d["result_fixed"] = True
        _write_json(p, d)

    def update(self, fname, fields):
        """試合の記録の一部の項目を書き換える。"""
        p = os.path.join(self.mdir, os.path.basename(fname))
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        d.update(fields)
        _write_json(p, d)

    def load(self, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        d["_file"] = os.path.basename(path)
        return d

    def reset_videos(self):
        """録画から作った記録を matches/_old/日時/ に移し、処理済み一覧を空にする（消しはしない）。"""
        import shutil
        old = os.path.join(self.mdir, "_old", f"{dt.datetime.now():%Y%m%d-%H%M%S}")
        n = 0
        for p in glob.glob(os.path.join(self.mdir, "*.json")):
            with open(p, encoding="utf-8") as f:
                src = json.load(f).get("source", "")
            if src.lower().endswith((".mp4", ".mkv", ".mov")):
                os.makedirs(old, exist_ok=True)
                shutil.move(p, os.path.join(old, os.path.basename(p)))
                n += 1
        if os.path.exists(self._done_path()):
            os.replace(self._done_path(), self._done_path() + ".bak")
        return n

    # 設定（録画のフォルダなど）
    def load_config(self):
        try:
            with open(os.path.join(self.root, "config.json"), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def save_config(self, cfg):
        _write_json(os.path.join(self.root, "config.json"), cfg)

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
