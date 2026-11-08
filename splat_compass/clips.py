"""録画から場面を切り出して、1本の動画にまとめる。

- 倒された場面（倒される8秒前から3秒後まで）
- 倒した場面（倒す5秒前から2秒後まで）
- 勝率が大きく動いた場面

録画から読んだ試合はその録画を使う。カメラから読んだ試合も、同じ時間に OBS で録画していれば
（ファイル名が OBS の既定の「2026-11-08 20-05-00.mp4」の形なら）、その録画から切り出せる。
ffmpeg があれば音つきで、無ければ OpenCV で映像だけを書き出す。

  python -m splat_compass.clips                     … 一番新しい試合の倒された場面
  python -m splat_compass.clips --kind kills --last 10 … 新しい10試合の倒した場面をまとめて1本に
"""
import argparse
import datetime as dt
import glob
import os
import re
import shutil
import subprocess

from . import clock
from .storage import Store

KINDS = {"deaths": "倒された場面", "kills": "倒した場面", "swings": "勝率が大きく動いた場面"}
BEFORE = {"deaths": 8.0, "kills": 5.0, "swings": 8.0}
AFTER = {"deaths": 3.0, "kills": 2.0, "swings": 4.0}


def _rec_start(path):
    m = re.search(r"(\d{4})-(\d{2})-(\d{2}) (\d{2})-(\d{2})-(\d{2})", os.path.basename(path))
    try:
        return dt.datetime(*map(int, m.groups())) if m else None
    except ValueError:
        return None


def _duration(path):
    import cv2
    cap = cv2.VideoCapture(path)
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS) or 0
    cap.release()
    return n / fps if fps else 0.0


def recording_times(video_dir):
    """OBS 形式の名前から開始時刻を読める録画を、古い順に返す。"""
    paths = glob.glob(os.path.join(video_dir, "*.mp4")) + glob.glob(os.path.join(video_dir, "*.mkv"))
    return sorted((start, path) for path in paths if (start := _rec_start(path)) is not None)


def locate(m, video_dir, recordings=None, durations=None):
    """(録画のパス, 試合開始が録画の何秒目か)。見つからなければ None。"""
    if not video_dir:
        return None
    src = m.get("source") or ""
    if src and src != "camera":
        p = src if os.path.isabs(src) else os.path.join(video_dir, src)
        return (p, m["video_start"]) if os.path.isfile(p) else None
    # カメラから読んだ試合: 同じ時間に録っていた録画を探す
    try:
        played = dt.datetime.fromisoformat(m["played_at"])
    except (KeyError, ValueError):
        return None
    if recordings is None:
        recordings = recording_times(video_dir)
    candidate = next(((start, path) for start, path in reversed(recordings) if start <= played), None)
    if candidate is None:
        return None
    st, p = candidate
    off = (played - st).total_seconds()
    if durations is None:
        durations = {}
    if p not in durations:
        durations[p] = _duration(p)
    if durations[p] > 0 and off + (m["video_end"] - m["video_start"]) <= durations[p] + 1:
        return p, off
    return None


def scenes(m, kind, store=None):
    """[(試合開始からの秒の始まり, 終わり, 説明)]。重なる場面はつなげる。"""
    me = m.get("me_slot")
    b, a = BEFORE[kind], AFTER[kind]
    raw = []
    if kind == "deaths" and me:
        raw = [(e[1] - b, e[1] + a, "倒された") for e in m["events"] if e[2] == me and "倒された" in e[3]]
    elif kind == "kills":
        raw = [(k["rel"] - b, k["rel"] + a, "倒した") for k in m.get("kills") or []]
    elif kind == "swings" and store is not None:
        from . import learning, winprob
        pr = learning.predict_match(store, m, winprob.load_model(store) if m.get("rule") == "ガチエリア" else None)
        if pr is not None:
            tt, pp = pr
            s0 = m["video_start"]
            moves = []
            for i in range(len(tt)):
                j = i
                while j + 1 < len(tt) and tt[j + 1] - tt[i] <= 5:
                    j += 1
                if j > i and abs(pp[j] - pp[i]) >= 0.08:
                    moves.append((abs(pp[j] - pp[i]), tt[i] - s0, pp[j] - pp[i]))
            moves.sort(reverse=True)
            picked = []
            for _, rel, d in moves:
                if len(picked) >= 6:
                    break
                if all(abs(rel - p) > 10 for p, _ in picked):
                    picked.append((rel, d))
            raw = [(rel - b, rel + 5 + a, f"勝率 {d * 100:+.0f}%") for rel, d in picked]
    raw.sort()
    out = []
    for s, e, label in raw:
        s = max(s, 0)
        if out and s <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], e), out[-1][2] + "・" + label)
        else:
            out.append((s, e, label))
    return out


def write(segments, out_path, log=print):
    """segments: [(録画のパス, 始まりの秒, 終わりの秒)] を順につないで out_path に書く。"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ff = shutil.which("ffmpeg")
    if ff:
        for audio in (True, False):
            cmd = [ff, "-y", "-loglevel", "error"]
            for p, s, e in segments:
                cmd += ["-ss", f"{s:.2f}", "-t", f"{e - s:.2f}", "-i", p]
            n = len(segments)
            ins = "".join(f"[{i}:v][{i}:a]" if audio else f"[{i}:v]" for i in range(n))
            # 見返す用なので 1280×720・30コマ/秒に縮めて軽くする
            fc = f"{ins}concat=n={n}:v=1:a={1 if audio else 0}[c]" + ("[a]" if audio else "") + ";[c]scale=1280:-2,fps=30[v]"
            cmd += ["-filter_complex", fc, "-map", "[v]"] + (["-map", "[a]"] if audio else []) + \
                   ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p"] + \
                   (["-c:a", "aac", "-b:a", "128k"] if audio else []) + [out_path]
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               creationflags=0x08000000 if os.name == "nt" else 0)
            if r.returncode == 0:
                return out_path
        log(f"ffmpeg で書き出せなかったので、映像だけで書き出します: {r.stderr[-200:]}")
    import cv2  # ffmpeg が無いとき: 映像だけ（音なし）
    wr = None
    for p, s, e in segments:
        cap = cv2.VideoCapture(p)
        fps = cap.get(cv2.CAP_PROP_FPS) or 60
        step = max(1, round(fps / 30))  # 30コマ/秒にして軽くする
        cap.set(cv2.CAP_PROP_POS_MSEC, s * 1000)
        n = int((e - s) * fps)
        for i in range(n):
            ok = cap.grab()
            if not ok:
                break
            if i % step:
                continue
            ok, f = cap.retrieve()
            f = cv2.resize(f, (1280, 720), interpolation=cv2.INTER_AREA)
            if wr is None:
                wr = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps / step, (1280, 720))
            wr.write(f)
        cap.release()
    if wr is not None:
        wr.release()
        return out_path
    return None


def clip_dir(store):
    return os.path.join(store.root, "clips")


def make(store, ms, kind, video_dir, log=print):
    """試合の並びから kind の場面を集めて1本にする。書いたファイルのパス（場面が無ければ None）。"""
    segs, names = [], []
    for m in ms:
        loc = locate(m, video_dir)
        if loc is None:
            continue
        path, off = loc
        for s, e, label in scenes(m, kind, store):
            segs.append((path, off + s, off + e))
            names.append(f"{m.get('played_at', '')[5:16].replace('T', ' ')} {clock.fmt(m, s + BEFORE[kind])} {label}")
    if not segs:
        return None
    stem = os.path.splitext(ms[0]["_file"])[0] if len(ms) == 1 else f"最近{len(ms)}試合"
    out = os.path.join(clip_dir(store), f"{stem}_{KINDS[kind]}.mp4")
    log(f"{KINDS[kind]}を {len(segs)}場面 まとめています…")
    p = write(segs, out, log)
    if p:
        with open(os.path.splitext(p)[0] + ".txt", "w", encoding="utf-8") as f:  # どの場面が入っているか
            f.write("\n".join(names))
        log(f"できました: {p}")
    return p


def existing(store, m):
    """その試合について作った動画 [(種類の名前, パス)]"""
    stem = os.path.splitext(m["_file"])[0]
    out = []
    for k, name in KINDS.items():
        p = os.path.join(clip_dir(store), f"{stem}_{name}.mp4")
        if os.path.exists(p):
            out.append((name, p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--kind", default="deaths", choices=list(KINDS))
    ap.add_argument("--last", type=int, default=1)
    ap.add_argument("--video-dir", default=None)
    a = ap.parse_args()
    store = Store(a.data) if a.data else Store()
    vd = a.video_dir or store.load_config().get("video_dir")
    ms = sorted([m for m in store.matches() if m.get("video_end")], key=lambda m: m.get("played_at", ""))[-a.last:]
    p = make(store, ms, a.kind, vd)
    if not p:
        print("切り出せる場面がありませんでした（録画が見つからないか、場面がありません）")


if __name__ == "__main__":
    main()
