"""録画から場面を切り出して、1本の動画にまとめる。

- 倒された場面（倒される8秒前から3秒後まで）
- 倒した場面（倒す5秒前から2秒後まで）
- 勝率が大きく動いた場面

録画から読んだ試合はその録画を使う。カメラから読んだ試合も、同じ時間に OBS で録画していれば
（ファイル名が OBS の既定の「2026-11-08 20-05-00.mp4」の形なら）、その録画から切り出せる。
ffmpeg があれば MP4、無ければ OpenCV で音なしの WebM を書き出す。

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
import tempfile

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


def write(segments, out_path, log=print, progress=None):
    """場面を順につなぎ、完成したファイルのパスを返す。"""
    total = sum(e - s for _, s, e in segments)
    if total <= 0:
        return None
    last = None

    def report(seconds, phase):
        nonlocal last
        current = min(99, max(0, int(100 * seconds / total))), phase
        if progress is not None and current != last:
            progress(*current)
            last = current

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with tempfile.TemporaryDirectory(dir=os.path.dirname(out_path)) as tmp:
        staged = _encode(segments, os.path.join(tmp, "clip.mp4"), log, report)
        if staged is None:
            return None
        extension = os.path.splitext(staged)[1]
        output = os.path.splitext(out_path)[0] + extension
        os.replace(staged, output)
        return output


def _encode(segments, out_path, log, report):
    ff = shutil.which("ffmpeg")
    if ff:
        for audio in (True, False):
            phase = "書き出し中" if audio else "音声なしで再試行中"
            report(0, phase)
            cmd = [ff, "-y", "-nostdin", "-nostats", "-loglevel", "error", "-progress", "pipe:1"]
            for p, s, e in segments:
                cmd += ["-ss", f"{s:.2f}", "-t", f"{e - s:.2f}", "-i", p]
            n = len(segments)
            ins = "".join(f"[{i}:v][{i}:a]" if audio else f"[{i}:v]" for i in range(n))
            # 見返す用なので 1280×720・30コマ/秒に縮めて軽くする
            fc = f"{ins}concat=n={n}:v=1:a={1 if audio else 0}[c]" + ("[a]" if audio else "") + ";[c]scale=1280:-2,fps=30[v]"
            cmd += ["-filter_complex", fc, "-map", "[v]"] + (["-map", "[a]"] if audio else []) + \
                   ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p"] + \
                   (["-c:a", "aac", "-b:a", "128k"] if audio else []) + ["-movflags", "+faststart", out_path]
            code, error = _ffmpeg_progress(cmd, lambda seconds: report(seconds, phase))
            if code == 0 and os.path.isfile(out_path) and os.path.getsize(out_path):
                return out_path
        log(f"ffmpeg で書き出せなかったので、映像だけで書き出します: {error[-200:]}")
    phase = "映像のみ書き出し中"
    report(0, phase)
    return _encode_video_only(segments, out_path, log, lambda seconds: report(seconds, phase))


def _ffmpeg_progress(cmd, report):
    # 標準エラーをファイルに出し、読み取っていないパイプで処理が止まるのを防ぐ。
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as errors:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errors, encoding="utf-8", errors="replace",
                              creationflags=0x08000000 if os.name == "nt" else 0) as process:
            for line in process.stdout:
                key, _, value = line.strip().partition("=")
                if key == "out_time_us" and value.lstrip("-").isdigit():
                    report(int(value) / 1_000_000)
            code = process.wait()
        errors.seek(0)
        return code, errors.read()


def _encode_video_only(segments, out_path, log, report):
    import cv2  # ffmpeg が無いとき: 映像だけ（音なし）

    wr = None
    output = os.path.splitext(out_path)[0] + ".webm"
    size = (1280, 720)
    completed = 0.0
    try:
        for p, s, e in segments:
            cap = cv2.VideoCapture(p)
            try:
                fps = cap.get(cv2.CAP_PROP_FPS) or 60
                step = max(1, round(fps / 30))
                cap.set(cv2.CAP_PROP_POS_MSEC, s * 1000)
                for i in range(int((e - s) * fps)):
                    if not cap.grab():
                        break
                    if i % step:
                        continue
                    ok, frame = cap.retrieve()
                    if not ok:
                        continue
                    if wr is None:
                        wr = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"VP80"), fps / step, size)
                        if not wr.isOpened():
                            wr.release()
                            output = out_path
                            wr = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps / step, size)
                            if not wr.isOpened():
                                raise RuntimeError("動画を書き出せませんでした")
                            log("WebM で書き出せないため、外部プレーヤー向けの MP4 で保存します")
                    wr.write(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
                    report(completed + (i + 1) / fps)
            finally:
                cap.release()
            completed += e - s
    finally:
        if wr is not None:
            wr.release()
    return output if wr is not None and os.path.isfile(output) and os.path.getsize(output) else None


def clip_dir(store):
    return os.path.join(store.root, "clips")


def make(store, ms, kind, video_dir, log=print, progress=None):
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
    p = write(segs, out, log, progress)
    if p:
        with open(os.path.splitext(p)[0] + ".txt", "w", encoding="utf-8") as f:  # どの場面が入っているか
            f.write("\n".join(names))
        log(f"できました: {p}")
    return p


def existing(store, m):
    """その試合について作った動画 [(種類の名前, パス)]"""
    stem = os.path.splitext(m["_file"])[0]
    out = []
    for name in KINDS.values():
        base = os.path.join(clip_dir(store), f"{stem}_{name}")
        files = [base + ext for ext in (".mp4", ".webm") if os.path.isfile(base + ext)]
        if files:
            out.append((name, max(files, key=os.path.getmtime)))
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
