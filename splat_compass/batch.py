"""録画フォルダの中の、まだ処理していない録画を読んで試合の記録を貯める。

  python -m splat_compass.batch "D:\\Spla Capture"            … 未処理の録画を全部処理
  python -m splat_compass.batch "D:\\Spla Capture" --watch    … 新しい録画ができるたびに処理し続ける
  python -m splat_compass.batch "D:\\Spla Capture" --redo     … 新しい版で全部読み直す
"""
import argparse
import datetime as dt
import glob
import os
import re
import time

from .sources import video_frames
from .storage import Store
from .tracker import Tracker
from . import learning

FPS = 10  # 1秒に読むコマ数（溜まってすぐ使うスペシャルを見落とさないよう、多めに読む）


def recorded_at(path):
    """OBS の既定のファイル名（2026-10-26 16-32-34.mp4）から録画開始日時を取る。"""
    m = re.search(r"(\d{4})-(\d{2})-(\d{2}) (\d{2})-(\d{2})-(\d{2})", os.path.basename(path))
    if m:
        return dt.datetime(*map(int, m.groups()))
    return dt.datetime.fromtimestamp(os.path.getmtime(path))


def process(path, store):
    base = recorded_at(path)
    saved = []

    def on_end(m):
        if m.end is None or m.end - m.start < 150 or len(m.events) < 20:
            return  # 短すぎる区間は試合ではない（ロビーや途中で切れた録画）
        p = store.save_match(m, os.path.basename(path), base + dt.timedelta(seconds=m.start))
        saved.append(p)
        print(f"  試合 {len(saved)}: {m.rule or 'ルール不明'} / {m.result or '勝敗不明'} / 出来事{len(m.events)}件")

    tr = Tracker(FPS, on_match_end=on_end)
    for t, img in video_frames(path, FPS):
        tr.feed(t, img)
    tr.flush()
    return saved


def is_stable(path, wait=5):
    """録画中のファイルは大きさが変わり続けるので、止まっているものだけ処理する。"""
    a = os.path.getsize(path)
    time.sleep(wait)
    return a == os.path.getsize(path)


def run(folder, store, watch=False, redo=False):
    if redo:
        n = store.reset_videos()
        print(f"録画から作った記録 {n}件を matches/_old に移しました。全部読み直します。")
    while True:
        done = store.processed()
        todo = [p for p in sorted(glob.glob(os.path.join(folder, "*.mp4"))) if os.path.basename(p) not in done]
        for p in todo:
            if watch and not is_stable(p):
                continue
            print(f"{os.path.basename(p)} を処理中…")
            t0 = time.time()
            process(p, store)
            store.mark_processed(os.path.basename(p))
            print(f"  完了（{time.time() - t0:.0f}秒）")
        if todo:
            learning.retrain(store)
        if not watch:
            break
        time.sleep(30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--watch", action="store_true", help="新しい録画を待ち続ける")
    ap.add_argument("--data", help="記録の保存先（既定: ホームの splat-compass-data）")
    ap.add_argument("--redo", action="store_true", help="処理済みの録画も全部読み直す（前の記録は matches/_old に移す）")
    a = ap.parse_args()
    run(a.folder, Store(a.data), a.watch, a.redo)


if __name__ == "__main__":
    main()
