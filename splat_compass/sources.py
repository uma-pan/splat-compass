"""映像の入り口。録画ファイルとカメラ（キャプボ・OBS仮想カメラ）の両方を、
(時刻, RGB画像) の並びとして同じ形で渡す。"""
import threading
import time

import os

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # 使えない番号を試したときの警告を出さない
import cv2

from .hud import H, W


def _open_camera(index):
    """Windows では DirectShow → Media Foundation → 自動の順に試す。"""
    for name in ("CAP_DSHOW", "CAP_MSMF", "CAP_ANY"):
        if not hasattr(cv2, name):
            continue
        cap = cv2.VideoCapture(index, getattr(cv2, name))
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
    return None


def _fit(frame):
    if frame.shape[1] != W or frame.shape[0] != H:
        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
    return frame[..., ::-1]  # BGR → RGB


def video_frames(path, fps=5, start=0.0, realtime=False):
    """録画を fps コマ/秒に間引いて読む。realtime=True なら実際の時間の速さで流す（動作確認用）。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"開けませんでした: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 60
    step = max(1, round(src_fps / fps))
    if start:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    i = int(round(start * src_fps))
    began = time.time()
    while True:
        if i % step:
            if not cap.grab():
                break
            i += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        t = i / src_fps
        i += 1
        if realtime:
            wait = (t - start) - (time.time() - began)
            if wait > 0:
                time.sleep(wait)
        yield t, _fit(frame)
    cap.release()


class _Latest(threading.Thread):
    """カメラから常に最新のコマだけを持っておく（処理が遅れても溜まらないように）。"""

    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap, self.frame, self.ok = cap, None, True
        self.lock = threading.Lock()

    def run(self):
        while self.ok:
            ok, f = self.cap.read()
            if not ok:
                self.ok = False
                break
            with self.lock:
                self.frame = f

    def get(self):
        with self.lock:
            return self.frame


def camera_frames(index=0, fps=5):
    """キャプボや OBS 仮想カメラを番号で開いて読む。"""
    cap = _open_camera(index)
    if cap is None:
        raise RuntimeError(f"カメラ {index} を開けませんでした")
    reader = _Latest(cap)
    reader.start()
    began = time.time()
    interval = 1.0 / fps
    nxt = began
    try:
        while reader.ok:
            now = time.time()
            if now < nxt:
                time.sleep(nxt - now)
            nxt += interval
            f = reader.get()
            if f is None:
                continue
            yield time.time() - began, _fit(f)
    finally:  # 読むのをやめたら（カメラを選び直すときなど）、カメラを手放す
        reader.ok = False
        reader.join(timeout=2)
        cap.release()


def list_cameras(max_index=6, snapshot_dir=None):
    """使えそうなカメラ番号と解像度を調べる。snapshot_dir を渡すと、各カメラの1コマを画像で保存する。"""
    found = []
    for i in range(max_index):
        cap = _open_camera(i)
        if cap is None:
            continue
        ok, f = cap.read()
        if ok:
            found.append((i, f.shape[1], f.shape[0]))
            if snapshot_dir:
                os.makedirs(snapshot_dir, exist_ok=True)
                cv2.imwrite(os.path.join(snapshot_dir, f"camera_{i}.png"), f)
        cap.release()
    return found
