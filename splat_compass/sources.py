"""映像の入り口。録画ファイルとカメラ（キャプボ・OBS仮想カメラ）の両方を、
(時刻, RGB画像) の並びとして同じ形で渡す。"""
import threading
import time

import cv2

from .hud import H, W


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
    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    if not cap.isOpened():
        raise RuntimeError(f"カメラ {index} を開けませんでした")
    reader = _Latest(cap)
    reader.start()
    began = time.time()
    interval = 1.0 / fps
    nxt = began
    while reader.ok:
        now = time.time()
        if now < nxt:
            time.sleep(nxt - now)
        nxt += interval
        f = reader.get()
        if f is None:
            continue
        yield time.time() - began, _fit(f)
    cap.release()


def list_cameras(max_index=6):
    """使えそうなカメラ番号と解像度を調べる。"""
    found = []
    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
    for i in range(max_index):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
            ok, f = cap.read()
            if ok:
                found.append((i, f.shape[1], f.shape[0]))
        cap.release()
    return found
