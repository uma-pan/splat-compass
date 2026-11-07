"""キャプボ（または OBS 仮想カメラ）の映像をその場で読み、状況と通知を小さな窓に出す。
試合が終わるたびに記録を保存する。

  python -m splat_compass.live --list               … 使えるカメラ番号を調べる
  python -m splat_compass.live --camera 1           … カメラ1を読む
  python -m splat_compass.live --video 録画.mp4     … 録画を実際の速さで流して動作確認
  （--headless を付けると窓を出さずに文字だけで表示）
"""
import argparse
import datetime as dt
import os
import queue
import threading
import time

import numpy as np

from .hud import DEAD, SPECIAL
from .sources import camera_frames, list_cameras, video_frames
from .storage import Store
from .tracker import KIND_LABEL, Tracker
from . import learning, winprob

FPS = 5


class Notices:
    """今の状態から、知らせたいことを決める。条件はここを書き換えて調整する。"""

    HOLD_WARN = 20.0  # 自分がスペシャルを溜めたまま何秒たったら知らせるか

    def __init__(self):
        self.active = set()

    def check(self, tr, t):
        """新しく出た通知の一覧を返す。"""
        c = tr.counts()
        now = set()
        if c:
            ally, enemy = c["alive"]
            if c["specials"][1] >= 2:
                now.add(f"敵のスペシャル準備 {c['specials'][1]}人")
            if tr.special_since is not None and t - tr.special_since >= self.HOLD_WARN:
                now.add(f"スペシャルを溜めて{int(self.HOLD_WARN)}秒")
        new = now - self.active
        self.active = now
        return sorted(new)


def win_prob(tr, t, model):
    """ガチエリアの試合中なら、今の勝率（0〜1）。出せなければ None。"""
    m = tr.match
    if model is None or m is None or m.rule != "ガチエリア" or not m.timeline or tr.cur is None:
        return None
    last = m.timeline[-1]
    if t - last[0] > 2.5:
        return None
    evs = [e.row() for e in m.events]
    ctx = winprob.context(m.timeline, evs, t)
    pen = (last[5], last[6]) if len(last) >= 7 else (0, 0)
    f = winprob.features(t - m.start, last[1], last[2], last[3], last[4], tr.cur, ctx, pen)
    return float(model.predict(np.array([f]))[0])


def status_text(tr, t, wp=None):
    c = tr.counts()
    if not tr.in_match:
        return "試合を待っています"
    if not c:
        return "試合中（表示の読み取り待ち）"
    if c["me"] is None:
        me = "（まだ分かりません。一度倒されると分かります）"
    else:
        me = f"{tr.match.me_slot} " + {DEAD: "復活待ち", SPECIAL: "スペシャルあり"}.get(c["me"], "")
    hold = f"（{int(t - tr.special_since)}秒）" if tr.special_since is not None else ""
    rule = tr.match.rule or ""
    ally, enemy = c["alive"]
    tag = "（不利）" if ally < enemy else ("（有利）" if ally > enemy else "")
    wp_txt = f"  勝率 {wp * 100:.0f}%" if wp is not None else ""
    return (f"{rule}  {int(t - tr.match.start)}秒{wp_txt}\n"
            f"人数 {ally} 対 {enemy}{tag}\n"
            f"スペシャル準備 味方{c['specials'][0]} 敵{c['specials'][1]}\n"
            f"自分 {me}{hold}")


def beep():
    try:
        import winsound
        winsound.Beep(880, 150)
    except Exception:
        pass


def worker(frames, store, source, out_q, stop):
    """映像を読み続け、表示したいものを out_q に流す。"""
    notices = Notices()
    model = {"m": winprob.load_model(store)}

    def retrain_bg():
        def run():
            learning.retrain(store, log=lambda s: out_q.put(("log", s)))
            model["m"] = winprob.load_model(store)
            try:  # まとめのページも作り直しておく
                from . import report
                report.build(store, store.load_config().get("video_dir"))
            except Exception as e:
                out_q.put(("log", f"まとめのページを作れませんでした: {e}"))
        threading.Thread(target=run, daemon=True).start()

    def on_event(m, e):
        if e.slot == m.me_slot or e.kind in ("dead", "dead_with_special"):
            out_q.put(("log", f"{int(e.rel)//60}:{int(e.rel)%60:02d} {e.slot} {KIND_LABEL[e.kind]}"))

    def on_end(m):
        if m.end is None or m.end - m.start < 150 or len(m.events) < 20:
            return
        when = dt.datetime.now() - dt.timedelta(seconds=(time.time() - began) - m.start) if source == "camera" else None
        p = store.save_match(m, source, when)
        res = m.result or "勝敗不明（中断扱い）"
        out_q.put(("log", f"試合終了: {m.rule or 'ルール不明'} / {res} → 保存しました"))
        out_q.put(("saved", p))
        if m.rule == "ガチエリア" and model["m"] is not None:
            for rel, what, d in learning.swings(model["m"], store.load(p)):
                out_q.put(("log", f"  勝率が大きく動いた場面: {int(rel) // 60}:{int(rel) % 60:02d} {what}（{d * 100:+.0f}%）"))
        try:
            from . import feedback
            out_q.put(("log", "この試合の振り返り:"))
            for kind, text in feedback.for_latest(store, 1)[0][1]:
                out_q.put(("log", f"  [{kind}] {text}"))
        except Exception as e:
            out_q.put(("log", f"振り返りを出せませんでした: {e}"))
        retrain_bg()

    began = time.time()
    tr = Tracker(FPS, on_event=on_event, on_match_end=on_end)
    last_status = 0
    for t, img in frames:
        if stop.is_set():
            break
        tr.feed(t, img)
        for n in notices.check(tr, t):
            out_q.put(("notice", n))
        if time.time() - last_status > 0.3:
            out_q.put(("status", status_text(tr, t, win_prob(tr, t, model["m"]))))
            last_status = time.time()
    tr.flush()
    out_q.put(("log", "映像が終わりました"))


def run_headless(q, th):
    while th.is_alive() or not q.empty():
        try:
            kind, msg = q.get(timeout=0.5)
        except queue.Empty:
            continue
        if kind in ("log", "notice"):
            print(("【通知】" if kind == "notice" else "") + msg, flush=True)


def run_window(q, stop):
    import tkinter as tk

    root = tk.Tk()
    root.title("splat-compass")
    root.attributes("-topmost", True)
    root.geometry("320x300")
    status = tk.Label(root, text="起動中", font=("Meiryo", 13), justify="left", anchor="w")
    status.pack(fill="x", padx=8, pady=6)
    notice = tk.Label(root, text="", font=("Meiryo", 14, "bold"), fg="#c0392b", justify="left", anchor="w")
    notice.pack(fill="x", padx=8)
    log = tk.Text(root, height=8, font=("Meiryo", 9))
    log.pack(fill="both", expand=True, padx=8, pady=6)
    clear_at = [0]

    def poll():
        try:
            while True:
                kind, msg = q.get_nowait()
                if kind == "status":
                    status.config(text=msg)
                elif kind == "notice":
                    notice.config(text=msg)
                    clear_at[0] = time.time() + 4
                    beep()
                elif kind == "log":
                    log.insert("end", msg + "\n")
                    log.see("end")
        except queue.Empty:
            pass
        if clear_at[0] and time.time() > clear_at[0]:
            notice.config(text="")
            clear_at[0] = 0
        root.after(100, poll)

    def close():
        stop.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    poll()
    root.mainloop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, help="カメラ番号（--list で調べる）")
    ap.add_argument("--video", help="録画を実際の速さで流して試す")
    ap.add_argument("--fast", action="store_true", help="--video を待たずに最速で流す")
    ap.add_argument("--list", action="store_true", help="使えるカメラを調べる")
    ap.add_argument("--headless", action="store_true", help="窓を出さずに文字で表示")
    ap.add_argument("--data", help="記録の保存先")
    a = ap.parse_args()
    if a.list:
        snap = os.path.abspath("camera_check")
        cams = list_cameras(snapshot_dir=snap)
        if not cams:
            print("カメラが見つかりませんでした")
        for i, w, h in cams:
            print(f"カメラ {i}: {w}x{h}" + ("" if (w, h) == (1920, 1080) else "  ← 1920x1080 ではありません"))
        if cams:
            print(f"各カメラの映像を {snap} に画像で保存しました。スプラの画面が写っている番号を使ってください。")
        return
    if a.video:
        frames, source = video_frames(a.video, FPS, realtime=not a.fast), os.path.basename(a.video)
    elif a.camera is not None:
        frames, source = camera_frames(a.camera, FPS), "camera"
    else:
        ap.error("--camera か --video を指定してください")
    store = Store(a.data)
    q, stop = queue.Queue(), threading.Event()
    th = threading.Thread(target=worker, args=(frames, store, source, q, stop), daemon=True)
    th.start()
    if a.headless:
        run_headless(q, th)
    else:
        run_window(q, stop)


if __name__ == "__main__":
    main()
