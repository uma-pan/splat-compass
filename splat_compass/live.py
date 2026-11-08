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

from .hud import DEAD, SLOT_NAMES, SPECIAL, StageReader
from .sources import camera_frames, list_cameras, video_frames
from .storage import Store
from .tracker import END_TRIM, KIND_LABEL, Tracker
from . import clock, learning, winprob

FPS = 10  # 1秒に読むコマ数（溜まってすぐ使うスペシャルを見落とさないよう、多めに読む）


class Notices:
    """今の状態から、知らせたいことを決める。条件はここを書き換えて調整する。"""

    HOLD_WARN = 20.0  # 自分がスペシャルを溜めたまま何秒たったら知らせるか

    def __init__(self):
        self.active = set()
        self.specials = [None] * 8  # 8人のスペシャルの名前（ブキが読めたら入る）
        self.once = []  # 出来事から出す、1回きりの通知
        self.base = None  # 普段の自分の1分あたりの回数（feedback.baseline）
        self.pace_warned = 0

    def set_weapons(self, names):
        from .weapon_info import Info
        info = Info()
        self.specials = [info.get(n)["スペシャル"] if n else None for n in names]

    PACE_FROM = 90.0  # 試合が始まってこの秒数までは、デスのペースを知らせない

    def reset(self):
        self.specials = [None] * 8
        self.once = []
        self.pace_warned = 0  # 何回目のデスで知らせたか

    def pace(self, tr, t):
        """自分のデスのペースが普段より速ければ知らせる。最初は3回目以降、そのあとは2回増えるごと。"""
        p = my_pace(tr, t, self.base)
        if p is None or t - tr.match.start < self.PACE_FROM:
            return
        deaths, exp = p["deaths"], p["deaths_usual"]
        if exp is None:  # 普段の自分がまだ出せない（同じ種類の試合が少ない）
            return
        if deaths >= 3 and deaths >= exp * 1.5 + 1 and deaths >= self.pace_warned + 2:
            self.pace_warned = deaths
            self.once.append(f"デスが多め {deaths}回（普段なら {exp:.1f}回くらい）")

    def event(self, m, e):
        """出来事から知らせたいもの。味方がスペシャルを溜めたまま倒されたら、その種類を出す。"""
        if e.kind == "dead_with_special" and e.slot[0] == "左":
            sp = self.specials[SLOT_NAMES.index(e.slot)]
            if sp and e.slot != m.me_slot:
                self.once.append(f"味方の{sp}が溜めたまま倒された")

    def check(self, tr, t):
        """新しく出た通知の一覧を返す。"""
        if tr.match is not None:
            self.pace(tr, t)
        c = tr.counts()
        now = set()
        new_once, self.once = self.once, []
        if c:
            ally, enemy = c["alive"]
            if c["specials"][1] >= 2:
                names = [self.specials[k] for k in range(4, 8) if tr.cur and tr.cur[k] == SPECIAL and self.specials[k]]
                tail = f"（{'・'.join(names)}）" if names else ""
                now.add(f"敵のスペシャル準備 {c['specials'][1]}人{tail}")
            if tr.special_since is not None and t - tr.special_since >= self.HOLD_WARN:
                now.add(f"スペシャルを溜めて{int(self.HOLD_WARN)}秒")
        new = now - self.active
        self.active = now
        return sorted(new) + new_once


def my_pace(tr, t, base):
    """この試合の自分のキル・デスと、普段のペースならここまでに何回か。自分の枠が分からなければ None。"""
    m = tr.match
    if m is None or not m.me_slot:
        return None
    deaths = sum(1 for e in m.events if e.slot == m.me_slot and e.kind in ("dead", "dead_with_special"))
    mins = max(t - m.start, 1.0) / 60
    d = {"kills": len(m.kills), "deaths": deaths, "deaths_usual": None, "kills_usual": None}
    if base:
        d["deaths_usual"] = base["deaths"] * mins
        if base.get("kills") is not None:
            d["kills_usual"] = base["kills"] * mins
    return d


def load_base(store):
    """普段の自分の1分あたりの回数。試合が少なければ None。"""
    try:
        from . import feedback
        from .storage import mode_of
        mode = store.mode()  # 今の試合と同じ種類の試合で「普段」を出す
        ms = sorted([m for m in store.matches(all=True) if m.get("video_end") and not m.get("interrupted") and mode_of(m) == mode],
                    key=lambda m: m.get("played_at", ""))
        return feedback.baseline(ms)
    except Exception:
        return None


def win_prob(tr, t, models):
    """試合中の今の勝率（0〜1）と、どのモデルで出したか。出せなければ (None, None)。
    ガチエリアは勝率モデル、ほかのルールはエリアから作った簡単なモデルで目安を出す。"""
    m = tr.match
    if m is None or not m.timeline or tr.cur is None:
        return None, None
    last = m.timeline[-1]
    if t - last[0] > 2.5:
        return None, None
    evs = [e.row() for e in m.events]
    ctx = winprob.context(m.timeline, evs, t)
    pen = (last[5], last[6]) if len(last) >= 7 else (0, 0)
    if m.rule == "ガチエリア" and models.get("area") is not None:
        f = winprob.features(t - m.start, last[1], last[2], last[3], last[4], tr.cur, ctx, pen)
        return float(models["area"].predict(np.array([f]))[0]), "エリア"
    # ルールが読めなかった試合も、カウントが読めていれば目安を出す
    if (m.rule in winprob.SIMPLE_RULES or m.rule is None) and models.get("simple") is not None:
        f = winprob.features(t - m.start, last[1], last[2], 0, 0, tr.cur, ctx, pen)
        x = np.array([[f[c] for c in winprob.SIMPLE_COLS]])
        return float(models["simple"].predict(x)[0]), "目安"
    return None, None


def status_text(tr, t):
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
    return (f"{rule}  {int(t - tr.match.start)}秒\n"
            f"人数 {ally} 対 {enemy}{tag}\n"
            f"スペシャル準備 味方{c['specials'][0]} 敵{c['specials'][1]}\n"
            f"自分 {me}{hold}")


def status_dict(tr, t, base=None):
    """画面（アプリ）向けの今の状態。"""
    if not tr.in_match:
        return {"in_match": False, "pending": tr.finished is not None}
    m = tr.match
    d = {"in_match": True, "rule": m.rule, "elapsed": round(t - m.start, 1), "clock": clock.fmt(m, t - m.start),
         "stage": m.stage,
         "slots": list(tr.cur) if tr.cur else None, "me": tr.me_index,
         "hold": round(t - tr.special_since, 1) if tr.special_since is not None else None,
         "counts": m.timeline[-1][1:3] if m.timeline else None, "kills": len(m.kills),
         "ink": [m.inks[-1][1], round(t - m.inks[-1][0])] if m.inks else None,
         "pace": my_pace(tr, t, base)}
    return d


def icon_data_url(icon):
    import base64
    import cv2
    ok, buf = cv2.imencode(".png", icon[..., ::-1])
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


class WeaponReader:
    """試合中に8人のブキの名前を決める。名前の見本は、名前が増えたら読み直す。"""

    def __init__(self, store):
        self.store, self.namer, self.mtime = store, None, None

    def read(self, icons):
        from . import weapons
        mt = weapons.labels_mtime(self.store)
        if self.namer is None or mt != self.mtime:
            self.namer, self.mtime = weapons.Namer(self.store), mt
        out = []
        for ic in icons:
            if ic is None:
                out.append({"name": None, "conf": "none", "img": ""})
            else:
                n, conf = self.namer.predict(ic)
                out.append({"name": n, "conf": conf, "img": icon_data_url(ic)})
        return out


def beep():
    try:
        import winsound
        winsound.Beep(880, 150)
    except Exception:
        pass


MIN_MATCH = 60.0   # これより短いものは試合として保存しない（ノックアウトは1分台で終わることもある）
MIN_EVENTS = 5


def worker(frames, store, source, out_q, stop, cmds=None):
    """cmds: 画面からの指示（('end', 勝敗) で試合を手で終える）が入るキュー。"""
    """映像を読み続け、表示したいものを out_q に流す。"""
    notices = Notices()
    notices.base = load_base(store)
    model = {"m": winprob.load_model(store)}
    models = {"area": model["m"], "simple": winprob.load_simple(store)}

    busy = {"bg": False, "again": False}

    def retrain_bg():
        """勝率モデルの学習のやり直しとまとめのページの作り直しを、別のプロセスで動かす。
        同じプロセスで動かすと、試合中の読み取りと計算を取り合って、次の試合の読み取りが遅れるため。"""
        if busy["bg"]:
            busy["again"] = True  # 今のが終わったら、もう一度やる
            return
        busy["bg"] = True

        def run():
            import subprocess
            import sys
            try:
                flags = 0x00004000 if os.name == "nt" else 0  # Windows では優先度を下げる（BELOW_NORMAL）
                if getattr(sys, "frozen", False):  # 配る版（exe）は、自分自身を「試合後の作業」として起動する
                    cmd = [sys.executable, "after-match", "--data", store.root]
                    flags |= 0x08000000 if os.name == "nt" else 0  # 黒い窓を出さない
                else:
                    cmd = [sys.executable, "-m", "splat_compass.after_match", "--data", store.root]
                p = subprocess.Popen(cmd,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     encoding="utf-8", errors="replace", creationflags=flags,
                                     env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                                     cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                for line in p.stdout:
                    line = line.rstrip()
                    if line == "@report":
                        out_q.put(("report", None))
                    elif line:
                        out_q.put(("log", line))
                p.wait()
                model["m"] = winprob.load_model(store)
                models.update(area=model["m"], simple=winprob.load_simple(store))
            except Exception as e:
                out_q.put(("log", f"試合後の学習のやり直しができませんでした: {e}"))
            finally:
                busy["bg"] = False
                if busy["again"]:
                    busy["again"] = False
                    retrain_bg()
        threading.Thread(target=run, daemon=True).start()

    def on_event(m, e):
        notices.event(m, e)
        if e.slot == m.me_slot or e.kind in ("dead", "dead_with_special"):
            out_q.put(("log", f"{clock.fmt(m, e.rel)} {e.slot} {KIND_LABEL[e.kind]}"))

    def on_end(m):
        if m.end is not None:  # 終わりの演出中に読んだ分は捨てたので、グラフからも消す
            out_q.put(("wp_trim", m.end - m.start - getattr(m, "trim", END_TRIM)))
        if not getattr(m, "by_hand", False) and (m.end is None or m.end - m.start < MIN_MATCH or len(m.events) < MIN_EVENTS):
            out_q.put(("log", f"短すぎる（{int((m.end or m.start) - m.start)}秒）ので試合として保存しませんでした"))
            return
        when = dt.datetime.now() - dt.timedelta(seconds=(time.time() - began) - m.start) if source == "camera" else None
        p = store.save_match(m, source, when)
        out_q.put(("weapons_saved", os.path.splitext(os.path.basename(p))[0]))
        res = m.result or "勝敗不明（中断扱い）"
        out_q.put(("log", f"試合終了: {m.rule or 'ルール不明'} / {res} → 保存しました"))
        out_q.put(("saved", p))

        def after():  # 振り返りは少し重いので、読み取りを止めないよう別のスレッドで出す
            if m.rule == "ガチエリア" and model["m"] is not None:
                for rel, what, d in learning.swings(model["m"], store.load(p)):
                    out_q.put(("log", f"  勝率が大きく動いた場面: {clock.fmt(m, rel)} {what}（{d * 100:+.0f}%）"))
            try:
                from . import feedback
                lines = feedback.for_latest(store, 1)[0][1]
                out_q.put(("log", "この試合の振り返り:"))
                for kind, text in lines:
                    out_q.put(("log", f"  [{kind}] {text}"))
            except Exception as e:
                out_q.put(("log", f"振り返りを出せませんでした: {e}"))
            notices.base = load_base(store)
            tr.stage_reader = StageReader(store)  # 手で付けたステージの名前を見本に足す
            retrain_bg()
        threading.Thread(target=after, daemon=True).start()

    began = time.time()
    def on_log(text):
        out_q.put(("log", f"（{dt.datetime.now():%H:%M:%S}）{text}"))
    tr = Tracker(FPS, on_event=on_event, on_match_end=on_end, on_log=on_log, stage_reader=StageReader(store))
    last_status, last_wp, cur_start = 0, 0, None
    wreader, wsent = WeaponReader(store), None
    kills_seen = 0

    def send_weapons(icons):
        try:
            res = wreader.read(icons)
            notices.set_weapons([x["name"] for x in res])
            out_q.put(("weapons", res))
        except Exception as e:
            out_q.put(("log", f"ブキを読めませんでした: {e}"))
    perf = {"t0": time.time(), "n": 0, "busy": 0.0, "max": 0.0, "wait": 0.0}
    t_got = time.time()
    for t, img in frames:
        got = time.time()
        perf["wait"] += got - t_got
        if stop.is_set():
            break
        while cmds is not None and not cmds.empty():
            cmd, arg = cmds.get()
            if cmd == "end":
                if tr.end_by_hand(t, arg):
                    out_q.put(("log", f"試合を手で終えました（{arg or '勝敗不明'}）"))
                else:
                    out_q.put(("log", "終える試合がありません"))
        tr.feed(t, img)
        try:  # 通知は失敗しても、読み取りは止めない
            for n in notices.check(tr, t):
                out_q.put(("notice", n))
        except Exception as e:
            if not getattr(notices, "err_shown", False):
                notices.err_shown = True
                out_q.put(("log", f"通知を出せませんでした: {e}"))
        start = tr.match.start if tr.match is not None else None
        if start != cur_start:
            cur_start = start
            if start is not None:  # 新しい試合が始まったらグラフを消す（終わった試合のグラフは残しておく）
                out_q.put(("wp_reset", None))
                notices.reset()
        if time.time() - last_status > 0.3:
            out_q.put(("status", status_text(tr, t)))
            out_q.put(("state", status_dict(tr, t, notices.base)))
            last_status = time.time()
        if start is not None and len(tr.match.kills) > kills_seen:
            rel = tr.match.kills[-1] - start
            out_q.put(("log", f"{clock.fmt(tr.match, rel)} 倒した（{len(tr.match.kills)}人目）"))
        kills_seen = len(tr.match.kills) if start is not None else 0
        if start is not None and wsent != start:
            icons = tr.match.icons
            if all(i is not None for i in icons) or t - start > 60:  # 8人そろったら（遅くとも1分で）ブキを読む
                wsent = start
                threading.Thread(target=send_weapons, args=(list(icons),), daemon=True).start()
        if start is not None and t - last_wp >= 1.0:
            last_wp = t
            p, kind = win_prob(tr, t, models)
            if p is not None:
                out_q.put(("wp", (t - start, p, kind)))
        # 読み取りの速さを見張る。遅れていたら記録に残す（原因を調べる手がかり）
        t_got = time.time()
        dt_ = t_got - got
        perf["n"] += 1
        perf["busy"] += dt_
        perf["max"] = max(perf["max"], dt_)
        if t_got - perf["t0"] >= 30:
            el = t_got - perf["t0"]
            fps = perf["n"] / el
            if start is not None and fps < FPS * 0.7:
                out_q.put(("log", f"読み取りが遅れています: 毎秒 {fps:.1f} コマ（1コマの処理 平均 {perf['busy'] / max(perf['n'], 1) * 1000:.0f}ms・"
                                  f"最大 {perf['max'] * 1000:.0f}ms、映像を待った時間 {perf['wait'] / el * 100:.0f}%、"
                                  f"動いているスレッド {threading.active_count()}）"))
            perf.update(t0=t_got, n=0, busy=0.0, max=0.0, wait=0.0)
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
        elif kind == "wp" and int(msg[0]) % 15 == 0:
            print(f"{int(msg[0]) // 60}:{int(msg[0]) % 60:02d} 勝率{'' if msg[2] == 'エリア' else 'の目安'} {msg[1] * 100:.0f}%", flush=True)


def run_window(q, stop):
    import tkinter as tk

    root = tk.Tk()
    root.title("splat-compass")
    root.attributes("-topmost", True)
    root.geometry("340x430")
    wp_label = tk.Label(root, text="勝率 —", font=("Meiryo", 22, "bold"), anchor="w")
    wp_label.pack(fill="x", padx=8, pady=(6, 0))
    GW, GH = 320, 90
    graph = tk.Canvas(root, width=GW, height=GH, bg="#fafafa", highlightthickness=1, highlightbackground="#ccc")
    graph.pack(padx=8)
    hist = []
    status = tk.Label(root, text="起動中", font=("Meiryo", 12), justify="left", anchor="w")
    status.pack(fill="x", padx=8, pady=6)

    def draw_graph():
        """この試合の勝率の動き。横は試合時間（5分を基本に、延長に入ったら伸ばす）、縦は0〜100%。"""
        graph.delete("all")
        graph.create_line(0, GH / 2, GW, GH / 2, fill="#bbb", dash=(3, 3))
        for k in (1, 2, 3, 4):  # 1分ごとの目盛り
            graph.create_line(GW * k / 5, 0, GW * k / 5, GH, fill="#eee")
        if len(hist) < 2:
            return
        span = max(300.0, hist[-1][0] + 10)
        pts = []
        for rel, p in hist:
            pts += [rel / span * GW, (1 - p) * GH]
        graph.create_line(*pts, fill="#3d7bd9", width=2)

    def show_wp(rel, p, kind):
        hist.append((rel, p))
        color = "#27ae60" if p >= 0.6 else ("#c0392b" if p <= 0.4 else "#333")
        wp_label.config(text=(f"勝率 {p * 100:.0f}%" if kind == "エリア" else f"勝率の目安 {p * 100:.0f}%"), fg=color)
        draw_graph()
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
                elif kind == "wp":
                    show_wp(*msg)
                elif kind == "wp_reset":
                    hist.clear()
                    wp_label.config(text="勝率 —", fg="#333")
                    draw_graph()
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
    draw_graph()
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
