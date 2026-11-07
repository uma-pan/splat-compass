"""8人のブキを、画面上部のアイコンの絵から見分ける。

しくみ:
  1. 試合ごとに8人のアイコンを切り出して weapons/<試合>.png に残す
  2. 似た絵をまとめて、まとまりごとに一度だけ名前を付けてもらう（weapon_labels.json）
  3. あとは名前の付いた絵と見比べて、一番似ているものの名前を使う

  python -m splat_compass.weapons --extract "D:\\Spla Capture"   … 前の録画からアイコンを切り出す
  python -m splat_compass.weapons --label                        … 名前を付ける画面を開く
  python -m splat_compass.weapons                                … ブキ別の勝率を表示
"""
import argparse
import base64
import json
import os
from collections import Counter, defaultdict

import cv2
import numpy as np

from . import killcam
from .hud import ALIVE, SLOT_NAMES, crop_icon, read_slots
from .storage import Store

W, H = 80, 84             # アイコン1枚の大きさ
SAME = 0.75               # まとめるときの似ている度合い
AUTO = 0.80               # 名前の付いた絵とこれ以上似ていれば、同じブキとみなす
HINT = 0.65               # 名前の候補として出す下限
UNKNOWN = "?"             # 「わからない」と答えた絵


# ---------- 絵の保存と読み込み ----------
def _dir(store):
    d = os.path.join(store.root, "weapons")
    os.makedirs(d, exist_ok=True)
    return d


def save_strip(store, stem, icons):
    """8枚を横に並べた1枚の画像にして保存し、ファイル名を返す。icons は RGB。"""
    tiles = [i if i is not None else np.zeros((H, W, 3), np.uint8) for i in icons]
    ok, buf = cv2.imencode(".png", np.concatenate(tiles, axis=1)[..., ::-1])
    if not ok:
        return None
    fname = f"{stem}.png"
    buf.tofile(os.path.join(_dir(store), fname))
    return fname


def load_strip(store, m):
    """試合の8枚（RGB、無いところは None）。記録が無ければ None。"""
    fname = m.get("weapon_icons")
    if not fname:
        return None
    p = os.path.join(_dir(store), fname)
    if not os.path.exists(p):
        return None
    im = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
    if im is None or im.shape[1] < 8 * W:
        return None
    im = im[..., ::-1]
    out = []
    for k in range(8):
        t = im[:, k * W:(k + 1) * W]
        out.append(t.copy() if t.max() > 0 else None)
    return out


def feature(icon):
    """絵の輪郭（明るさ・色の変わり目）を少しぼかしたもの。背景のイカ・タコの色に左右されにくい。"""
    r = icon[18:72, 10:70].astype(np.float32)
    gx = cv2.Sobel(r, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(r, cv2.CV_32F, 0, 1)
    m = np.minimum(np.sqrt(gx ** 2 + gy ** 2).max(2), 300)
    m = cv2.GaussianBlur(m, (7, 7), 0)
    v = cv2.resize(m, (30, 27)).flatten()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


# ---------- 名前 ----------
def load_labels(store):
    try:
        with open(os.path.join(store.root, "weapon_labels.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_labels(store, labels):
    p = os.path.join(store.root, "weapon_labels.json")
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=1, sort_keys=True)
    os.replace(p + ".tmp", p)


def key_of(m, k):
    return f"{os.path.splitext(m['_file'])[0]}:{k}"


def all_icons(store, ms=None):
    """[(キー, 絵, 試合, 枠番号)]"""
    out = []
    for m in ms if ms is not None else store.matches():
        icons = load_strip(store, m)
        if not icons:
            continue
        for k, ic in enumerate(icons):
            if ic is not None:
                out.append((key_of(m, k), ic, m, k))
    return out


class Namer:
    """名前の付いた絵と見比べて名前を決める。"""

    def __init__(self, store, items=None, labels=None):
        self.labels = labels if labels is not None else load_labels(store)
        items = items if items is not None else all_icons(store)
        named = [(feature(ic), self.labels[key]) for key, ic, _, _ in items
                 if self.labels.get(key) not in (None, "", UNKNOWN)]
        self.F = np.array([f for f, _ in named]) if named else np.zeros((0, 810), np.float32)
        self.names = [n for _, n in named]

    def nearest(self, f):
        if not self.names:
            return None, 0.0
        s = self.F @ f
        j = int(s.argmax())
        return self.names[j], float(s[j])

    def name(self, key, icon):
        n = self.labels.get(key)
        if n:
            return None if n == UNKNOWN else n
        n, s = self.nearest(feature(icon))
        return n if s >= AUTO else None


def weapons_of(store, m, namer=None):
    """試合の8人のブキの名前（分からない枠は None）。"""
    icons = load_strip(store, m)
    if not icons:
        return None
    namer = namer or Namer(store)
    return [namer.name(key_of(m, k), ic) if ic is not None else None for k, ic in enumerate(icons)]


# ---------- 前の録画から切り出す ----------
def extract_past(store, video_dir, redo=False):
    n = 0
    for m in store.matches():
        if m.get("weapon_icons") and not redo:
            continue
        src = m.get("source") or ""
        path = src if os.path.isabs(src) else os.path.join(video_dir, src)
        if not os.path.exists(path) or m.get("video_start") is None:
            continue
        cap = cv2.VideoCapture(path)
        got = [None] * 8
        for dt in (15, 22, 30, 40, 55):
            cap.set(cv2.CAP_PROP_POS_MSEC, (m["video_start"] + dt) * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            img = frame[..., ::-1]
            if img.shape[:2] != (1080, 1920):
                img = cv2.resize(img, (1920, 1080))
            st = read_slots(img)
            for k in range(8):
                if got[k] is None and st[k] == ALIVE:
                    got[k] = crop_icon(img, k)
            if all(g is not None for g in got):
                break
        cap.release()
        if not any(g is not None for g in got):
            continue
        stem = os.path.splitext(m["_file"])[0]
        fname = save_strip(store, stem, got)
        p = os.path.join(store.mdir, m["_file"])
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        d["weapon_icons"] = fname
        with open(p, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
        n += 1
        print(f"  {m['_file']}: {sum(g is not None for g in got)}人分")
    return n


# ---------- まとめる ----------
def group(F, th=SAME):
    """似た絵をまとめる。[[番号...]]（大きい順）"""
    cents, mem = [], []
    for i, f in enumerate(F):
        if cents:
            s = np.array(cents) @ f
            j = int(s.argmax())
            if s[j] > th:
                mem[j].append(i)
                c = F[mem[j]].mean(0)
                cents[j] = c / np.linalg.norm(c)
                continue
        cents.append(f)
        mem.append([i])
    while len(cents) > 1:
        C = np.array(cents)
        S = C @ C.T
        np.fill_diagonal(S, -1)
        a, b = np.unravel_index(S.argmax(), S.shape)
        if S[a, b] <= th:
            break
        mem[a] += mem[b]
        c = F[mem[a]].mean(0)
        cents[a] = c / np.linalg.norm(c)
        del mem[b], cents[b]
    return sorted(mem, key=len, reverse=True)


def killcam_names(store, m, labels=None, protos=None):
    """その試合で自分を倒したメインのブキの名前。"""
    out = set()
    for kc in m.get("killed_by") or []:
        n = killcam.name_of(store, os.path.join(store.root, "killcam", kc["image"]), labels, protos)
        if n and killcam.kind_of(n) == "メイン":
            out.add(n)
    return out


def _side(m, k):
    """自分から見た k番目の枠: 'me' / 'ally' / 'enemy' / None（自分の枠が分からない）"""
    me = m.get("me_slot")
    if me not in SLOT_NAMES:
        return None
    i = SLOT_NAMES.index(me)
    if k == i:
        return "me"
    return "ally" if (k < 4) == (i < 4) else "enemy"


def suggest(store, members, items, namer, kc_cache):
    """まとまりの名前の候補。似た名前付きの絵 → 倒された画面との重なり の順に探す。"""
    votes = Counter()
    for i in members:
        n, s = namer.nearest(feature(items[i][1]))
        if n and s >= HINT:
            votes[n] += 1
    if votes:
        return votes.most_common(1)[0][0], "似た絵"
    # 相手側にいた試合で、自分を倒したブキの名前
    co, seen = Counter(), 0
    for i in members:
        _, _, m, k = items[i]
        if _side(m, k) != "enemy":
            continue
        seen += 1
        co.update(kc_cache(m))
    if co:
        n, c = co.most_common(1)[0]
        if c >= 2 and c / max(seen, 1) >= 0.3:
            return n, f"相手にいた{seen}試合のうち{c}試合でこのブキに倒されている"
    return "", ""


def queue(store):
    """名前を付けてほしいまとまり。[(番号のリスト, 候補, 理由)] と items"""
    items = all_icons(store)
    labels = load_labels(store)
    namer = Namer(store, items, labels)
    todo = [i for i, (key, ic, _, _) in enumerate(items)
            if not labels.get(key) and namer.nearest(feature(ic))[1] < AUTO]
    if not todo:
        return [], items
    F = np.array([feature(items[i][1]) for i in todo])
    kl, kp = killcam.load_labels(store)
    cache = {}

    def kc(m):
        if m["_file"] not in cache:
            cache[m["_file"]] = killcam_names(store, m, kl, kp)
        return cache[m["_file"]]

    out = []
    for g in group(F):
        c = F[g].mean(0)
        g = sorted(g, key=lambda j: -float(F[j] @ c))  # 似ていない絵ほど後ろに並ぶ（外しやすいように）
        members = [todo[j] for j in g]
        name, why = suggest(store, members, items, namer, kc)
        out.append((members, name, why))
    return out, items


# ---------- 名前を付ける画面 ----------
def label_gui(store):
    import tkinter as tk
    from tkinter import ttk

    groups, items = queue(store)
    if not groups:
        print("名前の分からないアイコンはありません")
        return
    labels = load_labels(store)
    known = set(n for n in labels.values() if n and n != UNKNOWN)
    kl, _ = killcam.load_labels(store)
    known |= {n for n in kl.values() if n and killcam.kind_of(n) == "メイン"}

    def photo(icon, dim=False):
        im = icon.astype(np.float32)
        if dim:
            im = im * 0.25 + 40
        ok, buf = cv2.imencode(".png", im.astype(np.uint8)[..., ::-1])
        return tk.PhotoImage(data=base64.b64encode(buf.tobytes()).decode())

    root = tk.Tk()
    root.title("ブキの名前を付ける")
    root.geometry("1060x720")
    top = tk.Label(root, font=("Meiryo", 12), anchor="w", justify="left")
    top.pack(fill="x", padx=10, pady=(10, 0))
    hint = tk.Label(root, font=("Meiryo", 10), fg="#666", anchor="w", justify="left")
    hint.pack(fill="x", padx=10)
    grid = tk.Frame(root)
    grid.pack(fill="both", expand=True, padx=10, pady=6)
    bottom = tk.Frame(root)
    bottom.pack(fill="x", padx=10, pady=10)
    tk.Label(bottom, text="ブキの名前:", font=("Meiryo", 12)).pack(side="left")
    name_var = tk.StringVar()
    box = ttk.Combobox(bottom, textvariable=name_var, values=sorted(known), font=("Meiryo", 12), width=28)
    box.pack(side="left", padx=6)

    state = {"i": 0, "out": set(), "imgs": [], "cells": []}
    MAXSHOW = 60

    def show():
        for w in grid.winfo_children():
            w.destroy()
        state["out"], state["imgs"], state["cells"] = set(), [], []
        if state["i"] >= len(groups):
            top.config(text="おわりです。ありがとうございました。")
            hint.config(text="")
            root.after(1200, root.destroy)
            return
        members, name, why = groups[state["i"]]
        sides = Counter(_side(items[j][2], items[j][3]) for j in members)
        top.config(text=f"このブキの名前を入れてください（{state['i'] + 1} / {len(groups)} まとまり、{len(members)}枚）"
                        + ("　※自分が使った試合あり" if sides.get("me") else ""))
        hint.config(text=("違うブキが混ざっていたら、その絵を押して外してください。"
                          + (f"\n候補「{name}」: {why}" if name else "")
                          + (f"\n表示しきれない {len(members) - MAXSHOW}枚も同じ名前になります" if len(members) > MAXSHOW else "")))
        for n, j in enumerate(members[:MAXSHOW]):
            a, b = photo(items[j][1]), photo(items[j][1], True)
            state["imgs"] += [a, b]
            lb = tk.Label(grid, image=a, bd=2, relief="flat", cursor="hand2")
            lb.grid(row=n // 12, column=n % 12, padx=2, pady=2)

            def toggle(_e, j=j, lb=lb, a=a, b=b):
                if j in state["out"]:
                    state["out"].discard(j)
                    lb.config(image=a)
                else:
                    state["out"].add(j)
                    lb.config(image=b)
            lb.bind("<Button-1>", toggle)
        name_var.set(name)
        box.focus_set()
        box.icursor("end")

    def finish_group(value):
        members = groups[state["i"]][0]
        out = [j for j in members if j in state["out"]]
        if value is not None:
            for j in members:
                if j not in state["out"]:
                    labels[items[j][0]] = value
            save_labels(store, labels)
            if value != UNKNOWN:
                known.add(value)
                box.config(values=sorted(known))
        if out:  # 外した絵は、あとでもう一度出す
            groups.append((out, "", ""))
        state["i"] += 1
        show()

    def ok(_e=None):
        v = name_var.get().strip()
        if v:
            finish_group(v)

    tk.Button(bottom, text="決定 (Enter)", font=("Meiryo", 11), command=ok).pack(side="left", padx=4)
    tk.Button(bottom, text="わからない", font=("Meiryo", 11), command=lambda: finish_group(UNKNOWN)).pack(side="left", padx=4)
    tk.Button(bottom, text="あとで", font=("Meiryo", 11), command=lambda: finish_group(None)).pack(side="left", padx=4)
    tk.Button(bottom, text="終わる", font=("Meiryo", 11), command=root.destroy).pack(side="right")
    root.bind("<Return>", ok)
    show()
    root.mainloop()


# ---------- 集計 ----------
def stats(store, ms):
    """{'me'/'ally'/'enemy': {名前: [試合数, 勝ち数]}}, 名前の分かった割合"""
    namer = Namer(store)
    out = {"me": defaultdict(lambda: [0, 0]), "ally": defaultdict(lambda: [0, 0]),
           "enemy": defaultdict(lambda: [0, 0])}
    seen = named = 0
    for m in ms:
        if m.get("result") not in ("勝ち", "負け"):
            continue
        names = weapons_of(store, m, namer)
        if not names:
            continue
        win = m["result"] == "勝ち"
        done = set()  # 同じブキが2人いても1試合として数える
        for k, n in enumerate(names):
            side = _side(m, k)
            if side is None:
                continue
            seen += 1
            if not n:
                continue
            named += 1
            if (side, n) in done:
                continue
            done.add((side, n))
            out[side][n][0] += 1
            out[side][n][1] += win
    return out, (named / seen if seen else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="記録の保存先")
    ap.add_argument("--extract", metavar="録画のフォルダ", nargs="?", const="", help="前の録画からアイコンを切り出す")
    ap.add_argument("--redo", action="store_true", help="切り出し済みの試合もやり直す")
    ap.add_argument("--label", action="store_true", help="名前を付ける画面を開く")
    a = ap.parse_args()
    store = Store(a.data)
    if a.extract is not None:
        vd = a.extract or store.load_config().get("video_dir")
        if not vd:
            print("録画のフォルダを指定してください")
            return
        print(f"{extract_past(store, vd, a.redo)}試合分のアイコンを切り出しました")
        return
    if a.label:
        label_gui(store)
        return
    ms = [m for m in store.matches() if m.get("video_end") and not m.get("interrupted")]
    st, rate = stats(store, ms)
    if rate is None:
        print("アイコンの記録がまだありません。--extract で前の録画から切り出せます")
        return
    print(f"ブキの名前が分かった割合: {rate * 100:.0f}%（残りは --label で名前を付けられます）")
    for side, title in (("me", "自分が使ったブキ"), ("enemy", "相手にいた時の勝率"), ("ally", "味方にいた時の勝率")):
        rows = sorted(((n, c, w) for n, (c, w) in st[side].items() if c >= 3), key=lambda x: x[2] / x[1])
        if not rows:
            continue
        print(f"\n■ {title}（3試合以上）")
        for n, c, w in rows:
            print(f"  {n:<22} {c:3d}試合  勝率 {w / c * 100:3.0f}%")


if __name__ == "__main__":
    main()
