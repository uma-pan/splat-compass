"""8人のブキを、画面上部のアイコンの絵から見分ける。

しくみ:
  1. 試合ごとに8人のアイコンを切り出して weapons/<試合>.png に残す
  2. 似た絵を並べて見せ、同じブキの絵に一度だけ名前を付けてもらう（weapon_labels.json）
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
PICK = 0.75               # 名前を付ける画面で、これ以上似た絵をたどってつながったものを最初から選んでおく
AUTO = 0.75               # 名前の付いた絵とこれ以上似ていれば、同じブキとみなす
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


def _edges(icon):
    """絵の輪郭（明るさ・色の変わり目）を少しぼかしたもの。"""
    r = icon[18:72, 6:74].astype(np.float32)
    gx = cv2.Sobel(r, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(r, cv2.CV_32F, 0, 1)
    m = np.minimum(np.sqrt(gx ** 2 + gy ** 2).max(2), 300)
    m = cv2.GaussianBlur(m, (7, 7), 0)
    v = cv2.resize(m, (30, 27), interpolation=cv2.INTER_AREA).flatten()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


def _colors(icon):
    """ブキの部分の色。背景（イカ・タコの形のチームの色）に近いところは消してから見る。
    輪郭だけでは見分けにくい、形の似たブキ（R-PEN とリッター、ローラーの仲間など）を分けるため。"""
    r = cv2.cvtColor(np.ascontiguousarray(icon[18:72, 6:74]), cv2.COLOR_RGB2LAB).astype(np.float32)
    fill = np.median(r.reshape(-1, 3), 0)
    w = np.clip((np.linalg.norm(r - fill, axis=2) - 25) / 25, 0, 1)[..., None]
    gray = np.array([128, 128, 128], np.float32)
    x = r * w + gray * (1 - w)
    x = cv2.resize(cv2.GaussianBlur(x, (3, 3), 0), (34, 27), interpolation=cv2.INTER_AREA)
    x = (x - gray).flatten()
    return x / (np.linalg.norm(x) + 1e-6)


COLOR_WEIGHT = 2.0


def feature(icon):
    v = np.concatenate([_edges(icon), COLOR_WEIGHT * _colors(icon)])
    return v / np.linalg.norm(v)


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
        self.F = np.array([f for f, _ in named]) if named else None
        self.names = [n for _, n in named]

    def nearest(self, f):
        if not self.names:
            return None, 0.0
        s = self.F @ f
        j = int(s.argmax())
        return self.names[j], float(s[j])

    def predict(self, icon, key=None):
        """(名前, 確かさ)。確かさ: 'label' 手で付けた / 'auto' 似た絵がある / 'guess' 候補どまり / 'none' 分からない"""
        n = self.labels.get(key) if key else None
        if n:
            return (None, "label") if n == UNKNOWN else (n, "label")
        n, s = self.nearest(feature(icon))
        if n and s >= AUTO:
            return n, "auto"
        if n and s >= HINT:
            return n, "guess"
        return None, "none"

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


def set_label(store, key, name):
    """1枚の絵の名前を手で決める（空なら手で付けた名前を消す）。次から名前を決める見本に使われる。"""
    labels = load_labels(store)
    if name:
        labels[key] = name
    else:
        labels.pop(key, None)
    save_labels(store, labels)


def labels_mtime(store):
    try:
        return os.path.getmtime(os.path.join(store.root, "weapon_labels.json"))
    except OSError:
        return 0


def review(store, last=20):
    """直近の試合の8人のブキの読み取り結果。[(試合, [(キー, 名前, 確かさ, 自分から見た側) ×8 / 無い枠は None])]"""
    ms = [m for m in store.matches() if m.get("weapon_icons")]
    ms.sort(key=lambda m: m.get("played_at", ""), reverse=True)
    ms = ms[:last]
    namer = Namer(store)
    out = []
    for m in ms:
        icons = load_strip(store, m)
        if not icons:
            continue
        row = []
        for k, ic in enumerate(icons):
            if ic is None:
                row.append(None)
                continue
            key = key_of(m, k)
            n, conf = namer.predict(ic, key)
            row.append((key, n, conf, _side(m, k)))
        out.append((m, row))
    return out, sorted(set(namer.names))


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


# ---------- 名前を付ける ----------
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
    """選んだ絵の名前の候補。似た名前付きの絵 → 倒された画面との重なり の順に探す。"""
    votes = Counter()
    for i in members:
        n, s = namer.nearest(feature(items[i][1]))
        if n and s >= HINT:
            votes[n] += 1
    if votes:
        return votes.most_common(1)[0][0], "似た絵に付けた名前"
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


def chains(S, th):
    """似ている度合いが th 以上の絵どうしを、つながりでまとめる。[[番号...]]
    （同じブキでも映り方で似方がばらつくので、近い絵をたどってつなげる）"""
    n = len(S)
    par = list(range(n))

    def root(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    a, b = np.where(np.triu(S >= th, 1))
    for i, j in zip(a, b):
        par[root(i)] = root(j)
    out = defaultdict(list)
    for i in range(n):
        out[root(i)].append(i)
    return list(out.values())


class Pool:
    """名前付けに使う、全部の絵と名前。"""

    def __init__(self, store):
        self.store = store
        self.items = all_icons(store)
        self.labels = load_labels(store)
        self.F = np.array([feature(it[1]) for it in self.items]) if self.items else np.zeros((0, 1))
        kl, kp = killcam.load_labels(store)
        self._kc, self._kl, self._kp = {}, kl, kp

    def kc(self, m):
        if m["_file"] not in self._kc:
            self._kc[m["_file"]] = killcam_names(self.store, m, self._kl, self._kp)
        return self._kc[m["_file"]]

    def name(self, i):
        return self.labels.get(self.items[i][0])

    def named(self):
        return [i for i in range(len(self.items)) if self.name(i) not in (None, "", UNKNOWN)]

    def unnamed(self):
        """名前が無く、名前の付いた絵とも十分には似ていない絵。"""
        todo = [i for i in range(len(self.items)) if not self.name(i)]
        nm = self.named()
        if todo and nm:
            best = (self.F[todo] @ self.F[nm].T).max(1)
            todo = [i for i, b in zip(todo, best) if b < AUTO]
        return todo

    def namer(self):
        n = Namer.__new__(Namer)
        nm = self.named()
        n.labels = self.labels
        n.F = self.F[nm] if nm else None
        n.names = [self.name(i) for i in nm]
        return n

    def save(self):
        save_labels(self.store, self.labels)


def label_gui(store):
    import tkinter as tk
    from tkinter import ttk

    print("アイコンを読み込んでいます…")
    pool = Pool(store)
    if not pool.items:
        print("アイコンの記録がありません。--extract で前の録画から切り出せます")
        return
    print(f"{len(pool.items)}枚。名前を付ける画面を開きます（タスクバーも見てください）")
    known = {n for n in pool.labels.values() if n and n != UNKNOWN}
    known |= {n for n in pool._kl.values() if n and killcam.kind_of(n) == "メイン"}
    MAXSHOW, COLS = 60, 12

    def photo(icon, dim=False):
        im = icon.astype(np.float32)
        if dim:
            im = im * 0.25 + 40
        ok, buf = cv2.imencode(".png", im.astype(np.uint8)[..., ::-1])
        return tk.PhotoImage(data=base64.b64encode(buf.tobytes()).decode())

    root = tk.Tk()
    root.title("ブキの名前を付ける")
    root.geometry("1080x760")
    modes = tk.Frame(root)
    modes.pack(fill="x", padx=10, pady=(8, 0))
    top = tk.Label(root, font=("Meiryo", 12), anchor="w", justify="left")
    top.pack(fill="x", padx=10, pady=(6, 0))
    hint = tk.Label(root, font=("Meiryo", 10), fg="#555", anchor="w", justify="left")
    hint.pack(fill="x", padx=10)
    grid = tk.Frame(root)
    grid.pack(fill="both", expand=True, padx=10, pady=6)
    bottom = tk.Frame(root)
    bottom.pack(fill="x", padx=10, pady=10)
    tk.Label(bottom, text="ブキの名前:", font=("Meiryo", 12)).pack(side="left")
    name_var = tk.StringVar()
    box = ttk.Combobox(bottom, textvariable=name_var, values=sorted(known), font=("Meiryo", 12), width=26)
    box.pack(side="left", padx=6)
    buttons = tk.Frame(bottom)
    buttons.pack(side="left", fill="x", expand=True)

    st = {"mode": "new", "shown": [], "sel": set(), "imgs": [], "skip": set(), "fix_name": None}

    def draw(indices, selected):
        """絵を並べる。押すと選ぶ／外すが切り替わる。選んでいない絵は暗く出す。"""
        for w in grid.winfo_children():
            w.destroy()
        st["shown"], st["sel"], st["imgs"] = indices, set(selected), []
        for n, j in enumerate(indices):
            a, b = photo(pool.items[j][1]), photo(pool.items[j][1], True)
            st["imgs"] += [a, b]
            lb = tk.Label(grid, image=a if j in st["sel"] else b, bd=0, cursor="hand2")
            lb.grid(row=n // COLS, column=n % COLS, padx=2, pady=2)

            def toggle(_e, j=j, lb=lb, a=a, b=b):
                if j in st["sel"]:
                    st["sel"].discard(j)
                    lb.config(image=b)
                else:
                    st["sel"].add(j)
                    lb.config(image=a)
                count()
            lb.bind("<Button-1>", toggle)
        count()

    def count():
        if st["mode"] == "new":
            hint.config(text=f"明るい絵（{len(st['sel'])}枚）に名前を付けます。違うブキは押して暗くし、"
                             "同じブキなのに暗い絵は押して明るくしてください。" + st.get("why", ""))
        else:
            hint.config(text=f"押した絵（{len(st['sel'])}枚）は名前を消すか、下の名前に付け直します。"
                             "似ていない絵ほど前に並んでいます。")

    # ---- 名前を付ける ----
    def next_new():
        st["mode"] = "new"
        todo = [i for i in pool.unnamed() if pool.items[i][0] not in st["skip"]]
        if not todo:
            top.config(text="名前の無い絵はもうありません。")
            draw([], [])
            return
        F = pool.F[todo]
        S = F @ F.T
        comp = chains(S, PICK)
        big = max(comp, key=len)  # 似た絵のつながりが一番大きいものから聞く
        seed = max(big, key=lambda k: float(S[k, big].sum()))
        members = set(big)
        order = sorted(big, key=lambda k: -S[seed, k])[:MAXSHOW]
        rest = [k for k in np.argsort(-S[seed]) if k not in members][:MAXSHOW - len(order)]
        idx = [todo[k] for k in order + rest]
        sel = [todo[k] for k in order]
        nm = pool.namer()
        name, why = suggest(store, sel, pool.items, nm, pool.kc)
        st["why"] = f"\n候補「{name}」: {why}" if name else ""
        mine = any(_side(pool.items[j][2], pool.items[j][3]) == "me" for j in sel)
        top.config(text=f"このブキの名前を入れてください（名前の無い絵 残り{len(todo)}枚）"
                        + ("　※自分が使った試合あり" if mine else ""))
        draw(idx, sel)
        name_var.set(name)
        box.focus_set()
        box.icursor("end")

    def put(value):
        for j in st["sel"]:
            pool.labels[pool.items[j][0]] = value
        pool.save()
        if value != UNKNOWN:
            known.add(value)
            box.config(values=sorted(known))

    def ok(_e=None):
        v = name_var.get().strip()
        if st["mode"] == "new":
            if v and st["sel"]:
                put(v)
            next_new()
        elif v and st["sel"]:
            put(v)
            show_fix(st["fix_name"])

    def unknown():
        if st["mode"] == "new" and st["sel"]:
            put(UNKNOWN)
            next_new()

    def later():
        if st["mode"] == "new":
            st["skip"].update(pool.items[j][0] for j in (st["sel"] or set(st["shown"][:1])))
            next_new()

    # ---- 付けた名前を見直す ----
    fix_var = tk.StringVar()
    fix_box = ttk.Combobox(modes, textvariable=fix_var, state="readonly", font=("Meiryo", 11), width=30)

    def names_with_count():
        c = Counter(n for n in pool.labels.values() if n)
        return [f"{n}（{k}枚）" for n, k in sorted(c.items(), key=lambda x: -x[1])]

    def start_fix():
        st["mode"] = "fix"
        fix_box.config(values=names_with_count())
        if fix_box["values"]:
            fix_box.current(0)
            show_fix(fix_box.get().rsplit("（", 1)[0])

    def show_fix(name):
        st["mode"], st["fix_name"] = "fix", name
        idx = [i for i in range(len(pool.items)) if pool.name(i) == name]
        if not idx:
            top.config(text=f"「{name}」の絵はもうありません。")
            draw([], [])
            return
        c = pool.F[idx].mean(0)
        idx.sort(key=lambda i: float(pool.F[i] @ c))  # 似ていない絵ほど前
        top.config(text=f"「{name}」と名前を付けた絵（{len(idx)}枚）。違うブキの絵を押して選んでください。")
        draw(idx[:MAXSHOW], [])
        name_var.set("")

    def erase():
        if st["mode"] == "fix" and st["sel"]:
            for j in st["sel"]:
                pool.labels.pop(pool.items[j][0], None)
            pool.save()
            show_fix(st["fix_name"])

    fix_box.bind("<<ComboboxSelected>>", lambda _e: show_fix(fix_var.get().rsplit("（", 1)[0]))
    tk.Button(modes, text="名前を付ける", font=("Meiryo", 10), command=next_new).pack(side="left")
    tk.Button(modes, text="付けた名前を見直す", font=("Meiryo", 10), command=start_fix).pack(side="left", padx=6)
    fix_box.pack(side="left")

    tk.Button(buttons, text="決定 (Enter)", font=("Meiryo", 11), command=ok).pack(side="left", padx=4)
    tk.Button(buttons, text="わからない", font=("Meiryo", 11), command=unknown).pack(side="left", padx=4)
    tk.Button(buttons, text="あとで", font=("Meiryo", 11), command=later).pack(side="left", padx=4)
    tk.Button(buttons, text="選んだ絵の名前を消す", font=("Meiryo", 11), command=erase).pack(side="left", padx=4)
    tk.Button(buttons, text="終わる", font=("Meiryo", 11), command=root.destroy).pack(side="right")
    root.bind("<Return>", ok)
    next_new()
    # ほかの窓の後ろに隠れないよう、開いたときだけ一番手前に出す
    root.lift()
    root.attributes("-topmost", True)
    root.after(800, lambda: root.attributes("-topmost", False))
    root.focus_force()
    root.mainloop()
    print("名前を付ける画面を閉じました")


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
