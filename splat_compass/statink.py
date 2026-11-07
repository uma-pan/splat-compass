"""stat.ink の公開データ（世界中のプレイヤーが上げた試合結果）を取り込み、集計する。

自分の試合だけでは足りない「ブキごとの勝率」「ブキごとの平均キル・デス」を、ずっと多い試合から出す。

  python -m splat_compass.statink            … 直近30日分をダウンロードして集計（取ってある日は飛ばす）
  python -m splat_compass.statink --days 120
  python -m splat_compass.statink --no-download  … ダウンロードせずに集計だけやり直す

データ: stat.ink（https://stat.ink/downloads）。使うときは出典として stat.ink を書くこと。
注意:
- 上げているのはやり込んでいる人が多く、腕前は平均より高めに偏る。
- 各試合の A1 は上げた本人。ブキの数や勝率が偏るので、ブキごとの集計からは除く（stat.ink の説明より）。
"""
import argparse
import csv
import io
import json
import os
import re
import time
import urllib.parse
import urllib.request
from array import array
from collections import defaultdict

import numpy as np

from .storage import Store
from .weapon_info import norm

BASE = "https://stat.ink"
DOWNLOADS = BASE + "/downloads"
WEAPON_API = BASE + "/api/v3/weapon"
UA = "splat-compass (personal analysis tool; https://github.com/uma-pan/splat-compass)"
LOBBIES = ("xmatch", "bankara_challenge", "bankara_open")  # ガチルールのマッチだけ使う
MODES = {"area": "ガチエリア", "yagura": "ガチヤグラ", "hoko": "ガチホコ", "asari": "ガチアサリ"}
PLAYERS = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]


def _dir(store):
    d = os.path.join(store.root, "statink")
    os.makedirs(os.path.join(d, "csv"), exist_ok=True)
    return d


def _get(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


# ---------- ダウンロード ----------
def find_csv_links(html, page_url=DOWNLOADS):
    """ダウンロードページから、スプラトゥーン3のバトル結果の日ごとの CSV を探す。[(日付, URL)] 新しい順。"""
    out = {}
    for href in re.findall(r'href="([^"]+?\.csv(?:\.gz)?)"', html):
        if "battle" not in href or ("splatoon-3" not in href and "spl3" not in href and "splatoon3" not in href):
            continue
        m = re.search(r"(\d{4}-\d{2}-\d{2})", href)
        if not m:
            continue
        url = urllib.parse.urljoin(page_url, href.replace("&amp;", "&"))
        out[m.group(1)] = url
    return sorted(out.items(), reverse=True)


def download(store, days=30, log=print):
    d = _dir(store)
    log("stat.ink のダウンロードページを読んでいます…")
    html = _get(DOWNLOADS).decode("utf-8", "replace")
    with open(os.path.join(d, "downloads_page.html"), "w", encoding="utf-8") as f:
        f.write(html)  # ページの形が変わったときに確かめられるよう残しておく
    links = find_csv_links(html)
    if not links:
        cands = sorted(set(re.findall(r'href="([^"]*(?:csv|zip|battle|splatoon-?3)[^"]*)"', html)))
        log(f"ダウンロードページにあったそれらしいリンク {len(cands)} 件:")
        for c in cands[:40]:
            log("  " + c)
        raise RuntimeError("ダウンロードページに日ごとの CSV が見つかりませんでした。"
                           f"ページを {os.path.join(d, 'downloads_page.html')} に保存したので、確かめて直します")
    got = 0
    for day, url in links[:days]:
        path = os.path.join(d, "csv", f"{day}.csv")
        if os.path.exists(path):
            continue
        data = _get(url)
        if url.endswith(".gz"):
            import gzip
            data = gzip.decompress(data)
        with open(path + ".tmp", "wb") as f:
            f.write(data)
        os.replace(path + ".tmp", path)
        got += 1
        log(f"  {day} をダウンロードしました（{len(data) // 1024}KB）")
        time.sleep(1.0)  # 相手のサーバーに負担をかけないよう、間をあける
    log(f"新しく {got} 日分をダウンロードしました（全部で {len(os.listdir(os.path.join(d, 'csv')))} 日分）")
    try:
        names = weapon_names_from_api(json.loads(_get(WEAPON_API).decode("utf-8")))
        with open(os.path.join(d, "weapons.json"), "w", encoding="utf-8") as f:
            json.dump(names, f, ensure_ascii=False, indent=1)
        log(f"ブキの日本語名を {len(names)} 件読みました")
    except Exception as e:
        log(f"ブキの日本語名を読めませんでした（{e}）。前に読んだものを使います")


def weapon_names_from_api(data):
    """stat.ink のブキ一覧（API）から {キー: 日本語名}。"""
    items = data if isinstance(data, list) else data.get("weapons") or data.get("data") or []
    out = {}
    for w in items:
        key = w.get("key")
        name = w.get("name")
        ja = name.get("ja_JP") if isinstance(name, dict) else None
        if key and ja:
            out[key] = ja
    return out


def weapon_names(store):
    try:
        with open(os.path.join(_dir(store), "weapons.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


# ---------- 読み込み ----------
def battles(store, lobbies=LOBBIES):
    """ガチルールの試合を1つずつ返す: {'mode', 'a_win', 'time', 'players': [(名前, キル, デス, スペシャル)]×8}"""
    names = weapon_names(store)
    cdir = os.path.join(_dir(store), "csv")
    for fn in sorted(os.listdir(cdir)):
        if not fn.endswith(".csv"):
            continue
        with open(os.path.join(cdir, fn), encoding="utf-8", newline="") as f:
            lines = (ln for ln in f if not ln.startswith("#"))
            for row in csv.DictReader(lines):
                mode = MODES.get(row.get("mode"))
                if mode is None or row.get("lobby") not in lobbies or row.get("win") not in ("alpha", "bravo"):
                    continue
                ps = []
                for p in PLAYERS:
                    key = row.get(f"{p}-weapon")
                    if not key:
                        break
                    ps.append((norm(names.get(key, key)), _int(row.get(f"{p}-kill")),
                               _int(row.get(f"{p}-death")), _int(row.get(f"{p}-special"))))
                if len(ps) != 8:
                    continue
                yield {"mode": mode, "a_win": row["win"] == "alpha", "time": _int(row.get("time")) or 300, "players": ps}


def _int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


# ---------- 集計 ----------
def fit_strength(A, B, y, n_names, l2=1e-4, iters=300, lr=1.0):
    """ブキの強さ（編成の有利さの元）。チームの強さ = 4人のブキの強さの合計 として、
    「Aチームが勝つ確率 = シグモイド(上げた人の分の補正 + Aの強さ − Bの強さ)」を学習する。
    A, B: 試合ごとの4人のブキの番号 (試合数, 4)、y: Aが勝ったか。"""
    w = np.zeros(n_names)
    b0 = 0.0
    n = len(y)
    for _ in range(iters):
        s = b0 + w[A].sum(1) - w[B].sum(1)
        p = 1 / (1 + np.exp(-s))
        r = (p - y) / n
        g = np.zeros(n_names)
        for k in range(4):
            np.add.at(g, A[:, k], r)
            np.add.at(g, B[:, k], -r)
        w -= lr * (g + l2 * w)
        b0 -= lr * r.sum()
    return w, b0


def summarize(store, log=print):
    """ブキごとの勝率・平均キル/デス・強さを集計して statink/summary.json に残す。
    試合数が多くてもメモリを食わないよう、ブキは番号にして詰めて持つ。"""
    idx = {}
    per_mode = defaultdict(lambda: {"w": array("H"), "y": array("b")})
    kd = defaultdict(lambda: [0, 0, 0, 0, 0])  # [人数, キル, デス, スペシャル, 秒]
    total = 0
    for b in battles(store):
        total += 1
        pm = per_mode[b["mode"]]
        for i, (n, k, d, sp) in enumerate(b["players"]):
            pm["w"].append(idx.setdefault(n, len(idx)))
            if i == 0 or k is None or d is None:  # A1 は上げた本人なので除く
                continue
            e = kd[n]
            e[0] += 1
            e[1] += k
            e[2] += d
            e[3] += sp or 0
            e[4] += b["time"]
        pm["y"].append(1 if b["a_win"] else 0)
    if not total:
        raise RuntimeError("使える試合がありませんでした（まだダウンロードしていないかもしれません）")
    names = [None] * len(idx)
    for n, i in idx.items():
        names[i] = n
    out = {"battles": total, "files": sorted(os.listdir(os.path.join(_dir(store), "csv"))), "modes": {},
           "kd": dict(kd), "made_at": time.strftime("%Y-%m-%d %H:%M")}
    for mode, pm in per_mode.items():
        W = np.frombuffer(pm["w"], dtype=np.uint16).reshape(-1, 8).astype(np.int64)
        y = np.frombuffer(pm["y"], dtype=np.int8).astype(float)
        A, B = W[:, :4], W[:, 4:]
        # 勝率（A1 を除く7人）
        wins = np.c_[np.repeat(y[:, None], 3, 1), np.repeat(1 - y[:, None], 4, 1)]
        cnt = np.bincount(W[:, 1:].ravel(), minlength=len(names))
        won = np.bincount(W[:, 1:].ravel(), weights=wins.ravel(), minlength=len(names))
        wr = {names[i]: [int(cnt[i]), int(won[i])] for i in range(len(names)) if cnt[i]}
        w, bias = fit_strength(A, B, y, len(names))
        used = np.bincount(W.ravel(), minlength=len(names)) > 0
        strength = {names[i]: float(w[i]) for i in range(len(names)) if used[i]}
        out["modes"][mode] = {"battles": int(len(y)), "wr": wr, "strength": strength, "bias": float(bias)}
        log(f"  {mode}: {len(y)}試合")
    with open(os.path.join(_dir(store), "summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    log(f"集計しました（{total}試合）")
    return out


def load_summary(store):
    try:
        with open(os.path.join(store.root, "statink", "summary.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def summary_mtime(store):
    p = os.path.join(store.root, "statink", "summary.json")
    return os.path.getmtime(p) if os.path.exists(p) else 0


def comp_prior(summary, mode, ally, enemy):
    """編成だけで見た勝率（stat.ink のブキの強さから）。分からないブキは強さ0として扱う。"""
    if not summary or mode not in summary.get("modes", {}):
        return None
    s = summary["modes"][mode]["strength"]
    x = sum(s.get(norm(n), 0.0) for n in ally if n) - sum(s.get(norm(n), 0.0) for n in enemy if n)
    return float(1 / (1 + np.exp(-x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="記録の保存先")
    ap.add_argument("--days", type=int, default=30, help="直近何日分を使うか")
    ap.add_argument("--no-download", action="store_true", help="ダウンロードせずに集計だけやり直す")
    a = ap.parse_args()
    store = Store(a.data)
    if not a.no_download:
        download(store, a.days)
    summarize(store)


if __name__ == "__main__":
    main()
