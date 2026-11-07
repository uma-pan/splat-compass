"""勝率予測モデル（いまはガチエリアのみ）。

毎秒の試合の状態から「この状態から勝つ確率」を予測するロジスティック回帰を作る。
評価は試合ごとに分けた交差検証（OOF）で行い、同じ試合の秒どうしが学習と評価に混ざらないようにする。
OOF の予測を使って、出来事の前後で勝率がどれだけ動いたかを数える。

  python -m splat_compass.winprob
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

from .storage import Store

SLOTS = ["左1", "左2", "左3", "左4", "右1", "右2", "右3", "右4"]
MATCH_LEN = 300.0  # 延長を除いた試合時間（秒）

FEATURES = [
    "経過時間", "カウント差", "カウント差×経過時間", "味方カウント", "敵カウント",
    "確保（味方+1/敵-1）", "確保×経過時間", "人数差", "味方の生存", "敵の生存",
    "味方のスペシャル準備", "敵のスペシャル準備",
]


def _states_at(m):
    """出来事を時刻順にたどって、各時刻の生存・スペシャルの状態を返す関数を作る。"""
    evs = sorted(m["events"], key=lambda e: e[0])

    def at(t):
        st = {s: 1 for s in SLOTS}
        for e in evs:
            if e[0] > t:
                break
            slot, what = e[2], e[3]
            if "倒された" in what:
                st[slot] = 0
            elif what in ("復活", "スペシャル発動"):
                st[slot] = 1
            elif what == "スペシャル溜まった":
                st[slot] = 2
        return st
    return at


def rows(m):
    """1試合を (時刻, 特徴量) の並びにする。"""
    at = _states_at(m)
    out = []
    for t, l, r, lf, rf in m.get("timeline") or []:
        el = min((t - m["video_start"]) / MATCH_LEN, 1.2)
        st = at(t)
        ally = sum(st[s] > 0 for s in SLOTS[:4])
        enemy = sum(st[s] > 0 for s in SLOTS[4:])
        ctl = 1 if lf and not rf else (-1 if rf and not lf else 0)
        diff = (r - l) / 100  # カウントは少ないほど有利。正なら味方がリード
        out.append((t, [el, diff, diff * el, l / 100, r / 100, ctl, ctl * el, (ally - enemy) / 4,
                        ally / 4, enemy / 4, sum(st[s] == 2 for s in SLOTS[:4]) / 4,
                        sum(st[s] == 2 for s in SLOTS[4:]) / 4]))
    return out


class Logistic:
    """numpy だけで書いたロジスティック回帰（L2 正則化つき）。"""

    def __init__(self, l2=1e-2, iters=2000, lr=0.5):
        self.l2, self.iters, self.lr = l2, iters, lr

    def fit(self, X, y, w=None):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        w = np.ones(len(X)) if w is None else w
        self.b = np.zeros(Z.shape[1])
        for _ in range(self.iters):
            p = 1 / (1 + np.exp(-Z @ self.b))
            g = Z.T @ ((p - y) * w) / w.sum() + self.l2 * np.r_[0, self.b[1:]]
            self.b -= self.lr * g
        return self

    def predict(self, X):
        Z = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        return 1 / (1 + np.exp(-Z @ self.b))


def _dataset(matches, cols=None):
    X, y, g, ts = [], [], [], []
    for i, m in enumerate(matches):
        lab = 1.0 if m["result"] == "勝ち" else 0.0
        for t, f in rows(m):
            X.append(f if cols is None else [f[c] for c in cols])
            y.append(lab)
            g.append(i)
            ts.append(t)
    return np.array(X), np.array(y), np.array(g), np.array(ts)


def oof_predict(matches, cols=None, k=5, seed=0):
    """試合ごとに k 分割した交差検証で、全部の秒の OOF 予測を作る。試合ごとの重みは秒数によらず同じにする。"""
    X, y, g, ts = _dataset(matches, cols)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(matches))
    fold = np.empty(len(matches), int)
    fold[order] = np.arange(len(matches)) % k
    n_per = np.bincount(g, minlength=len(matches))
    w = 1.0 / n_per[g]
    p = np.zeros(len(y))
    for f in range(k):
        tr, te = fold[g] != f, fold[g] == f
        p[te] = Logistic().fit(X[tr], y[tr], w[tr]).predict(X[te])
    return X, y, g, ts, p, w


def metrics(y, p, w):
    eps = 1e-9
    ll = -np.average(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps), weights=w)
    brier = np.average((p - y) ** 2, weights=w)
    pos, neg = p[y == 1], p[y == 0]
    wp, wn = w[y == 1], w[y == 0]
    # 重みつき AUC（勝ち試合の秒と負け試合の秒の組で、勝ち側の予測が高い割合）
    order = np.argsort(neg)
    neg_s, wn_s = neg[order], wn[order]
    cum = np.r_[0, np.cumsum(wn_s)]
    idx = np.searchsorted(neg_s, pos, side="left")
    idx_r = np.searchsorted(neg_s, pos, side="right")
    auc = np.sum(wp * (cum[idx] + 0.5 * (cum[idx_r] - cum[idx]))) / (wp.sum() * wn.sum())
    return ll, brier, auc


def event_values(matches, ts, g, p, before=1.0, after=4.0):
    """出来事の前後で OOF 予測の勝率がどれだけ動いたか。"""
    by_match = defaultdict(list)
    for i in range(len(p)):
        by_match[g[i]].append((ts[i], p[i]))
    vals = defaultdict(list)
    for gi, m in enumerate(matches):
        seq = sorted(by_match.get(gi, []))
        if len(seq) < 10:
            continue
        tt = np.array([a for a, _ in seq])
        pp = np.array([b for _, b in seq])
        me = m.get("me_slot")

        def wp(t):
            i = np.searchsorted(tt, t)
            return pp[min(max(i, 0), len(pp) - 1)]
        for t, rel, slot, what, *_ in m["events"]:
            if t - before < tt[0] or t + after > tt[-1]:
                continue
            side = "自分" if slot == me else ("味方" if slot[0] == "左" else "敵")
            kind = "倒された" if "倒された" in what else what
            if kind in ("倒された", "スペシャル発動"):
                vals[(side, kind)].append(wp(t + after) - wp(t - before))
    return vals


def report(matches):
    ms = [m for m in matches if m.get("rule") == "ガチエリア" and m.get("timeline")
          and m.get("result") in ("勝ち", "負け")]
    if len(ms) < 20:
        print(f"勝敗とカウントの記録があるガチエリアの試合が {len(ms)} 試合しかありません（20試合以上必要）")
        return None
    print(f"ガチエリア {len(ms)}試合で、試合ごとに分けた5分割の交差検証（OOF）")
    base_cols = [0, 1, 2]  # 経過時間とカウント差だけ
    for name, cols in (("カウント差と時間だけ", base_cols), ("全部の特徴", None)):
        X, y, g, ts, p, w = oof_predict(ms, cols)
        ll, br, auc = metrics(y, p, w)
        print(f"  {name:<12} 対数損失 {ll:.3f}  Brier {br:.3f}  AUC {auc:.3f}")
    print(f"  （何も分からないときの対数損失は {np.log(2):.3f}）")
    X, y, g, ts, p, w = oof_predict(ms)
    print("  試合の時間帯ごとの AUC（全部の特徴）:", "  ".join(
        f"{a}〜{b}秒 {metrics(y[s], p[s], w[s])[2]:.2f}" for a, b in ((0, 100), (100, 200), (200, 400))
        for s in [(X[:, 0] * MATCH_LEN >= a) & (X[:, 0] * MATCH_LEN < b)] if s.sum() > 100))
    full = Logistic().fit(X, y, w)
    print("\n■ 全試合で学習したときの係数（標準化後。正なら勝ちやすくなる向き）")
    for name, b in sorted(zip(FEATURES, full.b[1:]), key=lambda x: -abs(x[1])):
        print(f"  {name:<16} {b:+.2f}")
    for after in (4.0, 15.0):
      print(f"\n■ 出来事の前後（1秒前→{int(after)}秒後）で勝率がどれだけ動いたか（OOF の予測で計算）")
      vals = event_values(ms, ts, g, p, after=after)
      for key in [("自分", "倒された"), ("味方", "倒された"), ("敵", "倒された"),
                ("自分", "スペシャル発動"), ("味方", "スペシャル発動"), ("敵", "スペシャル発動")]:
        v = vals.get(key, [])
        if v:
            print(f"  {key[0]}が{key[1]:<10} {len(v):5d}回  平均 {np.mean(v) * 100:+5.1f}%")
    return full


def save_model(model, store):
    path = os.path.join(store.root, "winprob_area.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"features": FEATURES, "mu": model.mu.tolist(), "sd": model.sd.tolist(),
                   "b": model.b.tolist()}, f, ensure_ascii=False, indent=1)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="記録の保存先")
    a = ap.parse_args()
    store = Store(a.data)
    model = report([m for m in store.matches() if not m.get("interrupted")])
    if model is not None:
        print(f"\nモデルを保存しました: {save_model(model, store)}")


if __name__ == "__main__":
    main()
