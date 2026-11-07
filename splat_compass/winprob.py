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
    "味方カウントの減り（直近15秒）", "敵カウントの減り（直近15秒）", "勢い（敵の減り−味方の減り）",
    "延長戦", "味方のデス（直近10秒）", "敵のデス（直近10秒）",
]
MOMENTUM = 15.0  # カウントの勢いを見る秒数
RECENT = 10.0    # 直近のデスを数える秒数
AUX_AHEAD = 30.0  # 近い未来の予測（補助）で見る秒数


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


def context(timeline, events, t):
    """勢いと直近のデスを計算するための材料: (15秒前の味方カウント, 敵カウント, 直近10秒の味方のデス, 敵のデス)。"""
    l_ago = r_ago = None
    for row in timeline:
        if row[0] > t - MOMENTUM:
            break
        l_ago, r_ago = row[1], row[2]
    if l_ago is None and timeline:
        l_ago, r_ago = timeline[0][1], timeline[0][2]
    ad = ed = 0
    for e in events:
        if t - RECENT < e[0] <= t and "倒された" in e[3]:
            if e[2][0] == "左":
                ad += 1
            else:
                ed += 1
    return l_ago, r_ago, ad, ed


def features(elapsed, l, r, lf, rf, slots, ctx=(None, None, 0, 0)):
    """1時点の特徴量。slots は8枠の状態（0=倒されている 1=生きている 2=スペシャルあり、左4つが味方）。"""
    el = min(elapsed / MATCH_LEN, 1.2)
    ally = sum(x > 0 for x in slots[:4])
    enemy = sum(x > 0 for x in slots[4:])
    ctl = 1 if lf and not rf else (-1 if rf and not lf else 0)
    diff = (r - l) / 100  # カウントは少ないほど有利。正なら味方がリード
    l_ago, r_ago, ad, ed = ctx
    ld = ((l_ago if l_ago is not None else l) - l) / 100  # 味方カウントの減り（味方が進めた量）
    rd = ((r_ago if r_ago is not None else r) - r) / 100
    return [el, diff, diff * el, l / 100, r / 100, ctl, ctl * el, (ally - enemy) / 4,
            ally / 4, enemy / 4, sum(x == 2 for x in slots[:4]) / 4, sum(x == 2 for x in slots[4:]) / 4,
            ld, rd, ld - rd, float(elapsed > MATCH_LEN), ad / 4, ed / 4]


def rows(m):
    """1試合を (時刻, 特徴量) の並びにする。"""
    at = _states_at(m)
    tl = m.get("timeline") or []
    evs = sorted(m["events"], key=lambda e: e[0])
    out = []
    for t, l, r, lf, rf in tl:
        st = at(t)
        out.append((t, features(t - m["video_start"], l, r, lf, rf, [st[s] for s in SLOTS],
                                context(tl, evs, t))))
    return out


def aux_target(m):
    """各時点から30秒後に、カウント差（味方のリード）がどれだけ動いたか。"""
    tl = m.get("timeline") or []
    out = []
    j = 0
    for i, (t, l, r, *_) in enumerate(tl):
        while j < len(tl) - 1 and tl[j][0] < t + AUX_AHEAD:
            j += 1
        out.append(((tl[j][2] - tl[j][1]) - (r - l)) / 100)
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


class Ridge:
    def fit(self, X, y, w, l2=1.0):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        W = w / w.mean()
        A = Z.T @ (Z * W[:, None]) + l2 * np.diag(np.r_[0, np.ones(Z.shape[1] - 1)])
        self.b = np.linalg.solve(A, Z.T @ (y * W))
        return self

    def predict(self, X):
        return np.c_[np.ones(len(X)), (X - self.mu) / self.sd] @ self.b


class WinModel:
    """勝率モデル。aux=True なら、まず「30秒後のカウント差の動き」を予測し、それも入力に加えて勝率を出す。"""

    def __init__(self, aux=False):
        self.aux = aux
        self.ridge = None

    def fit(self, X, y, w, y_aux=None):
        if self.aux:
            self.ridge = Ridge().fit(X, y_aux, w)
            X = np.c_[X, self.ridge.predict(X)]
        self.logit = Logistic().fit(X, y, w)
        return self

    def predict(self, X):
        if self.aux:
            X = np.c_[X, self.ridge.predict(X)]
        return self.logit.predict(X)

    # 保存用
    def to_dict(self):
        d = {"aux": self.aux, "mu": self.logit.mu.tolist(), "sd": self.logit.sd.tolist(), "b": self.logit.b.tolist()}
        if self.aux:
            d["ridge"] = {"mu": self.ridge.mu.tolist(), "sd": self.ridge.sd.tolist(), "b": self.ridge.b.tolist()}
        return d

    @classmethod
    def from_dict(cls, d):
        m = cls(d.get("aux", False))
        m.logit = Logistic()
        m.logit.mu, m.logit.sd, m.logit.b = np.array(d["mu"]), np.array(d["sd"]), np.array(d["b"])
        if m.aux:
            m.ridge = Ridge()
            r = d["ridge"]
            m.ridge.mu, m.ridge.sd, m.ridge.b = np.array(r["mu"]), np.array(r["sd"]), np.array(r["b"])
        return m


def _dataset(matches, cols=None):
    X, y, g, ts, ya = [], [], [], [], []
    for i, m in enumerate(matches):
        lab = 1.0 if m["result"] == "勝ち" else 0.0
        for (t, f), a in zip(rows(m), aux_target(m)):
            X.append(f if cols is None else [f[c] for c in cols])
            y.append(lab)
            g.append(i)
            ts.append(t)
            ya.append(a)
    return np.array(X), np.array(y), np.array(g), np.array(ts), np.array(ya)


def oof_predict(matches, cols=None, k=5, seed=0, match_w=None, aux=False):
    """試合ごとに k 分割した交差検証で、全部の秒の OOF 予測を作る。
    1試合の重みは秒数によらず同じ（match_w を渡せば試合ごとに変えられる）。"""
    X, y, g, ts, ya = _dataset(matches, cols)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(matches))
    fold = np.empty(len(matches), int)
    fold[order] = np.arange(len(matches)) % k
    n_per = np.bincount(g, minlength=len(matches))
    w = 1.0 / n_per[g]
    if match_w is not None:
        w = w * np.asarray(match_w)[g]
    p = np.zeros(len(y))
    for f in range(k):
        tr, te = fold[g] != f, fold[g] == f
        p[te] = WinModel(aux).fit(X[tr], y[tr], w[tr], ya[tr]).predict(X[te])
    return X, y, g, ts, p, w, ya


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


USE_AUX = False  # 補助の予測を使うか（report の比較で決める。80試合では差がなかったので今は使わない）
OLD_COLS = list(range(12))  # 勢い・延長・直近のデスを足す前の特徴


def calibration(y, p, w, bins=(0, .2, .4, .6, .8, 1.0001)):
    """予測した勝率の帯ごとに、実際に勝った割合を比べる。"""
    out = []
    for a, b in zip(bins, bins[1:]):
        s = (p >= a) & (p < b)
        if w[s].sum() > 0:
            out.append((a, b, float(np.average(p[s], weights=w[s])), float(np.average(y[s], weights=w[s])),
                        int(len(set(np.where(s)[0])))))
    return out


def report(matches):
    ms = [m for m in matches if m.get("rule") == "ガチエリア" and m.get("timeline")
          and m.get("result") in ("勝ち", "負け")]
    if len(ms) < 20:
        print(f"勝敗とカウントの記録があるガチエリアの試合が {len(ms)} 試合しかありません（20試合以上必要）")
        return None
    print(f"ガチエリア {len(ms)}試合で、試合ごとに分けた5分割の交差検証（OOF）。5回くじを変えた平均")
    configs = (("カウント差と時間だけ", [0, 1, 2], False), ("前回の特徴", OLD_COLS, False),
               ("勢い・延長・直近のデスを追加", None, False), ("＋30秒後の予測を補助に", None, True))
    for name, cols, aux in configs:
        res = [metrics(*(lambda r: (r[1], r[4], r[5]))(oof_predict(ms, cols, seed=sd, aux=aux))) for sd in range(5)]
        ll, br, auc = np.mean(res, axis=0)
        print(f"  {name:<18} 対数損失 {ll:.3f}  Brier {br:.3f}  AUC {auc:.3f}")
    print(f"  （何も分からないときの対数損失は {np.log(2):.3f}）")
    X, y, g, ts, p, w, ya = oof_predict(ms, aux=USE_AUX)
    print("  時間帯ごとの AUC:", "  ".join(
        f"{a}〜{b}秒 {metrics(y[s], p[s], w[s])[2]:.2f}" for a, b in ((0, 100), (100, 200), (200, 400))
        for s in [(X[:, 0] * MATCH_LEN >= a) & (X[:, 0] * MATCH_LEN < b)] if s.sum() > 100))
    print("\n■ 確率のずれ（予測した勝率の帯ごとに、実際に勝った割合）")
    for a, b, pm, ym, n in calibration(y, p, w):
        print(f"  予測 {a * 100:3.0f}〜{min(b, 1) * 100:3.0f}%  予測の平均 {pm * 100:4.1f}%  実際 {ym * 100:4.1f}%  （{n}秒分）")
    full = WinModel(USE_AUX).fit(X, y, w, ya)
    names = FEATURES + (["30秒後のカウント差の動き（予測）"] if USE_AUX else [])
    print("\n■ 全試合で学習したときの係数（標準化後。正なら勝ちやすくなる向き）")
    for name, b in sorted(zip(names, full.logit.b[1:]), key=lambda x: -abs(x[1]))[:10]:
        print(f"  {name:<24} {b:+.2f}")
    print("\n■ 出来事の前後（1秒前→15秒後）で勝率がどれだけ動いたか（OOF の予測で計算）")
    vals = event_values(ms, ts, g, p, after=15.0)
    for key in [("自分", "倒された"), ("味方", "倒された"), ("敵", "倒された"),
                ("自分", "スペシャル発動"), ("味方", "スペシャル発動"), ("敵", "スペシャル発動")]:
        v = vals.get(key, [])
        if v:
            print(f"  {key[0]}が{key[1]:<10} {len(v):5d}回  平均 {np.mean(v) * 100:+5.1f}%")
    return full


def load_model(store):
    path = os.path.join(store.root, "winprob_area.json")
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("features") != FEATURES:
            return None  # 特徴が変わった古いモデルは使わない（次の作り直しで置き換わる）
        m = WinModel.from_dict(d["model"])
    except (OSError, ValueError, KeyError):
        return None
    m.trained_on = set(d.get("trained_on", []))
    m.oof_logloss = d.get("oof_logloss")
    return m


def save_model(model, store, trained_on=(), oof_logloss=None):
    path = os.path.join(store.root, "winprob_area.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"features": FEATURES, "model": model.to_dict(), "trained_on": sorted(trained_on),
                   "oof_logloss": oof_logloss}, f, ensure_ascii=False, indent=1)
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
