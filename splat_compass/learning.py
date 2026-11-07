"""試合が貯まるたびに勝率モデルを作り直し、良くなったときだけ入れ替える。

- 新しい試合ほど重く学習する（半減期 HALF_LIFE 試合）
- 前のモデルが学習に使っていない新しい試合で、前のモデルと新しいモデル（OOF の予測）を比べ、
  悪くならなかったときだけ入れ替える
- 毎回の結果を model_history.json に残す（試合が増えるほど当たるようになっているかが分かる）
"""
import datetime as dt
import json
import os

import numpy as np

from . import winprob
from .storage import Store

HALF_LIFE = 150   # この試合数だけ前の試合は、重みが半分になる
MIN_MATCHES = 20
MIN_NEW = 3       # 新旧を比べるのに必要な、新しい試合の数
TOLERANCE = 0.01  # 対数損失がこれ以上悪くならなければ入れ替える


def usable(store):
    ms = [m for m in store.matches() if m.get("rule") == "ガチエリア" and m.get("timeline")
          and m.get("result") in ("勝ち", "負け") and not m.get("interrupted")]
    return sorted(ms, key=lambda m: m.get("played_at", ""))


def _logloss(y, p, w):
    eps = 1e-9
    return float(-np.average(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps), weights=w))


def retrain(store=None, log=print):
    store = store or Store()
    ms = usable(store)
    if len(ms) < MIN_MATCHES:
        log(f"勝率モデル: ガチエリアの試合がまだ {len(ms)} 試合なので作りません（{MIN_MATCHES}試合から）")
        return None
    try:  # ほかのルールで使う目安のモデルは毎回作り直す（軽い）
        winprob.save_simple(winprob.fit_simple(ms), store, len(ms))
    except Exception as e:
        log(f"目安のモデルを作れませんでした: {e}")
    ids = [m["_file"] for m in ms]
    n = len(ms)
    match_w = 0.5 ** ((n - 1 - np.arange(n)) / HALF_LIFE)
    X, y, g, ts, p, w, ya = winprob.oof_predict(ms, match_w=match_w, aux=winprob.USE_AUX, spl=winprob.USE_SPLINE)
    ll, brier, auc = winprob.metrics(y, p, w)
    old = winprob.load_model(store)
    replace, reason = True, "初めてのモデル"
    if old is not None and not old.trained_on:
        reason = "前のモデルの学習に使った試合が分からないので入れ替え"
    elif old is not None:
        new_idx = [i for i, k in enumerate(ids) if k not in old.trained_on]
        if len(new_idx) >= MIN_NEW:
            sel = np.isin(g, new_idx)
            ll_old = _logloss(y[sel], old.predict(X[sel]), w[sel])
            ll_new = _logloss(y[sel], p[sel], w[sel])
            replace = bool(ll_new <= ll_old + TOLERANCE)
            reason = f"新しい{len(new_idx)}試合で 前 {ll_old:.3f} / 新 {ll_new:.3f}"
        else:
            replace = bool(old.oof_logloss is None or ll <= old.oof_logloss + TOLERANCE)
            reason = f"OOF 前 {old.oof_logloss} / 新 {ll:.3f}"
    if replace:
        full = winprob.WinModel(winprob.USE_AUX, winprob.USE_SPLINE).fit(X, y, w, ya)
        winprob.save_model(full, store, ids, float(ll))
    hist_path = os.path.join(store.root, "model_history.json")
    try:
        with open(hist_path, encoding="utf-8") as f:
            hist = json.load(f)
    except (OSError, ValueError):
        hist = []
    hist.append({"at": dt.datetime.now().isoformat(timespec="seconds"), "matches": n,
                 "oof_logloss": round(ll, 4), "oof_auc": round(auc, 4), "replaced": replace, "reason": reason})
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=1)
    log(f"勝率モデル: {n}試合で作り直し（OOF 対数損失 {ll:.3f} / AUC {auc:.3f}）→ "
        + ("入れ替えました" if replace else "前のモデルのままにしました") + f"（{reason}）")
    return replace


def predict_match(store, m, area_model=None):
    """1試合の毎秒の勝率 (時刻の並び, 勝率の並び)。出せなければ None。
    ガチエリアは勝率モデル、ほかのルールはエリアから作った簡単なモデル（目安）。"""
    rs = winprob.rows(m)
    if len(rs) < 10:
        return None
    X = np.array([f for _, f in rs])
    tt = np.array([t for t, _ in rs])
    if m.get("rule") == "ガチエリア":
        model = area_model or winprob.load_model(store)
        return (tt, model.predict(X)) if model is not None else None
    if m.get("rule") in winprob.SIMPLE_RULES:
        simple = winprob.load_simple(store)
        return (tt, simple.predict(X[:, winprob.SIMPLE_COLS])) if simple is not None else None
    return None


def contribution(store, m, before=1.0, after=4.0, area_model=None):
    """自分の行動の前後で勝率がどれだけ動いたかの合計。{'倒した'/'倒された'/'スペシャル': [回数, 合計]}
    1つ1つの行動の前1秒と後4秒の勝率の差を足す。ほかの人の動きも混ざるので、目安として見る。"""
    pr = predict_match(store, m, area_model)
    me = m.get("me_slot")
    if pr is None or not me:
        return None
    tt, pp = pr

    def wp(t):
        return pp[min(max(np.searchsorted(tt, t), 0), len(pp) - 1)]
    out = {"倒した": [0, 0.0], "倒された": [0, 0.0], "スペシャル": [0, 0.0]}

    def add(key, t):
        if tt[0] <= t - before and t + after <= tt[-1]:
            out[key][0] += 1
            out[key][1] += float(wp(t + after) - wp(t - before))
    for k in m.get("kills") or []:
        add("倒した", k["t"])
    for t, rel, slot, what, *_ in m["events"]:
        if slot != me:
            continue
        if "倒された" in what:
            add("倒された", t)
        elif what == "スペシャル発動":
            add("スペシャル", t)
    return out


def swings(model, m, top=3, before=1.0, after=4.0):
    """1試合の中で勝率が大きく動いた出来事を返す。[(試合開始からの秒, 説明, 勝率の動き)]"""
    rs = winprob.rows(m)
    if len(rs) < 10:
        return []
    tt = np.array([t for t, _ in rs])
    pp = model.predict(np.array([f for _, f in rs]))

    def wp(t):
        return pp[min(max(np.searchsorted(tt, t), 0), len(pp) - 1)]
    me = m.get("me_slot")
    groups = []  # 1秒以内に続いた出来事はひとまとめにする（全滅など）
    evs = list(m["events"]) + [[k["t"], k["rel"], me or "", "自分のキル"] for k in m.get("kills") or []]
    for t, rel, slot, what, *_ in sorted(evs, key=lambda e: e[0]):
        if what not in ("倒された", "スペシャルを持ったまま倒された", "スペシャル発動", "自分のキル"):
            continue
        if t - before < tt[0] or t + after > tt[-1]:
            continue
        if what == "自分のキル":
            label = "自分のキル"
        else:
            who = "自分" if slot == me else ("味方" if slot[0] == "左" else "敵")
            label = f"{who}{'のスペシャル' if what == 'スペシャル発動' else 'デス'}"
        if groups and t - groups[-1]["t_last"] <= 1.0:
            groups[-1]["labels"].append(label)
            groups[-1]["t_last"] = t
        else:
            groups.append({"t": t, "t_last": t, "rel": rel, "labels": [label]})
    out = []
    for gr in groups:
        counts = {}
        for lb in gr["labels"]:
            counts[lb] = counts.get(lb, 0) + 1
        desc = "・".join(f"{k}×{v}" if v > 1 else k for k, v in counts.items())
        out.append((gr["rel"], desc, wp(gr["t_last"] + after) - wp(gr["t"] - before)))
    return sorted(out, key=lambda x: -abs(x[2]))[:top]


def main():
    retrain()


if __name__ == "__main__":
    main()
