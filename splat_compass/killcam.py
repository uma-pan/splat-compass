"""自分を倒した相手のブキ（倒されたときの画面の「〇〇 で やられた!」）を読む。
切り抜きは hud.crop_killcam で、試合中に自分の復活待ちの表示が出た瞬間に保存される。

文字を読む仕組み（OCR）は使わず、「〇〇 で」の白い文字の形をまるごと見本と照らし合わせる。
同じブキなら文字の形は毎回まったく同じなので、似た形をまとめて一度だけ名前を付ければ、あとは自動で分かる。

  python -m splat_compass.killcam             … まだ名前の無いまとまりの一覧（HTML）を作る
  python -m splat_compass.killcam --report    … 倒されたブキの集計
名前は splat-compass-data/killcam_labels.json に「まとまりの番号: ブキ名」で書く。
"""
import argparse
import glob
import html
import json
import os
import pathlib

import cv2
import numpy as np

from .storage import Store

TEXT_ROWS = (105, 160)        # 吹き出しの中の「〇〇 で」の行
SAME = 0.6                    # 文字の形の重なり（IoU）がこれ以上なら同じブキとみなす


SUBS = {"スプラッシュボム", "キューバンボム", "クイックボム", "スプリンクラー", "スプラッシュシールド", "タンサンボム",
        "カーリングボム", "ロボットボム", "ジャンプビーコン", "ポイントセンサー", "トラップ", "ポイズンミスト",
        "ラインマーカー", "トーピード"}
SPECIALS = {"ウルトラショット", "グレートバリア", "ショクワンダー", "マルチミサイル", "アメフラシ", "ナイスダマ",
            "ホップソナー", "キューインキ", "メガホンレーザー5.1ch", "ジェットパック", "ウルトラハンコ", "カニタンク",
            "サメライド", "トリプルトルネード", "エナジースタンド", "デコイチラシ", "テイオウイカ", "ウルトラチャクチ",
            "スミナガシート"}


def kind_of(name):
    """メイン・サブ・スペシャルのどれで倒されたか。"""
    if name in SPECIALS:
        return "スペシャル"
    if name in SUBS:
        return "サブ"
    return "メイン"


def is_bubble(bubble):
    return (bubble.max(axis=2) < 60).mean() >= 0.25


def text_mask(bubble):
    """吹き出しから「〇〇 で」の白い文字だけを取り出す（左右の余白は詰める）。"""
    a, b = TEXT_ROWS
    m = bubble[a:b].min(axis=2) > 200
    cols = np.where(m.any(axis=0))[0]
    if len(cols) < 20:
        return None
    m = m[:, cols[0]:cols[-1] + 1]
    return cv2.resize(m.astype(np.uint8), (160, 24), interpolation=cv2.INTER_NEAREST).astype(bool)


def _iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def _dir(store):
    d = os.path.join(store.root, "killcam")
    os.makedirs(d, exist_ok=True)
    return d


def load_labels(store):
    """{まとまりの番号: 名前} と、まとまりごとの見本の形。"""
    try:
        with open(os.path.join(store.root, "killcam_labels.json"), encoding="utf-8") as f:
            labels = json.load(f)
        from .weapon_info import canon
        labels = {k: canon(v) for k, v in labels.items()}  # 名前の書き方をそろえて読む
    except (OSError, ValueError):
        labels = {}
    p = os.path.join(store.root, "killcam_clusters.npz")
    protos = dict(np.load(p)) if os.path.exists(p) else {}
    return labels, protos


def save_protos(store, protos):
    np.savez(os.path.join(store.root, "killcam_clusters.npz"), **protos)


def assign(mask, protos):
    """一番形が近いまとまりの番号。どれとも似ていなければ None。"""
    best, score = None, 0.0
    for k, p in protos.items():
        s = _iou(mask, p)
        if s > score:
            best, score = k, s
    return best if score >= SAME else None


def cluster_all(store):
    """killcam フォルダの画像を全部まとめ直し、まとまりの見本を保存する。[(番号, [画像のパス])]"""
    labels, protos = load_labels(store)
    groups = {k: [] for k in protos}
    for p in sorted(glob.glob(os.path.join(_dir(store), "*.png"))):
        im = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            continue
        if not is_bubble(im):
            continue
        m = text_mask(im[..., ::-1])
        if m is None:
            continue
        k = assign(m, protos)
        if k is None:
            k = f"c{len(protos):03d}"
            protos[k] = m
            groups[k] = []
        groups[k].append(p)
    save_protos(store, protos)
    return labels, groups


_ASSIGNED = {}  # 画像ごとに、どのまとまりか（まとまりの見本が変わったら覚え直す）


def name_of(store, image_path, labels=None, protos=None):
    if labels is None:
        labels, protos = load_labels(store)
    key = (image_path, len(protos), tuple(sorted(protos))[:3])
    if key not in _ASSIGNED:
        im = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_COLOR)
        m = text_mask(im[..., ::-1]) if im is not None else None
        _ASSIGNED[key] = assign(m, protos) if m is not None else None
    k = _ASSIGNED[key]
    return labels.get(k) if k else None


def build_sheet(store):
    labels, groups = cluster_all(store)
    rows = []
    for k, ps in sorted(groups.items(), key=lambda x: -len(x[1])):
        if not ps:
            continue
        uri = pathlib.Path(ps[0]).resolve().as_uri()
        rows.append(f'<tr><td>{k}</td><td>{len(ps)}</td><td>{html.escape(labels.get(k, ""))}</td>'
                    f'<td><img loading="lazy" src="{uri}" height="80"></td></tr>')
    page = ("<!DOCTYPE html><html lang='ja'><meta charset='utf-8'><title>倒されたブキのまとまり</title>"
            "<body style='font-family:Meiryo'><p>名前が空のまとまりに、killcam_labels.json で名前を付けてください"
            "（例: {\"c000\": \"パブロ・ヒュー\"}）。</p><table border=1 cellpadding=4>"
            "<tr><th>番号</th><th>回数</th><th>名前</th><th>画面</th></tr>" + "".join(rows) + "</table></body></html>")
    path = os.path.join(store.root, "killcam_groups.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path, sum(1 for k, ps in groups.items() if ps and not labels.get(k))


def report(store):
    labels, protos = load_labels(store)
    count, total, unknown = {}, 0, 0
    for m in store.matches():
        for kc in m.get("killed_by") or []:
            total += 1
            name = name_of(store, os.path.join(_dir(store), kc["image"]), labels, protos)
            if name:
                count[name] = count.get(name, 0) + 1
            else:
                unknown += 1
    if not total:
        print("倒されたときの画面の記録がまだありません")
        return
    known = total - unknown
    print(f"倒された画面 {total}回（名前が分かったもの {known}回）")
    by_kind = {}
    for name, n in count.items():
        by_kind.setdefault(kind_of(name), []).append((name, n))
    for kind in ("メイン", "サブ", "スペシャル"):
        items = sorted(by_kind.get(kind, []), key=lambda x: -x[1])
        if not items:
            continue
        sub = sum(n for _, n in items)
        print(f"\n■ {kind}で倒された: {sub}回（{sub / known * 100:.0f}%）")
        for name, n in items[:15]:
            print(f"  {name:<20} {n:4d}回")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="記録の保存先")
    ap.add_argument("--report", action="store_true", help="倒されたブキの集計")
    a = ap.parse_args()
    store = Store(a.data)
    if a.report:
        report(store)
    else:
        path, n = build_sheet(store)
        print(f"名前の無いまとまり {n}個。一覧: {path}")


if __name__ == "__main__":
    main()
