"""配る版に同梱する、最初から使えるもの（defaults フォルダ）を、自分の記録から作る。

- weapon_refs.npz: ブキの見本（名前ごとに数枚分の特徴。絵そのものは入れない）
- winprob_area.json / winprob_simple.json: 勝率モデル

  python -m splat_compass.make_defaults
"""
import argparse
import json
import os

from . import weapons, winprob
from .storage import Store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    a = ap.parse_args()
    store = Store(a.data) if a.data else Store()
    os.makedirs(winprob.DEFAULTS_DIR, exist_ok=True)
    n, k = weapons.export_refs(store)
    print(f"ブキの見本: {n}種類 {k}枚分")
    for name in ("winprob_area.json", "winprob_simple.json"):
        src = os.path.join(store.root, name)
        if os.path.exists(src):
            with open(src, encoding="utf-8") as f:
                d = json.load(f)
            d.pop("trained_on", None)  # どの試合で学習したかは配らない
            with open(os.path.join(winprob.DEFAULTS_DIR, name), "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=1)
            print(f"勝率モデル: {name}")


if __name__ == "__main__":
    main()
