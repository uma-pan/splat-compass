"""試合が終わったあとの重い作業（勝率モデルの学習のやり直しと、まとめのページの作り直し）。

アプリの中で動かすと、試合中の読み取りと計算の取り合いになって読み取りが遅れるので、別のプロセスで動かす。

  python -m splat_compass.after_match --data <記録のフォルダ>
"""
import argparse

from . import learning, report
from .storage import Store


def main():
    import sys
    try:  # 呼び出し元は UTF-8 として読むので、Windows の既定の文字コードではなく UTF-8 で書く
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    a = ap.parse_args()
    store = Store(a.data) if a.data else Store()
    learning.retrain(store, log=lambda s: print(s, flush=True))
    try:
        report.build(store, store.load_config().get("video_dir"))
        print("@report", flush=True)
    except Exception as e:
        print(f"まとめのページを作れませんでした: {e}", flush=True)


if __name__ == "__main__":
    main()
