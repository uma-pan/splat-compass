"""配る版（exe）の入り口。PyInstaller でこのファイルから exe を作る。

  splat-compass.exe               … アプリを開く
  splat-compass.exe after-match   … 試合後の作業（アプリが自分で呼ぶ）
"""
import multiprocessing
import os
import sys


def _fix_stdout():
    """窓だけの exe では print の出力先が無いことがあるので、渡された出力先（パイプ）につなぎ直す。"""
    for name, fd in (("stdout", 1), ("stderr", 2)):
        if getattr(sys, name) is None:
            try:
                setattr(sys, name, open(fd, "w", encoding="utf-8", closefd=False))
            except OSError:
                setattr(sys, name, open(os.devnull, "w"))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    _fix_stdout()
    if len(sys.argv) > 1 and sys.argv[1] == "after-match":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from splat_compass import after_match
        after_match.main()
    else:
        from splat_compass import app
        app.main()
