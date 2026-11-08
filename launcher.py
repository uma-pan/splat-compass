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


def _log_errors():
    """配る版では黒い窓が無いので、エラーを記録のフォルダの logs/error.txt にも残す（困ったときに送ってもらう用）。"""
    import threading
    import traceback
    from splat_compass.storage import Store
    try:
        d = os.path.join(Store().root, "logs")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "error.txt")
    except Exception:
        return

    def write(text):
        try:
            import datetime
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"--- {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n{text}\n")
        except Exception:
            pass

    sys.excepthook = lambda t, v, tb: write("".join(traceback.format_exception(t, v, tb)))
    threading.excepthook = lambda a: write("".join(traceback.format_exception(a.exc_type, a.exc_value, a.exc_traceback)))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    _fix_stdout()
    if getattr(sys, "frozen", False):
        _log_errors()
    if len(sys.argv) > 1 and sys.argv[1] == "after-match":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from splat_compass import after_match
        after_match.main()
    else:
        from splat_compass import app
        app.main()
