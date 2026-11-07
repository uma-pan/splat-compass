"""見返したい場面を録画から選び、ブラウザで再生できる一覧（HTML）を作る。

選ぶ場面
- 勝率が大きく動いた場面（ガチエリア、勝率モデルがあるとき）
- 打開に失敗した・長引いた場面（エリアは確保されてから、ヤグラ・ホコは押し込まれてから）
- 自分がスペシャルを持ったまま倒された場面
- 自分が復活してすぐ（8秒以内）に倒された場面

  python -m splat_compass.scenes "D:\\Spla Capture"            … 全試合
  python -m splat_compass.scenes "D:\\Spla Capture" --last 10  … 新しい10試合だけ
"""
import argparse
import html
import json
import os
import pathlib
import webbrowser

from . import learning, winprob
from .area import episodes as area_episodes
from .objective import push_episodes
from .storage import Store

LONG_DAKAI = 30.0  # これより長くかかった打開は見返す候補にする（秒）
QUICK_DEATH = 8.0  # 復活してからこの秒数以内に倒されたら候補にする


def _mmss(sec):
    sec = max(0, int(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def find_scenes(m, model=None):
    """[(録画の中の開始秒, 終了秒, 種類, 説明)]"""
    s0 = m["video_start"]
    out = []
    me = m.get("me_slot")
    if model is not None and m.get("rule") == "ガチエリア" and m.get("timeline"):
        for rel, what, d in learning.swings(model, m, top=3):
            t = s0 + rel
            out.append((t - 10, t + 6, "勝率が大きく動いた", f"{what}（勝率 {d * 100:+.0f}%）"))
    if m.get("rule") == "ガチエリア":
        for e in area_episodes(m):
            if not e["success"] or e["seconds"] >= LONG_DAKAI:
                state = "失敗" if not e["success"] else f"{e['seconds']:.0f}秒かかった"
                out.append((e["start"] - 3, min(e["end"], e["start"] + 45) + 2, "打開",
                            f"エリアを取られてから打開{state}（取られたカウント {e['count_lost']}）"))
    if m.get("rule") in ("ガチヤグラ", "ガチホコ") and m.get("objective"):
        for e in push_episodes(m):
            if not e["success"] or e["seconds"] >= LONG_DAKAI:
                state = "中央まで戻せなかった" if not e["success"] else f"中央まで{e['seconds']:.0f}秒かかった"
                out.append((e["start"] - 3, min(e["end"], e["start"] + 45) + 2, "打開",
                            f"押し込まれてから{state}（最深 {-e['deepest'] * 100:.0f}%）"))
    if me:
        last_respawn = None
        for t, rel, slot, what, *_ in m["events"]:
            if slot != me:
                continue
            if what == "復活":
                last_respawn = t
            elif "倒された" in what:
                if "持ったまま" in what:
                    out.append((t - 10, t + 3, "スペシャル", "スペシャルを持ったまま倒された"))
                elif last_respawn is not None and t - last_respawn <= QUICK_DEATH:
                    out.append((last_respawn - 1, t + 3, "復活直後", f"復活して{t - last_respawn:.0f}秒で倒された"))
    return sorted((max(0, a), b, k, d) for a, b, k, d in out)


PAGE = """<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8"><title>見返す場面</title>
<style>
body {{ font-family: "Meiryo", sans-serif; margin: 0; background: #f4f4f4; color: #222; }}
#player {{ position: sticky; top: 0; background: #111; text-align: center; padding: 8px; z-index: 1; }}
video {{ max-width: 100%; max-height: 52vh; }}
#now {{ color: #ddd; font-size: 14px; margin-top: 4px; }}
main {{ padding: 12px 16px; max-width: 980px; margin: 0 auto; }}
.filters label {{ margin-right: 12px; }}
section {{ background: #fff; border-radius: 6px; padding: 10px 14px; margin: 12px 0; }}
h2 {{ font-size: 16px; margin: 0 0 6px; }}
.res-勝ち {{ color: #1a7f37; }} .res-負け {{ color: #c0392b; }}
.scene {{ display: flex; align-items: center; gap: 10px; padding: 4px 0; border-top: 1px solid #eee; }}
.scene button {{ padding: 2px 10px; cursor: pointer; }}
.kind {{ font-size: 12px; background: #eee; border-radius: 4px; padding: 1px 6px; white-space: nowrap; }}
.time {{ font-variant-numeric: tabular-nums; color: #666; width: 44px; }}
</style></head><body>
<div id="player"><video id="v" controls preload="none"></video><div id="now">場面の「再生」を押すと、ここで再生します</div></div>
<main>
<p>{summary}</p>
<div class="filters">{filters}</div>
{sections}
</main>
<script>
const v = document.getElementById("v");
let stopAt = null;
function play(src, a, b, label) {{
  if (v.getAttribute("src") !== src) {{ v.setAttribute("src", src); }}
  const go = () => {{ v.currentTime = a; v.play(); }};
  if (v.readyState >= 1) go(); else v.addEventListener("loadedmetadata", go, {{ once: true }});
  stopAt = b;
  document.getElementById("now").textContent = label;
  window.scrollTo({{ top: 0, behavior: "smooth" }});
}}
v.addEventListener("timeupdate", () => {{ if (stopAt !== null && v.currentTime >= stopAt) {{ v.pause(); stopAt = null; }} }});
document.querySelectorAll(".filters input").forEach(cb => cb.addEventListener("change", () => {{
  document.querySelectorAll(".scene[data-kind='" + cb.value + "']").forEach(el => el.style.display = cb.checked ? "" : "none");
}}));
</script></body></html>
"""


def file_link(path):
    return pathlib.Path(path).resolve().as_uri()


def scene_sections(store, video_dir, last=None, link=file_link):
    """場面の一覧の部品: (試合ごとの HTML の並び, 種類の並び, 場面の数)。
    link: ファイルの場所から、ページに書く URL を作る関数（アプリの画面では http の URL にする）。"""
    model = winprob.load_model(store)
    ms = [m for m in store.matches() if m.get("video_end") and not m.get("interrupted")]
    ms = [m for m in ms if os.path.exists(os.path.join(video_dir, m["source"]))]
    ms.sort(key=lambda m: m.get("played_at", ""), reverse=True)
    if last:
        ms = ms[:last]
    kinds, sections, total = [], [], 0
    for m in ms:
        sc = find_scenes(m, model)
        if not sc:
            continue
        src = link(os.path.join(video_dir, m["source"]))
        rows = []
        for a, b, kind, desc in sc:
            if kind not in kinds:
                kinds.append(kind)
            label = f"{m.get('played_at', '')[:16].replace('T', ' ')} {m.get('rule') or ''} {_mmss(a - m['video_start'])} {desc}"
            rows.append(
                f'<div class="scene" data-kind="{html.escape(kind)}">'
                f'<button onclick="{html.escape(f"play({json.dumps(src)}, {a:.1f}, {b:.1f}, {json.dumps(label, ensure_ascii=False)})", quote=True)}">再生</button>'
                f'<span class="time">{_mmss(a - m["video_start"])}</span>'
                f'<span class="kind">{html.escape(kind)}</span><span>{html.escape(desc)}</span></div>')
        shots = m.get("map_shots") or []
        if shots:
            imgs = "".join(
                f'<a href="{link(os.path.join(store.root, "maps", x["image"]))}" target="_blank">'
                f'<img src="{link(os.path.join(store.root, "maps", x["image"]))}" '
                f'title="{_mmss(x["rel"])}" height="72" style="margin:2px"></a>' for x in shots)
            rows.append(f'<div class="maps">マップ画面: {imgs}</div>')
        total += len(rows)
        res = m.get("result") or ""
        sections.append(
            f'<section><h2>{html.escape(m.get("played_at", "")[:16].replace("T", " "))}　{html.escape(m.get("rule") or "ルール不明")}　'
            f'<span class="res-{html.escape(res)}">{html.escape(res)}</span></h2>{"".join(rows)}</section>')
    return sections, kinds, total


def filters_html(kinds):
    return "".join(f'<label><input type="checkbox" value="{html.escape(k)}" checked> {html.escape(k)}</label>' for k in kinds)


def build(store, video_dir, last=None):
    sections, kinds, total = scene_sections(store, video_dir, last)
    page = PAGE.format(summary=f"{len(sections)}試合・{total}場面", filters=filters_html(kinds), sections="".join(sections))
    path = os.path.join(store.root, "scenes.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", help="録画のフォルダ")
    ap.add_argument("--last", type=int, help="新しい順にこの試合数だけ")
    ap.add_argument("--data", help="記録の保存先")
    ap.add_argument("--no-open", action="store_true", help="ブラウザで開かない")
    a = ap.parse_args()
    path, n = build(Store(a.data), a.videos, a.last)
    print(f"{n}場面の一覧を作りました: {path}")
    if not a.no_open:
        webbrowser.open(pathlib.Path(path).resolve().as_uri())


if __name__ == "__main__":
    main()
