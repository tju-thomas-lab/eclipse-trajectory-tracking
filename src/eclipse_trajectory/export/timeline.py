from __future__ import annotations

import html
from pathlib import Path

from eclipse_trajectory.schemas import ActionRecord


def write_timeline(session_dir: Path, actions: list[ActionRecord]) -> Path:
    cards = "\n".join(_action_card(action) for action in actions)
    if not cards:
        cards = '<p class="empty">No candidate visual-change events crossed the threshold.</p>'
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Eclipse trajectory timeline</title>
<style>
:root{{--bg:#10151d;--card:#19222e;--text:#eef3f8;--muted:#a8b4c2;--accent:#69d0c5;--warn:#ffcf70}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}}
header,main{{max-width:1160px;margin:auto;padding:24px}} header{{border-bottom:1px solid #2c3948}}
.warning{{color:var(--warn)}} .card{{background:var(--card);border:1px solid #2c3948;border-radius:10px;padding:18px;margin:18px 0}}
.meta{{color:var(--muted);font-variant-numeric:tabular-nums}} .frames{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:12px}}
figure{{margin:0}} img{{display:block;width:100%;height:auto;border-radius:5px}} figcaption{{color:var(--muted);margin-top:4px}}
.tag{{color:var(--accent)}} .empty{{color:var(--muted)}} @media(max-width:700px){{.frames{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>Eclipse trajectory timeline</h1>
<p class="warning">Local research artifact—screenshots may contain Protected Health Information.</p>
<p>Deterministic records describe measured state changes only. Unknown actions are intentionally not guessed.</p>
</header><main>{cards}</main></body></html>"""
    destination = session_dir / "timeline.html"
    destination.write_text(document, encoding="utf-8", newline="\n")
    return destination


def _action_card(action: ActionRecord) -> str:
    title = action.low_level_instruction or "Visible state change; operation unknown."
    figures = []
    for frame in action.evidence.frames:
        figures.append(
            f'<figure><img loading="lazy" src="{html.escape(frame.path)}" '
            f'alt="{html.escape(frame.role)} evidence"><figcaption>{html.escape(frame.role)} · '
            f"{frame.actual_timestamp_seconds:.3f}s</figcaption></figure>"
        )
    return (
        f'<article class="card" id="{html.escape(action.action_id)}">'
        f'<div class="meta">{action.start_time_seconds:.3f}s–{action.end_time_seconds:.3f}s · '
        f"{html.escape(action.action_id)}</div><h2>{html.escape(title)}</h2>"
        f'<div class="tag">{html.escape(action.primitive_action.type)}</div>'
        f'<div class="frames">{"".join(figures)}</div></article>'
    )
