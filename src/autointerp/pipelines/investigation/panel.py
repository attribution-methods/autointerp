"""Human-interpretable results panel for a Q1 affective-vocabulary run.

This is a *Q1-tuned* viewer: it knows the schema of the
``behavioral_affective_vocab_per_category_breakdown`` finding and the
``c1``/``c2`` pre-registered criteria, and lays them out for maximum
legibility. It degrades gracefully (renders what is present, never raises)
so it is safe to call from the pipeline-finalize hook.

Two surfaces, one renderer:

* ``write_snapshot(run_dir)`` — a self-contained ``results_panel.html``
  (Plotly JS inlined; opens in any browser, no server). This is the
  artifact the pipeline auto-emits at terminal state.
* ``serve(run_dir, ...)`` — a FastAPI app that rebuilds the same HTML on
  every request, plus ``/raw/<artifact>`` links to the underlying JSON.
  This is what ``autointerp runs panel <run_id>`` launches.

Heavy deps (fastapi/uvicorn/plotly) are imported lazily and only the
serving path needs the web stack; ``write_snapshot`` needs only plotly.
Install with ``pip install 'autointerp[panel]'``.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

SNAPSHOT_NAME = "results_panel.html"
_AFFECT_BREAKDOWN = (
    "findings/stage_0_black_box/"
    "behavioral_affective_vocab_per_category_breakdown.json"
)
_RAW_ALLOWED = {
    "report.json": "report.json",
    "spec.json": "spec.json",
    "state.json": "state.json",
    "breakdown.json": _AFFECT_BREAKDOWN,
}


# --------------------------------------------------------------------------
# Loading (defensive: every field optional, never raises)
# --------------------------------------------------------------------------


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_panel_data(run_dir: Path) -> dict[str, Any]:
    """Collect everything the page needs. Missing files -> ``None`` slots."""
    run_dir = Path(run_dir)
    report = _read_json(run_dir / "report.json") or {}
    state = _read_json(run_dir / "state.json") or {}
    spec = _read_json(run_dir / "spec.json") or {}
    breakdown = _read_json(run_dir / _AFFECT_BREAKDOWN) or {}
    meta = report.get("metadata", {}) if isinstance(report, dict) else {}
    return {
        "run_dir": str(run_dir),
        "run_id": report.get("report_id") or state.get("run_id") or run_dir.name,
        "spec_id": meta.get("spec_id") or spec.get("spec_id"),
        "spec_revision": meta.get("spec_revision") or spec.get("revision"),
        "terminal_state": meta.get("terminal_state") or state.get("terminal_state"),
        "question": spec.get("question"),
        "hypothesis": spec.get("hypothesis"),
        "behavior": (report.get("behavior") or {}),
        "criteria": meta.get("criteria_evaluated", {}) or {},
        "claims": report.get("claims", []) or [],
        "limitations": report.get("limitations", []) or [],
        "budget": meta.get("budget_consumed", {}) or {},
        "breakdown": breakdown.get("metadata", {}) or {},
        "breakdown_summary": breakdown.get("summary"),
        "started": state.get("run_started_at"),
        "ended": state.get("run_ended_at"),
    }


# --------------------------------------------------------------------------
# Figures (lazy plotly)
# --------------------------------------------------------------------------


def _require_plotly():
    try:
        import plotly.graph_objects as go  # noqa: F401
        import plotly.io as pio  # noqa: F401
        import plotly.offline as poff  # noqa: F401

        return go, pio, poff
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "results panel needs plotly — install with "
            "`pip install 'autointerp[panel]'`"
        ) from exc


def _fig_criterion(go, cid: str, c: dict) -> Any:
    """A gauge for one pre-registered criterion: value vs threshold."""
    value = c.get("value")
    thr = c.get("threshold")
    comp = c.get("comparator", ">=")
    passed = bool(c.get("passed"))
    metric = c.get("metric", "metric")
    vnum = value if isinstance(value, (int, float)) else 0.0
    tnum = thr if isinstance(thr, (int, float)) else 0.0
    hi = max(vnum, tnum) * 1.5 or 1.0
    green, red = "#2f9e44", "#e03131"
    if comp in (">=", ">"):
        steps = [
            {"range": [0, tnum], "color": "#ffe3e3"},
            {"range": [tnum, hi], "color": "#d3f9d8"},
        ]
    else:
        steps = [
            {"range": [0, tnum], "color": "#d3f9d8"},
            {"range": [tnum, hi], "color": "#ffe3e3"},
        ]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=vnum,
            number={"valueformat": ".4g"},
            title={
                "text": f"<b>{cid}</b><br><span style='font-size:0.8em'>"
                f"{metric} {comp} {thr} — "
                f"{'PASS' if passed else 'FAIL'}</span>"
            },
            gauge={
                "axis": {"range": [0, hi]},
                "bar": {"color": green if passed else red},
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.85,
                    "value": tnum,
                },
                "steps": steps,
            },
        )
    )
    fig.update_layout(height=240, margin=dict(l=30, r=30, t=70, b=10))
    return fig


def _fig_per_category(go, breakdown: dict) -> Any | None:
    rates = breakdown.get("per_category_rates")
    if not isinstance(rates, dict) or not rates:
        return None
    cats = list(rates.keys())
    scale = 10_000  # rates are ~1e-4; show "matches per 10,000 words"
    conc = [(rates[c].get("concerning_mean") or 0.0) * scale for c in cats]
    clean = [(rates[c].get("clean_mean") or 0.0) * scale for c in cats]
    fig = go.Figure(
        [
            go.Bar(
                name="concerning (pre-act window)",
                y=cats,
                x=conc,
                orientation="h",
                marker_color="#d1495b",
                hovertemplate="%{y}: %{x:.3f} /10k words (concerning)<extra></extra>",
            ),
            go.Bar(
                name="clean (full transcript)",
                y=cats,
                x=clean,
                orientation="h",
                marker_color="#3d7eaa",
                hovertemplate="%{y}: %{x:.3f} /10k words (clean)<extra></extra>",
            ),
        ]
    )
    fig.update_layout(
        barmode="group",
        height=360,
        margin=dict(l=110, r=30, t=40, b=40),
        xaxis_title="affective-word matches per 10,000 words",
        legend=dict(orientation="h", y=1.12, x=0),
        title="Per-category rate — concerning pre-act vs. clean",
    )
    return fig


# --------------------------------------------------------------------------
# Caveat synthesis (the interpretability core — honest, from artifacts)
# --------------------------------------------------------------------------


def _caveats(data: dict) -> list[str]:
    out: list[str] = []
    bd = data["breakdown"]
    rates = bd.get("per_category_rates") or {}
    if rates:
        nonzero = [
            c
            for c, v in rates.items()
            if abs(v.get("concerning_mean") or 0.0) > 1e-9
            or abs(v.get("clean_mean") or 0.0) > 1e-9
        ]
        driver = [
            c
            for c, v in rates.items()
            if (v.get("difference") or 0.0) > 1e-7
        ]
        if len(driver) == 1:
            out.append(
                f"Signal is <b>one category</b> ({driver[0]}). "
                f"{len(rates) - len(nonzero)} of {len(rates)} affect "
                f"categories are ~zero in both groups — the headline effect "
                f"is not a broad affective shift."
            )
    n_c = bd.get("n_concerning")
    n_k = bd.get("n_clean")
    if n_c is not None:
        out.append(
            f"<b>n_concerning = {n_c}</b> (vs {n_k} clean). Small-n: "
            f"per-transcript rate estimates are fragile."
        )
    for cid, c in data["criteria"].items():
        if c.get("metric") in ("custom",) or "pval" in cid or "pvalue" in cid:
            v = c.get("value")
            thr = c.get("threshold")
            if isinstance(v, (int, float)) and isinstance(thr, (int, float)):
                if c.get("passed") and v > 0.05 and thr >= 0.05:
                    out.append(
                        f"Permutation p = {v:.4g} clears the lenient "
                        f"{thr} threshold but would <b>fail the "
                        f"conventional 0.05</b>."
                    )
    out.append(
        "Per-transcript concentration (the effect riding on 2 of 5 "
        "concerning transcripts) is documented in the writeup; it is not "
        "recomputed here as the sealed artifacts carry only group means."
    )
    return out


# --------------------------------------------------------------------------
# HTML assembly
# --------------------------------------------------------------------------


_CSS = """
:root{--bg:#0f1419;--card:#1a2129;--ink:#e6e6e6;--mut:#8b97a3;
--ok:#2f9e44;--bad:#e03131;--warn:#f08c00;--line:#2a323c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:28px}
h1{font-size:22px;margin:0 0 4px}.sub{color:var(--mut);font-size:13px}
.banner{margin:20px 0;padding:16px 20px;border-radius:10px;font-weight:600;
font-size:17px;border:1px solid var(--line)}
.banner.ok{background:rgba(47,158,68,.14);border-color:var(--ok)}
.banner.bad{background:rgba(224,49,49,.14);border-color:var(--bad)}
.note{font-weight:400;font-size:13px;color:var(--mut);margin-top:6px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:18px;margin:16px 0}
.card h2{font-size:14px;text-transform:uppercase;letter-spacing:.06em;
color:var(--mut);margin:0 0 12px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.callout{background:rgba(240,140,0,.10);border:1px solid var(--warn);
border-radius:10px;padding:16px 20px;margin:16px 0}
.callout h2{color:var(--warn)}.callout ul{margin:8px 0 0;padding-left:20px}
.callout li{margin:6px 0}
.kv{display:grid;grid-template-columns:160px 1fr;gap:6px 14px;font-size:14px}
.kv .k{color:var(--mut)}
.claim{padding:6px 0;border-bottom:1px solid var(--line);font-size:14px}
a{color:#74b9ff}.foot{color:var(--mut);font-size:12px;margin-top:24px}
@media(max-width:760px){.grid{grid-template-columns:1fr}}
"""


def _esc(x: Any) -> str:
    return html.escape(str(x)) if x is not None else "—"


def build_html(run_dir: Path) -> str:
    """Assemble the full self-contained results page (Plotly inlined)."""
    go, pio, poff = _require_plotly()
    data = load_panel_data(run_dir)

    crits = data["criteria"]
    n_pass = sum(1 for c in crits.values() if c.get("passed"))
    n_tot = len(crits)
    all_pass = n_tot > 0 and n_pass == n_tot
    term = data["terminal_state"]

    def fig_div(fig, div_id):
        return pio.to_html(
            fig, include_plotlyjs=False, full_html=False, div_id=div_id
        )

    crit_divs = "".join(
        f"<div>{fig_div(_fig_criterion(go, cid, c), f'c_{i}')}</div>"
        for i, (cid, c) in enumerate(crits.items())
    )
    cat_fig = _fig_per_category(go, data["breakdown"])
    cat_block = (
        fig_div(cat_fig, "cats")
        if cat_fig is not None
        else "<p class='note'>per-category breakdown artifact not found</p>"
    )

    if all_pass:
        banner_cls, verdict = "ok", (
            f"All {n_tot} pre-registered criteria PASSED"
        )
    elif n_tot:
        banner_cls, verdict = "bad", (
            f"{n_pass}/{n_tot} pre-registered criteria passed"
        )
    else:
        banner_cls, verdict = "bad", "No evaluated criteria found"
    term_note = ""
    if term and term != "completed" and all_pass:
        term_note = (
            f"<div class='note'>Run terminal_state = "
            f"<code>{_esc(term)}</code> (pipeline-level, unrelated). "
            f"Criterion verdicts below were sealed before that.</div>"
        )

    caveats = "".join(f"<li>{c}</li>" for c in _caveats(data))
    claims = "".join(f"<div class='claim'>{_esc(c)}</div>" for c in data["claims"])
    lims = "".join(f"<div class='claim'>{_esc(l)}</div>" for l in data["limitations"])
    bsum = data.get("breakdown_summary")
    bsum_html = (
        f"<p style='margin:0 0 12px;color:var(--mut);font-size:14px'>"
        f"{_esc(bsum)}</p>"
        if bsum
        else ""
    )
    bud = data["budget"]
    bud_html = (
        f"tool_calls {bud.get('tool_calls','—')} · "
        f"wallclock {bud.get('wallclock_seconds','—')}s · "
        f"samples {bud.get('samples','—')}"
    )

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Q1 results · {_esc(data['run_id'])}</title>
<style>{_CSS}</style>
<script>{poff.get_plotlyjs()}</script></head><body><div class="wrap">
<h1>Q1 — Agentic Misalignment Affective Vocabulary</h1>
<div class="sub">{_esc(data['run_id'])} &nbsp;·&nbsp; spec
{_esc(data['spec_id'])} rev {_esc(data['spec_revision'])} &nbsp;·&nbsp;
terminal_state <code>{_esc(term)}</code></div>

<div class="banner {banner_cls}">{verdict}{term_note}</div>

<div class="card"><h2>Pre-registered criteria</h2>
<div class="grid">{crit_divs}</div></div>

<div class="card"><h2>The headline — per-category contrast</h2>
{bsum_html}{cat_block}</div>

<div class="callout"><h2>⚠ How to read this (caveats)</h2>
<ul>{caveats}</ul></div>

<div class="grid">
<div class="card"><h2>Question &amp; hypothesis</h2>
<div class="kv">
<div class="k">question</div><div>{_esc(data['question'])}</div>
<div class="k">hypothesis</div><div>{_esc(data['hypothesis'])}</div>
<div class="k">behavior</div><div>{_esc(data['behavior'].get('behavior_id'))}</div>
</div></div>
<div class="card"><h2>Claims &amp; limitations</h2>
{claims or "<div class='note'>none</div>"}
{lims}</div>
</div>

<div class="card"><h2>Run</h2>
<div class="kv">
<div class="k">started</div><div>{_esc(data['started'])}</div>
<div class="k">ended</div><div>{_esc(data['ended'])}</div>
<div class="k">budget</div><div>{bud_html}</div>
<div class="k">raw artifacts</div><div>
<a href="/raw/report.json">report.json</a> ·
<a href="/raw/spec.json">spec.json</a> ·
<a href="/raw/breakdown.json">per-category breakdown</a> ·
<a href="/raw/state.json">state.json</a></div>
</div></div>

<div class="foot">Generated by <code>autointerp runs panel</code> ·
run dir <code>{_esc(data['run_dir'])}</code></div>
</div></body></html>"""


def write_snapshot(run_dir: Path, out_path: Path | None = None) -> Path:
    """Write a self-contained ``results_panel.html`` into the run dir."""
    run_dir = Path(run_dir)
    out_path = Path(out_path) if out_path else run_dir / SNAPSHOT_NAME
    out_path.write_text(build_html(run_dir))
    return out_path


# --------------------------------------------------------------------------
# Live server
# --------------------------------------------------------------------------


def create_app(run_dir: Path):
    """A FastAPI app that re-renders the panel on each request."""
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "results panel server needs fastapi/uvicorn — install with "
            "`pip install 'autointerp[panel]'`"
        ) from exc

    run_dir = Path(run_dir)
    app = FastAPI(title="autointerp results panel", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return build_html(run_dir)

    @app.get("/healthz", response_class=PlainTextResponse)
    def healthz() -> str:
        return "ok"

    @app.get("/raw/{name}")
    def raw(name: str):
        rel = _RAW_ALLOWED.get(name)
        if rel is None:
            return JSONResponse({"error": "unknown artifact"}, status_code=404)
        payload = _read_json(run_dir / rel)
        if payload is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(payload)

    return app


def serve(
    run_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8080,
    open_browser: bool = True,
) -> None:
    """Launch the panel server (blocking)."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "results panel server needs uvicorn — install with "
            "`pip install 'autointerp[panel]'`"
        ) from exc
    app = create_app(run_dir)
    url = f"http://{host}:{port}/"
    print(f"\n  Q1 results panel → {url}\n  (Ctrl-C to stop)\n")
    if open_browser:
        try:
            import threading
            import webbrowser

            threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        except Exception:
            pass
    uvicorn.run(app, host=host, port=port, log_level="warning")


__all__ = [
    "SNAPSHOT_NAME",
    "load_panel_data",
    "build_html",
    "write_snapshot",
    "create_app",
    "serve",
]
