"""Side-channel literature review for the interactive shell (``/litrev``).

Turns a research question into arXiv search queries (one small LLM call),
fetches matching papers from the free arXiv API, and renders a numbered
panel of linked titles with 3–4 sentence summaries (the opening of each
paper's own abstract — faithful and free). Results are also written to
``outputs/litrev/`` as markdown.

Deliberately isolated from the agent: nothing here touches the
conversation context. The user reads the panel; if they want a paper in
the investigation, they bring it up themselves.
"""

from __future__ import annotations

import asyncio
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ARXIV_API = "https://export.arxiv.org/api/query"
_ATOM = "{http://www.w3.org/2005/Atom}"
_MAX_PAPERS = 8
_PER_QUERY = 5
_SUMMARY_SENTENCES = 4

LITREV_DIR = Path("outputs/litrev")


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: list[str]
    published: str  # ISO date string
    url: str
    abstract: str

    @property
    def year(self) -> str:
        return self.published[:4] if self.published else "?"


# ---------------------------------------------------------------------------
# query generation (the one LLM call)
# ---------------------------------------------------------------------------


async def generate_queries(topic: str, model_name: str, cost_tracker: Any = None) -> list[str]:
    """Turn a research question into 2–3 arXiv search strings.

    Questions are poor search queries ("How do models represent surprise?");
    keyword phrases retrieve far better. Falls back to the raw topic on any
    LLM failure so /litrev still works offline-ish.
    """
    from .agent_loop import _call_llm  # lazy litellm + warm-cache reuse

    prompt = (
        "Generate 3 arXiv search queries (keyword phrases, not questions) to "
        "find research literature about the topic below. One query per line, "
        "no numbering, no quotes, 3-7 words each, favor mechanistic "
        "interpretability / NLP terminology where apt.\n\n"
        f"Topic: {topic.strip()}"
    )
    try:
        response = await _call_llm(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            # Generous: reasoning models spend hidden tokens from this budget
            # before emitting any text; 300 was sometimes all-reasoning.
            max_tokens=2000,
        )
        if cost_tracker is not None:
            try:
                cost_tracker.add_response(response)
            except Exception:  # noqa: BLE001 — accounting must not break search
                pass
        content = str(response.choices[0].message.content or "")
        queries = [
            line.strip(" -*\"'")
            for line in content.splitlines()
            if line.strip()
            and len(line.split()) <= 10
            and ":" not in line  # drops "Here are three queries:" preambles
        ]
        return queries[:3] or [topic]
    except Exception:  # noqa: BLE001 — degrade to the raw topic
        return [topic]


# ---------------------------------------------------------------------------
# arXiv API
# ---------------------------------------------------------------------------


def _parse_atom(xml_text: str) -> list[Paper]:
    root = ET.fromstring(xml_text)
    papers: list[Paper] = []
    for entry in root.findall(f"{_ATOM}entry"):
        raw_id = (entry.findtext(f"{_ATOM}id") or "").strip()
        arxiv_id = raw_id.rsplit("/abs/", 1)[-1]
        title = " ".join((entry.findtext(f"{_ATOM}title") or "").split())
        abstract = " ".join((entry.findtext(f"{_ATOM}summary") or "").split())
        published = (entry.findtext(f"{_ATOM}published") or "").strip()
        authors = [
            (a.findtext(f"{_ATOM}name") or "").strip()
            for a in entry.findall(f"{_ATOM}author")
        ]
        if not (arxiv_id and title):
            continue
        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                title=title,
                authors=[a for a in authors if a],
                published=published,
                url=f"https://arxiv.org/abs/{arxiv_id.split('v')[0]}",
                abstract=abstract,
            )
        )
    return papers


# Unquoted multi-word `all:` queries get OR-ish parsing from the arXiv API
# (relevance then surfaces giant collaboration papers from hep-ex et al.).
# AND the terms and constrain to the categories this tool serves.
_STOPWORDS = frozenset(
    "a an the of in on for to with and or how do does is are what".split()
)
_CATEGORIES = ("cs.CL", "cs.LG", "cs.AI", "stat.ML", "q-bio.NC")


def _arxiv_query(query: str) -> str:
    terms = [
        t for t in re.findall(r"[A-Za-z0-9-]+", query.lower())
        if t not in _STOPWORDS
    ][:6]
    anded = " AND ".join(f"all:{t}" for t in terms) or f"all:{query}"
    cats = " OR ".join(f"cat:{c}" for c in _CATEGORIES)
    return f"({anded}) AND ({cats})"


def _fetch_arxiv(query: str, max_results: int = _PER_QUERY) -> list[Paper]:
    import httpx

    resp = httpx.get(
        _ARXIV_API,
        params={
            "search_query": _arxiv_query(query),
            "max_results": max_results,
            "sortBy": "relevance",
        },
        timeout=15.0,
        follow_redirects=True,
    )
    resp.raise_for_status()
    return _parse_atom(resp.text)


async def search_arxiv(queries: list[str]) -> list[Paper]:
    """Run all queries, interleave by rank, dedupe by arXiv id, cap."""
    per_query = await asyncio.gather(
        *(asyncio.to_thread(_fetch_arxiv, q) for q in queries),
        return_exceptions=True,
    )
    result_lists = [r for r in per_query if isinstance(r, list)]
    seen: set[str] = set()
    merged: list[Paper] = []
    for rank in range(max((len(r) for r in result_lists), default=0)):
        for results in result_lists:
            if rank < len(results):
                paper = results[rank]
                key = paper.arxiv_id.split("v")[0]
                if key not in seen:
                    seen.add(key)
                    merged.append(paper)
    return merged[:_MAX_PAPERS]


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def summarize_abstract(abstract: str, max_sentences: int = _SUMMARY_SENTENCES) -> str:
    """First few sentences of the abstract — what the paper is doing,
    in the authors' own words (no LLM, no hallucination, no cost)."""
    sentences = _SENTENCE_END.split(abstract.strip())
    return " ".join(sentences[:max_sentences]).strip()


def _authors_line(paper: Paper) -> str:
    names = paper.authors
    if len(names) > 3:
        names = names[:3] + ["et al."]
    return ", ".join(names)


def render_panel(topic: str, papers: list[Paper]) -> Any:
    from rich import box as rich_box
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text

    blocks: list[Text] = []
    for i, paper in enumerate(papers, 1):
        head = Text()
        head.append(f"{i}. ", style="bold cyan")
        head.append(paper.title, style=f"bold link {paper.url}")
        meta = Text(
            f"   {_authors_line(paper)} · {paper.year} · {paper.url}",
            style="bright_black",
        )
        body = Text(f"   {summarize_abstract(paper.abstract)}", style="dim")
        blocks.extend([head, meta, body, Text("")])
    if blocks:
        blocks.pop()  # trailing spacer
    else:
        blocks = [Text("No papers found — try /litrev <more specific topic>.")]
    return Panel(
        Group(*blocks),
        box=rich_box.ROUNDED,
        border_style="magenta",
        title=f"literature · {topic[:60]}",
        title_align="left",
        subtitle="[dim]side panel — not in the agent's context[/dim]",
        subtitle_align="right",
        padding=(0, 1),
    )


def write_markdown(topic: str, queries: list[str], papers: list[Paper],
                   out_dir: Path | None = None) -> Path:
    out = out_dir or LITREV_DIR
    out.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:48] or "litrev"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    path = out / f"{slug}-{stamp}.md"
    lines = [
        f"# Literature review — {topic}",
        "",
        f"_Generated {stamp}Z · queries: {json.dumps(queries)}_",
        "",
    ]
    for i, paper in enumerate(papers, 1):
        lines += [
            f"## {i}. [{paper.title}]({paper.url})",
            f"*{_authors_line(paper)} · {paper.year} · `{paper.arxiv_id}`*",
            "",
            summarize_abstract(paper.abstract),
            "",
        ]
    path.write_text("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# orchestration (called by the /litrev dispatch in repl.py)
# ---------------------------------------------------------------------------


async def run_litrev(topic: str, *, console: Any, model_name: str,
                     cost_tracker: Any = None, out_dir: Path | None = None) -> Path | None:
    with console.status(f"[dim]litrev: generating arXiv queries for “{topic[:50]}…”[/dim]"):
        queries = await generate_queries(topic, model_name, cost_tracker)
    with console.status(f"[dim]litrev: searching arXiv ({len(queries)} queries)…[/dim]"):
        papers = await search_arxiv(queries)
    console.print(f"[dim]queries: {' · '.join(queries)}[/dim]")
    console.print(render_panel(topic, papers))
    if not papers:
        return None
    path = write_markdown(topic, queries, papers, out_dir)
    console.print(f"[dim]saved to {path}[/dim]")
    return path


def draft_spec_question(spec_dir: Path | None = None) -> str | None:
    """The in-progress draft's research question, if a draft exists."""
    draft = (spec_dir or Path("outputs/specs")) / "_draft.json"
    if not draft.exists():
        return None
    try:
        question = json.loads(draft.read_text()).get("question")
    except (json.JSONDecodeError, OSError):
        return None
    return str(question).strip() if question else None


__all__ = [
    "Paper",
    "draft_spec_question",
    "generate_queries",
    "render_panel",
    "run_litrev",
    "search_arxiv",
    "summarize_abstract",
    "write_markdown",
]
