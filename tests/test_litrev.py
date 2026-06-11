"""Tests for the /litrev side-channel literature review (no network)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from autointerp_agent import litrev

_ATOM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v2</id>
    <title>Surprise  Directions in
      Language Models</title>
    <summary>We study surprise. We find a direction. It is causal. Steering
      works. Bonus sentence that should be trimmed away by the summary.</summary>
    <published>2024-01-05T00:00:00Z</published>
    <author><name>A. One</name></author>
    <author><name>B. Two</name></author>
    <author><name>C. Three</name></author>
    <author><name>D. Four</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2402.00002v1</id>
    <title>Another Paper</title>
    <summary>Single sentence abstract.</summary>
    <published>2024-02-01T00:00:00Z</published>
    <author><name>E. Five</name></author>
  </entry>
</feed>
"""


def _paper(arxiv_id: str, title: str = "T") -> litrev.Paper:
    return litrev.Paper(
        arxiv_id=arxiv_id, title=title, authors=["A"], published="2024-01-01",
        url=f"https://arxiv.org/abs/{arxiv_id}", abstract="One. Two. Three. Four. Five.",
    )


def test_parse_atom_extracts_fields() -> None:
    papers = litrev._parse_atom(_ATOM_XML)
    assert len(papers) == 2
    first = papers[0]
    assert first.arxiv_id == "2401.00001v2"
    assert first.title == "Surprise Directions in Language Models"  # whitespace folded
    assert first.url == "https://arxiv.org/abs/2401.00001"  # version stripped
    assert first.year == "2024"
    assert first.authors == ["A. One", "B. Two", "C. Three", "D. Four"]


def test_summarize_abstract_trims_sentences() -> None:
    abstract = "One. Two! Three? Four. Five. Six."
    out = litrev.summarize_abstract(abstract, max_sentences=4)
    assert out == "One. Two! Three? Four."
    assert litrev.summarize_abstract("Short.") == "Short."


def test_search_arxiv_interleaves_dedupes_caps(monkeypatch) -> None:
    lists = {
        "q1": [_paper("1"), _paper("2"), _paper("3")],
        "q2": [_paper("2v3"), _paper("4"), _paper("5")],  # 2v3 dedupes against 2
        "q3": [_paper(str(i)) for i in range(1, 12)],
    }
    monkeypatch.setattr(litrev, "_fetch_arxiv", lambda q, max_results=5: lists[q])
    merged = asyncio.run(litrev.search_arxiv(["q1", "q2", "q3"]))
    ids = [p.arxiv_id.split("v")[0] for p in merged]
    assert ids[0] == "1"  # rank-0 of q1 first
    assert len(ids) == len(set(ids))  # deduped
    assert len(ids) <= litrev._MAX_PAPERS


def test_search_arxiv_tolerates_failed_query(monkeypatch) -> None:
    def fetch(query, max_results=5):
        if query == "bad":
            raise RuntimeError("arxiv down")
        return [_paper("7")]

    monkeypatch.setattr(litrev, "_fetch_arxiv", fetch)
    merged = asyncio.run(litrev.search_arxiv(["bad", "good"]))
    assert [p.arxiv_id for p in merged] == ["7"]


def test_generate_queries_parses_lines_and_falls_back(monkeypatch) -> None:
    import autointerp_agent.agent_loop as al

    async def fake_ac(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content="surprise representation language models\n"
                        "- expectation violation probing\n"
                        "surprisal direction steering\n"
            ))]
        )

    monkeypatch.setattr(al, "acompletion", fake_ac)
    queries = asyncio.run(litrev.generate_queries("how surprise?", "m"))
    assert queries == [
        "surprise representation language models",
        "expectation violation probing",
        "surprisal direction steering",
    ]

    async def boom(**kwargs):
        raise RuntimeError("no llm")

    monkeypatch.setattr(al, "acompletion", boom)
    assert asyncio.run(litrev.generate_queries("how surprise?", "m")) == ["how surprise?"]


def test_write_markdown(tmp_path: Path) -> None:
    path = litrev.write_markdown(
        "Surprise in LMs?", ["q1"], [_paper("2401.1", "Nice Paper")], out_dir=tmp_path
    )
    text = path.read_text()
    assert "# Literature review — Surprise in LMs?" in text
    assert "[Nice Paper](https://arxiv.org/abs/2401.1)" in text
    assert path.name.startswith("surprise-in-lms")


def test_arxiv_query_ands_terms_and_constrains_categories() -> None:
    q = litrev._arxiv_query("how do models represent surprise in text")
    # Stopwords dropped, content terms ANDed (OR-ish parsing of unquoted
    # multi-word queries is what surfaced LHC papers for an NLP question).
    assert "all:models AND all:represent AND all:surprise AND all:text" in q
    assert "how" not in q.split(") AND ")[0]
    assert "cat:cs.CL" in q and q.startswith("(")


def test_fetch_arxiv_parses_response(monkeypatch) -> None:
    import httpx

    class _Resp:
        text = _ATOM_XML

        def raise_for_status(self) -> None:
            pass

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp())
    papers = litrev._fetch_arxiv("anything")
    assert [p.arxiv_id for p in papers] == ["2401.00001v2", "2402.00002v1"]


def test_draft_spec_question(tmp_path: Path) -> None:
    assert litrev.draft_spec_question(tmp_path) is None
    (tmp_path / "_draft.json").write_text('{"question": "  How does X? "}')
    assert litrev.draft_spec_question(tmp_path) == "How does X?"
    (tmp_path / "_draft.json").write_text("{not json")
    assert litrev.draft_spec_question(tmp_path) is None
