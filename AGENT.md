# Autointerp Agent

This repository defines a first-pass autointerp agent as a reusable scaffold:

- `skills/`: procedural knowledge for when and how to use each method.
- `src/autointerp/`: Python implementations for repeatable interpretability operations.
- `src/autointerp/schemas.py`: shared artifacts for behavior specs, prompt batches, findings, candidate sites, and validation reports.
- `src/autointerp_agent/`: an ML Intern-inspired agent runtime.
- `configs/agent.yaml`: a compact agent configuration and default skill set.
- `scripts/validate_skills.py`: local validation for skill frontmatter.

## Design Correction

Skills are not a replacement for tools, APIs, or libraries. Treat them as the layer that tells an agent which method to use, what evidence to expect, and how to avoid common failure modes. Keep deterministic or expensive operations in tools. Keep target-specific model access behind a narrow adapter so the same skills work for white-box models, API-only models, benchmark targets, and downstream integrations.

## Operating Loop

1. State the hypothesis or behavior to explain.
2. Pick the cheapest method that could falsify or sharpen it.
3. Run black-box probes before heavier white-box methods when model internals are unavailable or the behavior is not localized.
4. Cache activations once and reuse them across lenses, probes, SAE lookups, patching, and attribution.
5. Separate discovery from validation. Discovery methods can be noisy; validation requires held-out prompts and causal interventions.
6. Record evidence in an `InvestigationReport` so later agents and downstream repositories can consume it.

## Default System Prompt

You are an automated interpretability agent. Your task is to explain a model behavior, hidden behavior, capability, or failure mode using the available black-box and white-box tools. Prefer simple probes first, then use activation-level methods when they can answer a specific question. Do not treat a visualization or correlation as causal evidence until it survives a targeted intervention. Maintain a short research log with hypotheses, evidence, uncertainty, and next actions.

## Stages of the system

The agent operates in two discrete modes, each with its own CLI and tool
surface. They are bridged by the `InvestigationSpec` artifact:

1. **Stage 0 — conversational spec design.** Build a pre-registered,
   falsifiable `InvestigationSpec` with the user. Tools:
   `describe_spec`/`update_spec`/`show_spec`/`validate_spec`/`finalize_spec`
   and the metric reference tools. Output: an approved spec at
   `outputs/specs/<spec_id>_rev<n>.json`. See [docs/stage0.md](docs/stage0.md).

2. **Investigation pipeline — execute the approved spec.** A coding agent
   walks `spec.stages` in order under a gated tool surface. Free-form
   `bash`/`read_file`/`write_file`/`edit_file` for actually running
   experiments; gated Tier-2 tools (`commit_artifact`, `compute_metric`,
   `evaluate_criterion`, `advance_stage`, `current_stage`,
   `request_spec_revision`, `get_state`, `get_budget`) make
   pre-registration mechanically enforceable. Every run captures a full
   debugging transcript on disk (`assistant_turns.jsonl`,
   `tool_invocations.jsonl` with full output bodies, plus the gated
   `log.jsonl` audit). CLI:
   `python -m autointerp_agent.investigation --spec <path>`. See
   [docs/investigation.md](docs/investigation.md).

## Skill Selection

Use `black-box-auditing` for API-only investigation, `activation-cache` before any repeated white-box analysis, and `causal-validation` before final claims. Use method-specific skills when a concrete question points to that method, such as `logit-lens` for intermediate token predictions or `activation-patching` for localizing a causal site.
