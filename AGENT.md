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

## Skill Selection

Use `black-box-auditing` for API-only investigation, `activation-cache` before any repeated white-box analysis, and `causal-validation` before final claims. Use method-specific skills when a concrete question points to that method, such as `logit-lens` for intermediate token predictions or `activation-patching` for localizing a causal site.
