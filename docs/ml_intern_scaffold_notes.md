# ML Intern Scaffold Notes

Reference: `https://github.com/huggingface/ml-intern` at commit `7b561e3d94f7de615404e30173609f7f42ca0fc8`.

## What We Kept

- The overall shape of an async agent runtime.
- A tool router with built-in tools and optional MCP tools.
- Local command and file tools with read-before-write guardrails.
- A plan tool for long-running tasks.
- Approval checks for risky operations.
- CLI support for interactive and headless operation.
- Session events as a clean boundary for future UI or notification layers.

## What We Removed

- Hugging Face trace upload and telemetry defaults.
- HF Jobs and HF repo management.
- HF docs, papers, dataset, and GitHub search tools as built-ins.
- The web backend and frontend.
- Fine-tuning/SFT utilities.

## Why

The autointerp bottleneck is not generic ML infrastructure. It is robust, efficient interpretability work: targeted prompt generation, activation caching, causal localization, SAE feature triage, and validation. The runtime should stay small enough that researchers can modify it quickly before the workshop deadline.
