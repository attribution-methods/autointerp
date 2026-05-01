# Notices and Attribution

This project contains original code by the `attribution-methods` authors and an agent runtime scaffold that was designed after inspecting the public Hugging Face `ml-intern` repository.

## Hugging Face ML Intern

Reference repository:

- `https://github.com/huggingface/ml-intern`
- Reference commit inspected during scaffold work: `7b561e3d94f7de615404e30173609f7f42ca0fc8`

The following design ideas were adapted from ML Intern:

- interactive and headless CLI modes;
- an asynchronous agent loop around LiteLLM tool calling;
- a `ToolRouter` abstraction for built-in tools and MCP tools;
- local shell/read/write/edit tools with read-before-write guardrails;
- a planning tool with `pending`, `in_progress`, and `completed` statuses;
- approval policies for potentially risky or expensive operations;
- session events for UI or gateway integrations;
- context management and skill-aware system prompts.

The implementation in `src/autointerp_agent/` is specialized for automated interpretability work and removes Hugging Face-specific product surfaces such as trace uploading, backend/frontend services, HF Jobs, HF repo management, and default HF documentation tools.

If upstream ML Intern later publishes an explicit license or a preferred attribution format, update this notice accordingly.
