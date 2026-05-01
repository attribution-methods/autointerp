# Architecture

`autointerp` has two layers:

1. `autointerp`: domain tools for mechanistic interpretability.
2. `autointerp_agent`: an agent runtime that can select skills, call tools, request approval, and run locally or through MCP-style integrations.

## Runtime Loop

```text
user prompt
  -> ContextManager builds system + skill context
  -> LiteLLM model call with ToolRouter specs
  -> zero or more tool calls
  -> permission check
  -> tool execution
  -> tool output appended to context
  -> repeat until final answer or max iterations
```

## Tool Policy

The initial runtime includes local tools because the near-term bottleneck is engineering:

- `plan`: maintain a visible task plan.
- `list_skills`: inspect available method skills.
- `read_skill`: load a skill's `SKILL.md`.
- `bash`: run local commands with timeout and output truncation.
- `read_file`: read files with line numbers.
- `write_file`: write files after read-before-write checks.
- `edit_file`: exact string replacement after read-before-write checks.
- MCP tools: optional, loaded from config when `fastmcp` is installed.

Potentially destructive or expensive operations require approval unless `auto_approve` is set.

## Domain Layer

The method library includes:

- black-box audit probes;
- model loading and chat formatting;
- activation extraction and component specs;
- contrastive directions;
- activation steering and patching;
- logit lens and direct logit attribution;
- probes and gradient attribution;
- SAE feature post-processing;
- circuit ranking helpers.

The skills describe when and how to use these methods. The code implements repeatable operations.

## Near-Term Direction

The team should avoid full all-layer/all-token SAE decoding by default. Use cheap localization first:

1. black-box probes;
2. ablations or activation patching to find causal layers/sites;
3. targeted SAE feature inspection;
4. causal validation on held-out prompts.
