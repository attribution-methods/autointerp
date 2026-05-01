# Architecture

`autointerp` has three layers:

1. `autointerp`: schemas, pipelines, and domain tools for interpretability.
2. `skills`: procedural knowledge for choosing and applying methods.
3. `autointerp_agent`: an agent runtime that can select skills, call tools, request approval, and run locally or through MCP-style integrations.

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

## Artifact Layer

`autointerp.schemas` defines the portable JSON contracts that let components
exchange evidence:

- `BehaviorSpec`, `PromptBatch`, and `GenerationSample` for black-box work;
- `ActivationCacheRef`, `CandidateSite`, and `FeatureFinding` for white-box
  discovery;
- `InterventionResult`, `ValidationResult`, and `InvestigationReport` for
  causal checks and summaries.

Large tensors stay out of report JSON and are referenced by path. This keeps the
same report usable by the CLI agent, circuitbreaker, notebooks, and later MCP
servers.

## Near-Term Direction

The team should avoid full all-layer/all-token SAE decoding by default. Use cheap localization first:

1. black-box probes;
2. ablations or activation patching to find causal layers/sites;
3. targeted SAE feature inspection;
4. causal validation on held-out prompts.
