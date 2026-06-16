# Load-Bearing Tokens in Chain-of-Thought

**Question:** When LLMs reason in tokens, are there load-bearing tokens or sequences of tokens in the chain-of-thought?
**Run:** model `gpt2`, driver gpt-5-nano
**Verdict:** paused for plan revision

## What the agent did
The agent planned a three-stage investigation on GPT-2 over a 100-sample GSM8K dev set, contrasting a CoT prompt against a concise prompt. Stage 0 (black-box) measured `accuracy` and `logit_diff`; Stage 1 (localization) measured `kl_to_clean` and `ablation_drop`; Stage 2 (intervention) was to causally test load-bearing components via activation/path patching, scored by `patch_effect_recovery`. Two criteria were pre-registered: `cot_advantage` (accuracy >= 0.05) and `circuit_recovery` (patch_effect_recovery >= 0.4).

## What it found
Only 1 of 2 criteria was tested: `cot_advantage` passed with observed accuracy 1.0. Stage 1 committed `kl_to_clean` = 0.3875 and `ablation_drop` = 0.4. The run then paused: Stage 2 patching "cannot be completed with current tooling/time constraints," ad hoc placeholder inputs were rejected by the metrics engine, so the agent filed a formal spec-revision request rather than reaching a verdict.

## Takeaway (testbed reach)
GPT-2-class models do not genuinely produce chain-of-thought reasoning, so this question is not really tractable on the testbed; the run stalled before any causal patching evidence could be gathered.
