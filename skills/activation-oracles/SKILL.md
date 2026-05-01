---
name: activation-oracles
description: Activation oracle and verbalizer workflows for converting hidden states into natural-language hypotheses. Use when an auxiliary model, trained verbalizer, or nearest-neighbor explanation system can describe activations and suggest behavioral hypotheses for follow-up investigation.
---

# Activation Oracles

## Workflow

1. Select fixed prompts and layer positions before querying the oracle.
2. Query multiple times per activation when the oracle is stochastic.
3. Cluster or filter descriptions by semantic coherence and specificity.
4. Translate descriptions into testable behavioral hypotheses.
5. Validate hypotheses with black-box probes and causal interventions.

## Inputs

Provide the oracle with activation vectors, layer/component metadata, token context, and a narrowly scoped question such as: "What topic or behavior is represented here?"

## Evidence

Useful oracle outputs name concrete domains, entities, goals, or situations that guide follow-up prompts. Generic outputs about assistant behavior are weak evidence.

## Cautions

Oracle text is model-generated evidence about activations, not ground truth. Treat it as hypothesis generation unless validated independently.
