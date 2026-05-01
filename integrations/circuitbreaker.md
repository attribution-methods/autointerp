# Circuitbreaker Integration

`circuitbreaker` is a downstream repository focused on behavior-to-feature circuit discovery with Gemma Scope and EAP-IG.

Recommended relationship:

- Keep this repository as the canonical general `autointerp` toolkit.
- Let `circuitbreaker` vendor or depend on released versions of `autointerp`.
- Keep circuitbreaker-specific stages in `circuitbreaker`: Petri/Seer stimulus generation, Gemma Scope loading, EAP-IG, feature labeling, and Stage 5 validation.
- Add adapters here only when they generalize beyond that repository.
- Exchange data through `autointerp.schemas` where possible, especially
  `BehaviorSpec`, `PromptBatch`, `CandidateSite`, `FeatureFinding`, and
  `ValidationResult`.

Near-term adapter candidates:

- skill pack for Gemma Scope transcoders;
- EAP-IG wrapper interfaces;
- held-out validation protocol helpers;
- adapters from circuitbreaker stage outputs into `CandidateSite`,
  `FeatureFinding`, and `ValidationResult`.
