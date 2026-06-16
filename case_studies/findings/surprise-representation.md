# Representation of Surprise in Text

**Question:** How do models represent surprise in a given text?
**Run:** model `gpt2-medium`, driver gpt-5-nano
**Verdict:** NOT SUPPORTED

## What the agent did
The agent planned a two-stage investigation (black_box then localization) testing whether surprising events shift next-token probability mass toward surprise-related tokens (e.g. exclamations, gasps). It ran only Stage 0 (black_box) with the `logit_diff` metric, computing surprise-token logit mass minus neutral-token mass. The pre-registered success criterion was `logit_diff >= 0.05` on the dev split.

## What it found
The observed `logit_diff` was -7.473e-05, a tiny negative value far below the 0.05 threshold, so the `surprise_signal_dev` criterion failed (0/1 passed). The agent concluded the surprise signal was not detectable under this setup and the hypothesis was not supported, flagging that only 40 of the intended 200 synthetic prompts were processed.

## Takeaway (testbed reach)
The black_box stage ran cheaply on a small open-weights model, but the verdict rests on a simplistic fixed-token `logit_diff` proxy over a subsampled (40-prompt) dev set, so the negative result is more a non-detection than a strong refutation.
