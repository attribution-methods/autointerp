How does GPT-2-small mechanistically implement indirect object identification (IOI)?

Given an IOI prompt of the form 'When [A] and [B] went to the [place], [B] gave a [object] to', the model assigns higher logit to the indirect-object name (A) than to the subject name (B) at the final-token prediction position. Holds across both ABBA and BABA template orderings.
