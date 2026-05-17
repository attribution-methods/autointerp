How does the model internally represent 'surprise' — the property of a continuation being unexpected given its context — in a piece of text?

Given matched (surprising-continuation, mundane-continuation) text pairs, the model's internal activations differ in a way that tracks surprise rather than surface form. Operationalised as: a readout fit on one split separates surprising from mundane contexts on a disjoint split, and steering along the recovered representation causally shifts the model's behavior toward/away from treating a continuation as surprising.
