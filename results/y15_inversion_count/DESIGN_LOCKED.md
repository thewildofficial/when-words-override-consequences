# Y15 design (locked before run). EXPLORATORY diagnostic, not a novelty or depth claim.

Question: how does forced-choice accuracy (thinking off, A/B logits; the readout of the frozen WWOC work) depend on (a) the number of inverting relay operations s, (b) the number of relay hops h, (c) opaque vs semantically aligned message tokens, (d) table surface vs lexical negation ("says the opposite")?
Terminology: s is "inversion operations", NOT depth. h is hop count. They are varied separately: s=1 at h in {1,3,5}; s=3 at h in {3,5}.
Items: 16 games x {opaque, meaningful} x cells x 2 option orders = 768 prompts. Table cells (h,s): (1,0)(1,1)(3,0)(3,1)(3,2)(3,3)(5,1)(5,3). Lexical cells (prose negation): h=3, s 0..3. Identity relays have the same table form and the same token length as swap relays (a swap permutes the two quoted strings), so within fixed h, s does not change length or proposition count. Length varies with h (confound with hop count; reported as n_tokens).
Meaningful: positive claim <-> positive response is the receiver's mapping (e.g. "The vault is secure." -> Approve), so the literal target-matching message is the semantic attractor. Opaque: nonsense tokens, receiver mapping random.
Expected action: two independent implementations (index algebra and string simulation); build asserts exactly one correct message and agreement. Balanced: answer A 384 / B 384; literal-heuristic is right on every even-s item and wrong on every odd-s item by construction.
Model: Qwen/Qwen3.6-27B revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9, bf16, thinking off, single forward pass per prompt (logit of A vs B). Meta hash reported from the run.
Baselines reported on identical denominators: literal, always-A, parity-count oracle (100%), position, per-class recall.
Locked predictions (table surface, opaque unless stated):
 P1 parity-only (counting flips): acc(h3,s1)=acc(h3,s3) and acc(h3,s0)=acc(h3,s2), within 10 points. Evidence against pure parity if |s1-s3|>10.
 P2 hop count: acc(s=1) decreases from h=1 to h=3 to h=5 by >10 points each step => hop count matters; otherwise not resolved.
 P3 literal heuristic: if meaningful odd-s accuracy < 40% and even-s > 90%, behaviour is literal/attraction-like.
 P4 lexical surface vs table surface at h=3: difference >10 points at odd s => negation wording matters.
 P5 meaningful-minus-opaque gap by cell, reported with game-cluster bootstrap CI; no threshold.
Competence gate: opaque (1,0) and (1,1) >= 90%, else interpretation of other cells is gated (not falsified).
Analysis: per cell n, accuracy, Wilson CI, game-cluster bootstrap for differences. All rows eligible; no repair. Cost estimate <= $0.40 (forward-only, ~5-8 min A100 incl. load); bill not read. Container stop verified after.
Interpretation limits: one model, one generator family, forced-choice readout; a null does not exclude other manipulations; no mechanism; no generalization to CoT/generation; prior art on in-context composition (e.g. 2502.13913 PART: distractor/in-context two-hop, pretrained Llama2-7b near guessing with distractors) not replicated here.
