# Bounded literature assessment: latent composition papers vs the WWOC question (CPU, 07:30)
Scope: what causal evidence exists and whether WWOC's question overlaps. Claims scoped to each paper's own setting.

## Hopping Too Late (Biran et al., 2406.12775; read depth: PART+, abstract, intro, dataset, Sections 4-6 methods/results from the arXiv pdf text; Appendices and limitations not read)
- Setting: 82,020 factual two-hop queries; LLaMA 2 7B/13B, LLaMA 3 8B/70B; (also Pythia in a table). Natural pretrained facts, no fine-tuning.
- Causal evidence: attention knockout of t1->t2 (window of 7 layers), Patchscopes and vocabulary projection show the bridge entity is resolved in early layers and information moves to the last token; back-patching a later-layer state to an earlier layer fixes 32%-66% of incorrect cases while keeping correct cases at 100% (they note only existence of some successful source/target pair, a search over pairs).
- Their stated reading: later layers sometimes lack the needed functionality for the second hop. Caveat for transfer: "exists a patch that works" is a search-based upper bound, and facts are knowledge recall, not in-context policy rules.
## How Do LLMs Perform Two-Hop Reasoning in Context? (2502.13913; PART: Sections 1-4 body, appendix not read)
- Pretrained Llama2-7b-base with distractor chains: P(End-T) falls 0.82 (K=1) to 0.41, 0.31, 0.25, 0.20 as chains are added (Table 1 excerpt). Fine-tuning on 1,000 prompts restores it.
- Mechanism evidence is from a 3-layer attention-only transformer trained from scratch (attention logits, logit lens); the pretrained 7B evidence is behavioural only. A phase transition from uniform guessing to sequential querying is shown in the toy model.
## Lessons from Studying Two-Hop Latent Reasoning (2411.16353; PART: abstract, intro, Sec 2-3 text)
- Fully synthetic facts learned in separate documents: chance no-CoT accuracy though CoT and one-hop are near perfect; success when facts co-occur in a document/prompt or one fact is natural (about 20% no-CoT in the semi-synthetic setup). Fine-tuned models, not rule-in-context.
## Not read: Pfau filler tokens, Press compositionality gap (cited only); Feng et al. extractive structures; Lindsey biology (cited only).

## Overlap with the WWOC question
- Shared theme: composition without written steps is weaker than with written steps. WWOC's RBG-6 rescue (46/48 vs 18/48) is the same direction as CoT-vs-no-CoT gaps in these papers. It is a different task (in-context action policy, forced choice) so it neither replicates nor contradicts them.
- Distractor sensitivity (2502.13913) is a candidate competing explanation for the RBG-4 history harm that the existing opaque control does not exclude; that paper's own evidence is behavioural for the 7B model.
- Causal evidence in these papers comes from patching/knockout on factual recall; WWOC has no passing causal edit (semantic mechanisms: no patch site qualified), so nothing here licenses a mechanism claim for WWOC.
## Verdict
No new direction is established. The literature supplies a competing explanation (distractor-like interference) and prior art for composition limits; it does not make a WWOC-specific causal question tractable with our current evidence. Not exhaustive; novelty not claimed.
