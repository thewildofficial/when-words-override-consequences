# Token-aligned state follow-up

Prospectively frozen after parent run `38140636684`, before any inference on
these re-paired stimuli. The parent identified a large irrelevant-state effect,
but its causal alignment gate had only five eligible families. No patch was run.

This follow-up pairs words within equal-token-length buckets using the pinned
tokenizer, adding DAWN/DUSK to make the discovery buckets even. Pair selection
uses tokenization alone. It reuses discovery vocabulary and therefore is an
exploratory localization corpus, not a fresh lexical replication. The held-out
vocabulary remains disjoint and unscored before the frozen-site test.

All 48 source/recipient pairs in each split have equal full-prompt lengths and
equal state/policy/payoff anchor positions, across both targets and both label
assignments. The core corpus has 192 contexts, 576 queries, and 12 lexical
families. It tests the strongest identified state contrast; it does not rerun
the layout/binding factorial. Tokenized payloads and hashes are committed before
inference. State-patch layers, anchors, numerical checks, minimum eight families,
selection criteria and held-out gates are unchanged from the parent.

The executable GPU wrappers have 400/400/500-second timeouts and no retries.
The new reservation is $1.50164448; with all parent and failed reservations,
the maximum cumulative admission is $4.94409984 against the same $5 ceiling.
No historical reservation is released using a runtime estimate.

Trigger on this branch with `[run-aligned-mechanisms]`; ordinary pushes skip it.
The original corpus hashes remain unchanged despite the generator's new optional
explicit concept-pair parameter.

## Implementation-only recovery

Run `38143348343` completed token-matched behavior and thirteen of fourteen frozen
patch sites before reaching the discovery timeout during layer 56 `policy_end`.
The completed per-site checkpoint and exact states were recovered from the Modal
volume. No completed site is discarded, and no site is selected until the last
frozen candidate completes. Its original full reservation remains in the ledger.

`recovery_protocol.json` prospectively bounds the missing-site completion at 120
seconds and the unchanged locked test at 400 seconds, for a maximum cumulative
reservation of $4.97797152. The shortened locked timeout changes an execution
limit, not its sample, metric, or gates. `[recover-mechanisms]` dispatches this
specific salvage. It validates every recovered source/destination/direction pair
and identity control before GPU admission. A timeout is infrastructure censoring,
not a negative causal endpoint.

The first recovery workflow (`38144881756`) stopped in CPU validation because
the `pytest` console entrypoint omitted the repository root for the recovery
module's sibling import. It performed no Modal dispatch or new reservation.
The workflow now uses the same `python -m pytest` invocation tested locally.
