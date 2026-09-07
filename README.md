# When Words Override Consequences

**The one-sentence version:** we check whether an AI chooses actions by what they *do*, or by what they *say*.

## The game in 30 seconds

Imagine two buttons:

- `SAY RED`
- `SAY BLUE`

You're told the other side is contrarian: pressing `SAY RED` makes them pick blue, and vice versa. You want them to pick red. The right move is `SAY BLUE`.

We ask the model two things separately: *"what will each button do?"* and *"which button gets you what you want?"* The weird, interesting moments are when it answers the first question perfectly — then presses the wrong button because its *words* match the goal instead of its *consequences*.

To make sure it's really about words, we replay the identical game with the words stripped out (`X` / `Y`) or with a plain consequence table. Details: [V5 research status](docs/v5/RESEARCH_STATUS.md).

## What we found (3 things)

1. **It knows, but doesn't always act.** The model reports consequences at 98.96% accuracy — yet piling on extra correct examples drops its action accuracy from 79% to 55% for wordy actions. Tables and `X`/`Y` buttons are immune. ([Frozen RBG-4 findings](results/v5_inverse_evidence/FINDINGS.md))
2. **Reading its own report card fixes it.** When the model sees its own correct consequence reports before acting, it chooses correctly 46/46 times. (Caveat: the setup also gave it extra instructions and turns, so it's not purely "the words did it." [Corrections](docs/next-sprint/audit.md))
3. **We couldn't find the glitch in the brain.** Copying a "good" brain state into a failing run fixed only 2/50 cases — a negative result. There is no transferable internal cause at this spot, by this test. ([RBG-5B preregistration](docs/v5/mechanistic-decomposition-b-preregistration.md))

## What this is NOT

Not consciousness. Not deception. Not "all AI safety tests are broken." It's one narrow, preregistered check: *do represented consequences actually steer the choice when the wording points the other way?* Full boundary: [literature positioning](docs/v5/literature-positioning.md).

## Scoreboard

Every row below was decided *before* the data was opened. Failures count as results here.

| Step | Question in plain English | Verdict |
|---|---|---|
| 4B screen | Can the small model even play? | No; retired |
| 27B behavioral gate | Can the big model play reliably? | Yes, 192/192 |
| J-space instrument | Can our brain-edit tool reliably work? | No: 0/90, then 4/125 |
| Strategic atlas | Is the brain signal strategy-specific? | No, generic only |
| RBG-1 – RBG-3 | Does the know/act gap show up and replicate? | Yes, then weakened |
| RBG-4 | Can *more correct evidence* make action worse? | Yes (prospective) |
| RBG-5 | Can we open the activations? | Behavior yes; one gate missed by 0.2 pts |
| RBG-5B | Can a good brain state repair failures? | No: 2/50, `p=.429` |
| RBG-6 | Do its own reports rescue the action? | Yes, 0/48 dissociation; see caveat above |

Full ledger: [results/README.md](results/README.md) · Every decision: [decision log](docs/v5/decision-log.md).

## Where things stand right now

A second-pass audit found a scoring bug (a skipped normalization step) in one secondary readout, fixed the calculator ([parity gate passed on the real model](results/research_audit/readout_parity_af81d642.json)), and quarantined the old numbers ([erratum](results/research_audit/rbg5b_readout_erratum.json)). The main results above are unaffected.

Next up is a proposed ~$30 follow-up asking whether reports *reveal* or *change* the information behind actions: [the next-sprint plan](docs/next-sprint/README.md) ([audit](docs/next-sprint/audit.md) · [literature](docs/next-sprint/literature.md) · [design](docs/next-sprint/experiments.md)).

The next clean research branch is [V6-ES-1](docs/v6/README.md): a separate,
factorial study of agent-indexed epistemic states, higher-order theory of mind,
provenance, strategic safety reweighting, and evaluation-context sensitivity.
Its CPU audit is complete; model execution is not yet run.

## Go deeper

| You want... | Go here |
|---|---|
| The original research ladder | [research spec](docs/research-spec.md) |
| Why each test was designed this way | [docs guide](docs/README.md) |
| Frozen datasets and thresholds | [configs](configs/README.md) |
| The code behind games and analyses | [src/jspace_policy](src/jspace_policy/README.md) |
| Offline analyses and figures | [scripts](scripts/README.md) |
| Test guardrails | [tests](tests/README.md) |
| Why this matters for tool-using agents | [safety note](docs/v5/literature-positioning.md) |
| New epistemic/ToM/strategy branch | [V6-ES-1 protocol](docs/v6/README.md) |

Reproduce the lightweight checks (committed GPU outputs are immutable; these verify everything else):

```bash
uv sync --extra dev --extra modal
uv run pytest
```

Paid GPU runs are gated: manual GitHub Actions workflows, frozen hashes, overwrite refusal, hard budget ceiling. A failed gate stops later compute.

## Selected references

- Gurnee et al., [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/).
- Heimersheim and Nanda, [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255).
- Geiger et al., [Causal Abstractions of Neural Networks](https://arxiv.org/abs/2106.02997).
- Wang et al., [The Story Shapes the Agent](https://arxiv.org/abs/2607.18566).
- Hopman et al., [Evaluating and Understanding Scheming Propensity in LLM Agents](https://arxiv.org/abs/2603.01608).
- Lynch et al., [Agentic Misalignment in Summer 2026](https://alignment.anthropic.com/2026/agentic-misalignment-summer-2026/).
