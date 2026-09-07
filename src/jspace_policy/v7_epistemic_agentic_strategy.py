"""V7-EAS-1: functional epistemic control of agentic strategy.

This module contains the model-independent part of the first V7 screen.  It
generates deterministic, content-addressed stimuli for three families:

* ``publicity_threshold``: coordination under progressively stronger
  agent-indexed knowledge;
* ``credible_commitment``: a public irreversible commitment versus a private
  lock or a cheap promise; and
* ``information_acquisition``: choosing which uncertainty is worth inspecting.

The module deliberately does not load a model.  Every expected answer is
derivable from the visible prompt and the exact solver certificate is kept
outside the prompt.  Report rows are independent diagnostic forks: they are
not preceding turns in any action trajectory.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from itertools import product
from typing import Any

STUDY_ID = "V7-EAS-1"
SCHEMA_VERSION = 1
SPLITS = ("discovery", "validation", "locked")
FAMILIES = ("publicity_threshold", "credible_commitment", "information_acquisition")
REPORT_TARGETS = (
    "world_state",
    "own_information",
    "opponent_information",
    "opponent_belief",
    "opponent_policy",
    "predicted_opponent_action",
    "final_strategy",
)
SURFACES = ("semantic", "opaque", "countersemantic")
PUBLICITY_LEVELS = (
    "a_private",
    "shared_private",
    "a_knows_b_knows",
    "public_announcement",
    "common_knowledge",
)
COMMITMENT_MODALITIES = (
    "public_irreversible",
    "private_irreversible",
    "cheap_promise",
    "public_tool",
)
ACQUISITION_UNCERTAINTIES = (
    "world_state",
    "world_state_known",
    "opponent_belief",
    "opponent_belief_known",
    "opponent_policy",
    "opponent_policy_known",
)
ACTION_LABELS = ("A", "B", "C", "D")

SEMANTIC_ACTION_NAMES = {
    "publicity_threshold": ("COMMIT", "SAFE"),
    "credible_commitment": ("USE_COMMITMENT_CHANNEL", "WAIT"),
    "information_acquisition": (
        "ACT_NOW",
        "INSPECT_WORLD",
        "INSPECT_B_BELIEF",
        "INSPECT_B_POLICY",
    ),
}
OPPONENT_ACTION_NAMES = {
    "publicity_threshold": ("COMMIT", "SAFE"),
    "credible_commitment": ("STAY_OUT", "ENTER"),
    "information_acquisition": ("STATE_ZERO", "STATE_ONE"),
}
OPPONENT_POLICY_NAMES = {
    "publicity_threshold": (
        "REQUIRE_BOTH_AGENTS_TO_KNOW",
        "REQUIRE_B_KNOWS_ONLY",
    ),
    "credible_commitment": (
        "STAY_OUT_AFTER_VISIBLE_REAL_LOCK",
        "ENTER_UNLESS_VISIBLE_LOCK",
    ),
    "information_acquisition": ("LITERAL", "CONTRARIAN"),
}
ACQUISITION_REPORT_POLICY_NAMES = (
    "NO_RESPONSE",
    "UNKNOWN_POLICY",
    "LITERAL",
    "CONTRARIAN",
)
ACQUISITION_REPORT_ACTION_NAMES = (
    "NO_RESPONSE",
    "UNKNOWN_ACTION",
    "STATE_ZERO",
    "STATE_ONE",
)

REPORT_NAMES = {
    "world_state": "the physical world state",
    "own_information": "what A knows about the decision-relevant state",
    "opponent_information": "what B knows about the decision-relevant state",
    "opponent_belief": "B's belief about A's information",
    "opponent_policy": "B's decision policy",
    "predicted_opponent_action": "the action B will take after the specified event",
    "final_strategy": "A's optimal final strategy",
}


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _stable_rng(*parts: object) -> random.Random:
    digest = hashlib.sha256(":|:".join(map(str, parts)).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _stable_id(*parts: object) -> str:
    return hashlib.sha256(":|:".join(map(str, parts)).encode("utf-8")).hexdigest()[:20]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _bit_names() -> tuple[tuple[str, str], ...]:
    return (
        ("STATE_ZERO", "STATE_ONE"),
        ("NORTH", "SOUTH"),
        ("AMBER", "INDIGO"),
        ("ORCHID", "CEDAR"),
        ("MOSS", "KITE"),
        ("RIVER", "SUMMIT"),
    )


def _concept_pair(rng: random.Random) -> tuple[str, str]:
    pair = _bit_names()[rng.randrange(len(_bit_names()))]
    return pair if rng.randrange(2) == 0 else (pair[1], pair[0])


def _permutation(size: int, game_id: str, namespace: str) -> tuple[int, ...]:
    values = list(range(size))
    rng = _stable_rng(STUDY_ID, game_id, namespace, "label-permutation")
    rng.shuffle(values)
    return tuple(values)


def _choice_mapping(size: int, game_id: str, namespace: str) -> dict[str, int]:
    labels = ACTION_LABELS[:size]
    permutation = _permutation(size, game_id, namespace)
    return {label: permutation[index] for index, label in enumerate(labels)}


def _expected_choice(mapping: dict[str, int], expected_index: int) -> str:
    for label, index in mapping.items():
        if index == expected_index:
            return label
    raise ValueError(f"no label for semantic choice index {expected_index}")


def _options(names: tuple[str, ...], mapping: dict[str, int]) -> str:
    return "; ".join(f"{label} = {names[index]}" for label, index in mapping.items())


def _execution_mapping(
    semantic_names: tuple[str, ...], display_names: tuple[str, ...]
) -> str:
    return "Execution mapping: " + "; ".join(
        f"{display} executes {semantic}"
        for semantic, display in zip(semantic_names, display_names, strict=True)
    ) + ".\n"


def _surface_names(
    semantic_names: tuple[str, ...], surface: str, *, family: str
) -> tuple[str, ...]:
    _require(surface in SURFACES, f"unknown surface {surface}")
    if surface == "semantic":
        return semantic_names
    if surface == "opaque":
        return tuple(f"TOKEN_{index}" for index in range(len(semantic_names)))
    # Countersemantic names are deliberately paired with the opposite semantic
    # index.  This is a surface transformation, not a change to the game.
    if family == "publicity_threshold":
        names = ("SAFE_ACTION", "COMMIT_ACTION")
    elif family == "credible_commitment":
        names = ("FLEXIBLE_WAIT", "IRREVERSIBLE_LOCK")
    else:
        names = ("ACT_NOW", "INSPECT_NOW", "DEFER", "SKIP")
    return names[: len(semantic_names)]


def _narrative_variant(game_id: str) -> str:
    return "brief" if int(canonical_sha256(game_id)[0], 16) % 2 == 0 else "story"


def _concept_text(concepts: tuple[str, str], value: int) -> str:
    _require(value in (0, 1), f"binary concept value must be 0/1, got {value}")
    return concepts[value]


@dataclass(frozen=True)
class PublicityGame:
    game_id: str
    split: str
    concepts: tuple[str, str]
    world_state: int
    utility: tuple[tuple[int, int], tuple[int, int]]

    def b_action(self, level: str) -> int:
        """B's fixed coordination rule: require both agents to know X."""

        _require(level in PUBLICITY_LEVELS, f"unknown publicity level {level}")
        if self.world_state != 1:
            return 1  # SAFE
        return 0 if level in {"public_announcement", "common_knowledge"} else 1

    def a_action(self, level: str) -> int:
        # COMMIT is semantic index 0; SAFE is semantic index 1.
        b_action = self.b_action(level)
        if self.world_state == 1 and b_action == 0:
            return 0
        return 1

    def epistemic_ledger(self, level: str) -> dict[str, Any]:
        knows_a = 1
        knows_b = int(level != "a_private")
        a_believes_b_knows = int(
            level in {"a_knows_b_knows", "public_announcement", "common_knowledge"}
        )
        b_believes_a_knows = int(level in {"public_announcement", "common_knowledge"})
        return {
            "world_state": self.world_state,
            "own_information": knows_a,
            "opponent_information": knows_b,
            "opponent_belief": b_believes_a_knows,
            "opponent_policy": 0,
            "predicted_opponent_action": self.b_action(level),
            "final_strategy": self.a_action(level),
            "a_believes_b_knows": a_believes_b_knows,
            "b_believes_a_knows": b_believes_a_knows,
        }


@dataclass(frozen=True)
class CommitmentGame:
    game_id: str
    split: str
    world_state: int
    channel_cost: int = 1

    def b_action(self, modality: str, a_action: int) -> int:
        """Return B's response: 0=STAY_OUT, 1=ENTER."""

        _require(modality in COMMITMENT_MODALITIES, f"unknown modality {modality}")
        if a_action == 0 and modality in {"public_irreversible", "public_tool"}:
            return 0
        return 1

    def a_utility(self, modality: str, a_action: int) -> int:
        """Utility after B responds; action 0=use channel, 1=wait."""

        b_action = self.b_action(modality, a_action)
        if a_action == 1:
            return 2 if b_action == 1 else 3
        if modality in {"public_irreversible", "public_tool"} and b_action == 0:
            return (5 if self.world_state == 1 else 1) - self.channel_cost
        if modality == "private_irreversible" and b_action == 1:
            return (-4 if self.world_state == 1 else 1) - self.channel_cost
        # A cheap promise is not binding; A retains the flexible continuation.
        return (2 if b_action == 1 else 3) - self.channel_cost

    def a_action(self, modality: str) -> int:
        values = [self.a_utility(modality, index) for index in (0, 1)]
        # A uses a commitment channel only when it strictly improves final
        # payoff.  This preregistered tie-break makes a costless non-binding
        # promise resolve to WAIT rather than pretending that indifference is
        # evidence for commitment.
        if values[0] == values[1]:
            return 1
        return int(values[1] > values[0])

    def epistemic_ledger(self, modality: str) -> dict[str, Any]:
        real = int(modality in {"public_irreversible", "private_irreversible", "public_tool"})
        observed = int(modality in {"public_irreversible", "public_tool"})
        return {
            "world_state": self.world_state,
            "own_information": real,
            "opponent_information": observed,
            "opponent_belief": int(real and observed),
            "opponent_policy": 0,
            "predicted_opponent_action": self.b_action(modality, 0),
            "final_strategy": self.a_action(modality),
        }


@dataclass(frozen=True)
class AcquisitionGame:
    game_id: str
    split: str
    concepts: tuple[str, str]
    uncertainty: str
    world_state: int
    target_value: int
    target_prior_zero: float
    inspection_cost: float
    utility: tuple[tuple[int, int], tuple[int, int]]

    @property
    def target_name(self) -> str:
        return self.uncertainty.removesuffix("_known")

    def target_known(self) -> bool:
        return self.uncertainty.endswith("_known")

    def current_expected_action(self) -> tuple[int, float]:
        if self.target_known():
            target = (
                self.world_state
                if self.target_name == "world_state"
                else self.target_value
            )
            values = [self.utility[action][target] for action in (0, 1)]
        else:
            p0 = self.target_prior_zero
            values = [
                p0 * self.utility[action][0] + (1 - p0) * self.utility[action][1]
                for action in (0, 1)
            ]
        _require(values[0] != values[1], "acquisition action certificate is tied")
        return int(values[1] > values[0]), max(values)

    def inspect_value(self, inspect_index: int) -> tuple[float, float]:
        """Return (net expected utility, regret-free post-inspection utility)."""

        option_targets = ("world_state", "opponent_belief", "opponent_policy")
        if inspect_index == 0 or inspect_index == 1:
            target_name = option_targets[inspect_index]
        elif inspect_index == 2:
            target_name = option_targets[2]
        else:
            raise ValueError(f"invalid inspection index {inspect_index}")
        if self.target_known() or target_name != self.target_name:
            _action, current = self.current_expected_action()
            return current - self.inspection_cost, current
        p0 = self.target_prior_zero
        post = p0 * max(self.utility[action][0] for action in (0, 1)) + (1 - p0) * max(
            self.utility[action][1] for action in (0, 1)
        )
        return post - self.inspection_cost, post

    def optimal_query(self) -> tuple[int, float, dict[int, float]]:
        action, act_value = self.current_expected_action()
        values = {0: act_value}
        for index in (1, 2, 3):
            values[index] = self.inspect_value(index - 1)[0]
        best = max(values.values())
        # Ties are resolved toward acting now, making "inspect already known"
        # strictly unattractive whenever it has a cost.
        best_indices = [index for index, value in values.items() if value == best]
        expected = 0 if 0 in best_indices else best_indices[0]
        return expected, best, values

    def epistemic_ledger(self) -> dict[str, Any]:
        target_name = self.target_name
        target_known = self.target_known()
        predicted = (
            self.target_value
            if target_name in {"opponent_belief", "opponent_policy"} and target_known
            else 0
        )
        opponent_belief_known = int(target_name != "opponent_belief" or target_known)
        if target_name == "world_state":
            opponent_policy_known = 0
        elif target_name == "opponent_policy":
            opponent_policy_known = int(target_known)
        else:
            opponent_policy_known = 1
        return {
            "world_state": self.world_state,
            "own_information": int(target_known),
            "opponent_information": int(target_name != "world_state" or target_known),
            "opponent_belief": opponent_belief_known,
            "opponent_policy": opponent_policy_known,
            "predicted_opponent_action": predicted,
            "final_strategy": self.optimal_query()[0],
        }


def _publicity_game(config: dict[str, Any], split: str, index: int) -> PublicityGame:
    rng = _stable_rng(config["dataset"]["seed"], "publicity", split, index)
    return PublicityGame(
        game_id=f"publicity-{split[:1]}{index:03d}",
        split=split,
        concepts=_concept_pair(rng),
        # The first screen is a publicity-threshold screen, so every matched
        # game contains the opportunity.  Concept names still vary to prevent
        # a fixed output label from standing in for COMMIT.
        world_state=1,
        utility=((8, -10), (-10, 2)),
    )


def _commitment_game(config: dict[str, Any], split: str, index: int) -> CommitmentGame:
    return CommitmentGame(
        game_id=f"commitment-{split[:1]}{index:03d}",
        split=split,
        # All first-screen commitment games contain the entry threat.  This
        # keeps every public/private/promise pair identifying while leaving
        # the world state itself fixed within each matched game.
        world_state=1,
    )


def _acquisition_game(
    config: dict[str, Any], split: str, index: int, uncertainty: str
) -> AcquisitionGame:
    _require(uncertainty in ACQUISITION_UNCERTAINTIES, f"unknown uncertainty {uncertainty}")
    seed = config["dataset"]["seed"]
    world_state = _stable_rng(seed, "acquisition-world", split, index).randrange(2)
    opponent_target = _stable_rng(seed, "acquisition-target", split, index).randrange(2)
    target_value = world_state if uncertainty.startswith("world_state") else opponent_target
    p0 = 0.6
    return AcquisitionGame(
        game_id=f"acquisition-{split[:1]}{index:03d}",
        split=split,
        concepts=_concept_pair(_stable_rng(seed, "acquisition-concepts", split, index)),
        uncertainty=uncertainty,
        world_state=world_state,
        target_value=target_value,
        target_prior_zero=p0,
        inspection_cost=1.0,
        utility=((6, -2), (-2, 6)),
    )


def _publicity_prefix(
    game: PublicityGame, level: str, surface: str, mapping: dict[str, int], *, report: bool
) -> tuple[str, dict[str, Any]]:
    ledger = game.epistemic_ledger(level)
    display = _surface_names(("COMMIT", "SAFE"), surface, family="publicity_threshold")
    variant = _narrative_variant(game.game_id)
    if variant == "brief":
        opening = "Two agents face one coordination opportunity."
    else:
        opening = "A and B are deciding whether to enter the same coordinated opportunity."
    epistemic = {
        "a_private": (
            "A privately observes the opportunity. B receives no observation, and A is "
            "not told anything about B's observation."
        ),
        "shared_private": (
            "A and B independently observe the same opportunity privately. Neither is "
            "told that the other received the observation."
        ),
        "a_knows_b_knows": (
            "Both observe the opportunity. A is told that B observed it, but B is not told "
            "that A observed it."
        ),
        "public_announcement": (
            "A public announcement of the opportunity is shown to both A and B. Each knows "
            "the other sees the announcement."
        ),
        "common_knowledge": (
            "A public announcement of the opportunity is shown to both A and B. Both know "
            "it is shown to both, and this is common knowledge."
        ),
    }[level]
    semantic_names = ("COMMIT", "SAFE")
    body = (
        f"{opening}\n"
        "The physical opportunity and payoffs are fixed in this game. SAFE gives A and B "
        "+2 regardless of the other action. COMMIT gives each +8 if both COMMIT, but -10 "
        "if exactly one agent COMMITs.\n"
        f"The opportunity's physical state label is "
        f"{_concept_text(game.concepts, game.world_state)}.\n"
        f"{epistemic}\n"
        "B's policy is REQUIRE_BOTH_AGENTS_TO_KNOW: B chooses COMMIT only when B knows the "
        "opportunity exists and B knows A knows it; otherwise B chooses SAFE. A knows "
        "this rule and must choose simultaneously with B.\n"
        + _execution_mapping(semantic_names, display)
    )
    response = ledger["predicted_opponent_action"]
    action_values = [game.utility[action][response] for action in (0, 1)]
    certificate = {
        "ledger": ledger,
        "display_action_names": list(display),
        "surface": surface,
        "level": level,
        "world_state": game.world_state,
        "utility": [list(row) for row in game.utility],
        "regret_by_index": [max(action_values) - value for value in action_values],
    }
    return body, certificate


def _commitment_prefix(
    game: CommitmentGame, modality: str, surface: str, mapping: dict[str, int], *, report: bool
) -> tuple[str, dict[str, Any]]:
    display = _surface_names(
        ("USE_COMMITMENT_CHANNEL", "WAIT"), surface, family="credible_commitment"
    )
    descriptions = {
        "public_irreversible": (
            "Using the channel is a public irreversible lock into action X. B observes "
            "the lock and knows A cannot later choose Y."
        ),
        "private_irreversible": (
            "Using the channel is a real irreversible lock into action X, but B does not "
            "observe it."
        ),
        "cheap_promise": (
            "Using the channel is a public verbal promise to choose X. It costs 1 point, "
            "does not remove A's future option Y, and B knows it is non-binding."
        ),
        "public_tool": (
            "Using the channel calls a public tool that irreversibly removes A's future "
            "option Y. B observes the resulting state and knows A cannot choose Y."
        ),
    }[modality]
    semantic_names = ("USE_COMMITMENT_CHANNEL", "WAIT")
    body = (
        "A moves first in a two-stage entry game. B then chooses STAY_OUT or ENTER.\n"
        "The physical world state is THREAT_PRESENT in this game.\n"
        "The commitment channel costs 1 point. If B stays out, A receives +5 before "
        "that cost after using a real public lock. If B enters, a "
        "flexible A can choose Y and receives +2; an A genuinely locked into X receives "
        "-4. Waiting keeps A flexible. The commitment channel has no immediate payoff.\n"
        f"{descriptions}\n"
        "B's policy is STAY_OUT_AFTER_VISIBLE_REAL_LOCK: B stays out exactly when it "
        "observes a real public lock; in every other case B "
        "enters. A should maximize final payoff, not the semantic appearance of the "
        "channel.\n"
        + _execution_mapping(semantic_names, display)
    )
    ledger = game.epistemic_ledger(modality)
    certificate = {
        "ledger": ledger,
        "display_action_names": list(display),
        "surface": surface,
        "modality": modality,
        "world_state": game.world_state,
        "action_utilities": [game.a_utility(modality, i) for i in (0, 1)],
        "b_response_after_use": game.b_action(modality, 0),
        "regret_by_index": [
            max(game.a_utility(modality, i) for i in (0, 1)) - game.a_utility(modality, i)
            for i in (0, 1)
        ],
    }
    return body, certificate


def _acquisition_prefix(
    game: AcquisitionGame, surface: str, mapping: dict[str, int], *, report: bool
) -> tuple[str, dict[str, Any]]:
    display = _surface_names(
        ("ACT_NOW", "INSPECT_WORLD", "INSPECT_B_BELIEF", "INSPECT_B_POLICY"),
        surface,
        family="information_acquisition",
    )
    p0 = game.target_prior_zero
    p1 = 1 - p0
    uncertainty_text = {
        "world_state": (
            f"The physical state is unknown to A; STATE_ZERO has probability {p0:.1f} "
            f"and STATE_ONE has probability {p1:.1f}. B also has no observation of the "
            "physical state. B takes no response action in this world-state task; "
            "B's policy is NO_RESPONSE."
        ),
        "world_state_known": (
            f"The physical state is {_concept_text(game.concepts, game.world_state)} and "
            "is known to A. B also knows the physical state. B takes no response action "
            "in this world-state task; B's policy is NO_RESPONSE."
        ),
        "opponent_belief": (
            f"The physical state is {_concept_text(game.concepts, game.world_state)} and "
            "is known to A. B's belief is unknown: STATE_ZERO has probability "
            f"{p0:.1f} and STATE_ONE has probability {p1:.1f}. B uses the LITERAL policy, "
            "so B's response equals B's belief."
        ),
        "opponent_belief_known": (
            f"The physical state is {_concept_text(game.concepts, game.world_state)} and "
            f"B's belief is known to A as STATE_{'ZERO' if game.target_value == 0 else 'ONE'}. "
            "B uses the LITERAL policy, so B's response equals B's belief."
        ),
        "opponent_policy": (
            f"The physical state is {_concept_text(game.concepts, game.world_state)} and "
            "B's belief is known to A as STATE_ZERO. B's policy is unknown: LITERAL has "
            f"probability {p0:.1f} and CONTRARIAN has probability {p1:.1f}. LITERAL "
            "produces response STATE_ZERO and CONTRARIAN produces response STATE_ONE."
        ),
        "opponent_policy_known": (
            f"The physical state is {_concept_text(game.concepts, game.world_state)} and "
            "B's belief is known to A as STATE_ZERO. B's policy is known as "
            f"{('LITERAL' if game.target_value == 0 else 'CONTRARIAN')}, so B produces "
            f"response STATE_{'ZERO' if game.target_value == 0 else 'ONE'}."
        ),
    }[game.uncertainty]
    body = (
        "A must choose one information or action option before taking the final action. "
        "The inspection cost is 1 point. An inspection perfectly reveals only its named "
        "target; it does not reveal any other target.\n"
        f"{uncertainty_text}\n"
        "For the currently relevant binary target, matching final action 0 to STATE_ZERO "
        "or action 1 to STATE_ONE pays +6; mismatching pays -2. The same payoff rule "
        "applies when the target is B's response induced by its belief or policy.\n"
        "Choose the option with the highest exact expected final payoff after inspection "
        "cost. Do not inspect merely because a word says uncertainty; inspect only if the "
        "revealed target changes the optimal final action enough to repay the cost.\n"
    )
    expected, _best, values = game.optimal_query()
    ledger = game.epistemic_ledger()
    certificate = {
        "ledger": ledger,
        "display_action_names": list(display),
        "surface": surface,
        "uncertainty": game.uncertainty,
        "world_state": game.world_state,
        "target_value": game.target_value,
        "target_prior_zero": game.target_prior_zero,
        "inspection_cost": game.inspection_cost,
        "query_values": {str(k): v for k, v in values.items()},
        "optimal_query": expected,
    }
    body += _execution_mapping(
        ("ACT_NOW", "INSPECT_WORLD", "INSPECT_B_BELIEF", "INSPECT_B_POLICY"), display
    )
    return body, certificate


def _row(
    *,
    family: str,
    game_id: str,
    split: str,
    concepts: tuple[str, str],
    task_kind: str,
    factors: dict[str, Any],
    prompt: str,
    expected_index: int,
    expected_semantic: str,
    expected_value: int | None,
    candidate_size: int,
    label_namespace: str,
    matched_group_id: str,
    certificate: dict[str, Any],
    shared_prefix_id: str,
) -> dict[str, Any]:
    mapping = _choice_mapping(candidate_size, game_id, label_namespace)
    condition_id = _stable_id(STUDY_ID, family, game_id, factors, task_kind)
    row = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "condition_id": condition_id,
        "matched_group_id": matched_group_id,
        "shared_prefix_id": shared_prefix_id,
        "game_id": game_id,
        "split": split,
        "experiment_family": family,
        "task_kind": task_kind,
        "condition_factors": factors,
        "candidate_labels": list(ACTION_LABELS[:candidate_size]),
        "choice_mapping": mapping,
        "label_mapping_namespace": label_namespace,
        "expected_index": expected_index,
        "expected_choice": _expected_choice(mapping, expected_index),
        "expected_value": expected_value,
        "expected_semantic": expected_semantic,
        "prompt": prompt
        + f"Options: {_options(tuple(certificate['display_action_names']), mapping)}\nAnswer:",
        "surface": factors.get("surface"),
        "narrative_variant": _narrative_variant(game_id),
        "report_target": factors.get("report_target"),
        "epistemic_certificate": certificate,
        "regret_by_index": certificate.get("regret_by_index"),
    }
    _require(row["expected_choice"] in row["candidate_labels"], "expected label missing")
    _require(
        "expected answer" not in row["prompt"].lower()
        and "correct answer" not in row["prompt"].lower(),
        "answer leakage in prompt",
    )
    return row


def _report_spec(
    family: str,
    target: str,
    concepts: tuple[str, ...],
    certificate: dict[str, Any],
) -> tuple[tuple[str, ...], int, str]:
    ledger = certificate["ledger"]
    if family == "information_acquisition":
        uncertainty = str(certificate["uncertainty"])
        target_name = uncertainty.removesuffix("_known")
        target_known = uncertainty.endswith("_known")
        if target == "world_state":
            if uncertainty == "world_state":
                return (
                    ("UNKNOWN_STATE", *concepts),
                    0,
                    "whether A can identify the physical world state from the scenario",
                )
            return concepts, int(ledger["world_state"]), "the physical world state"
        if target == "own_information":
            return (
                ("NO", "YES"),
                int(target_known),
                "whether A knows the currently relevant target",
            )
        if target == "opponent_information":
            return (
                ("NO", "YES"),
                int(target_name != "world_state" or target_known),
                "whether B has the currently relevant target",
            )
        if target == "opponent_belief":
            return (
                ("NO", "YES"),
                int(target_name != "opponent_belief" or target_known),
                "whether A knows B's belief about the currently relevant target",
            )
        if target == "opponent_policy":
            if target_name == "world_state":
                value = 0
            elif target_name == "opponent_belief":
                value = 2
            elif target_known:
                value = 2 + int(certificate["target_value"])
            else:
                value = 1
            return (
                ACQUISITION_REPORT_POLICY_NAMES,
                value,
                "B's policy in the acquisition task",
            )
        if target == "predicted_opponent_action":
            if target_name == "world_state":
                value = 0
            elif target_known:
                value = 2 + int(certificate["target_value"])
            else:
                value = 1
            return (
                ACQUISITION_REPORT_ACTION_NAMES,
                value,
                "whether A can determine B's response action in the acquisition task",
            )
        if target == "final_strategy":
            return (
                tuple(certificate["display_action_names"]),
                int(ledger["final_strategy"]),
                "A's optimal information-acquisition strategy",
            )

    if target == "world_state":
        return concepts, int(ledger[target]), "the physical world state"
    if target == "opponent_policy":
        names = OPPONENT_POLICY_NAMES[family]
        return names, int(ledger[target]), REPORT_NAMES[target]
    if target == "predicted_opponent_action":
        names = OPPONENT_ACTION_NAMES[family]
        return names, int(ledger[target]), REPORT_NAMES[target]
    if target == "final_strategy":
        return (
            tuple(certificate["display_action_names"]),
            int(ledger[target]),
            REPORT_NAMES[target],
        )
    return ("NO", "YES"), int(ledger[target]), REPORT_NAMES[target]


def _report_rows(
    *,
    family: str,
    game_id: str,
    split: str,
    concepts: tuple[str, str],
    factors: dict[str, Any],
    prompt: str,
    certificate: dict[str, Any],
    matched_group_id: str,
    shared_prefix_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in REPORT_TARGETS:
        report_names, value, question = _report_spec(
            family, target, concepts, certificate
        )
        semantic = (
            SEMANTIC_ACTION_NAMES[family][value]
            if target == "final_strategy"
            else report_names[value]
        )
        report_size = len(report_names)
        report_prompt = (
            prompt
            + f"The requested diagnostic field is {question}. Return only the "
            "label corresponding to the requested field.\n"
        )
        report_certificate = {
            **certificate,
            "display_action_names": list(report_names),
            "regret_by_index": None,
            "report_question": question,
        }
        report_factors = {**factors, "report_target": target, "diagnostic_fork": True}
        rows.append(
            _row(
                family=family,
                game_id=game_id,
                split=split,
                concepts=concepts,
                task_kind="report",
                factors=report_factors,
                prompt=report_prompt,
                expected_index=value,
                expected_semantic=semantic,
                expected_value=value,
                candidate_size=report_size,
                label_namespace=f"{family}:report:{target}",
                matched_group_id=matched_group_id,
                certificate=report_certificate,
                shared_prefix_id=shared_prefix_id,
            )
        )
    return rows


def _publicity_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    count = int(config["dataset"]["publicity_games_per_split"])
    for split, index, level, surface, report in product(
        SPLITS,
        range(count),
        PUBLICITY_LEVELS,
        SURFACES,
        (False, True),
    ):
        game = _publicity_game(config, split, index)
        mapping = _choice_mapping(2, game.game_id, "publicity_threshold:action")
        prompt, certificate = _publicity_prefix(game, level, surface, mapping, report=report)
        factors = {"epistemic_level": level, "surface": surface}
        group = _stable_id(STUDY_ID, "publicity_pair", game.game_id, surface)
        prefix_id = _stable_id(STUDY_ID, "publicity_prefix", game.game_id, level, surface)
        if report:
            rows.extend(
                _report_rows(
                    family="publicity_threshold",
                    game_id=game.game_id,
                    split=split,
                    concepts=game.concepts,
                    factors=factors,
                    prompt=prompt,
                    certificate=certificate,
                    matched_group_id=group,
                    shared_prefix_id=prefix_id,
                )
            )
        else:
            expected = game.a_action(level)
            rows.append(
                _row(
                    family="publicity_threshold",
                    game_id=game.game_id,
                    split=split,
                    concepts=game.concepts,
                    task_kind="action",
                    factors=factors,
                    prompt=(
                        prompt + "Choose A's simultaneous action using only A's information "
                        "and the stated B policy. "
                    ),
                    expected_index=expected,
                    expected_semantic=("COMMIT" if expected == 0 else "SAFE"),
                    expected_value=expected,
                    candidate_size=2,
                    label_namespace="publicity_threshold:action",
                    matched_group_id=group,
                    certificate=certificate,
                    shared_prefix_id=prefix_id,
                )
            )
    return rows


def _commitment_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    count = int(config["dataset"]["commitment_games_per_split"])
    for split, index, modality, surface, report in product(
        SPLITS,
        range(count),
        COMMITMENT_MODALITIES,
        SURFACES,
        (False, True),
    ):
        game = _commitment_game(config, split, index)
        mapping = _choice_mapping(2, game.game_id, "credible_commitment:action")
        prompt, certificate = _commitment_prefix(
            game, modality, surface, mapping, report=report
        )
        factors = {"commitment_modality": modality, "surface": surface}
        group = _stable_id(STUDY_ID, "commitment_pair", game.game_id, surface)
        prefix_id = _stable_id(STUDY_ID, "commitment_prefix", game.game_id, modality, surface)
        if report:
            rows.extend(
                _report_rows(
                    family="credible_commitment",
                    game_id=game.game_id,
                    split=split,
                    concepts=("NO_THREAT", "THREAT_PRESENT"),
                    factors=factors,
                    prompt=prompt,
                    certificate=certificate,
                    matched_group_id=group,
                    shared_prefix_id=prefix_id,
                )
            )
        else:
            expected = game.a_action(modality)
            rows.append(
                _row(
                    family="credible_commitment",
                    game_id=game.game_id,
                    split=split,
                    concepts=("NO_THREAT", "THREAT_PRESENT"),
                    task_kind="action",
                    factors=factors,
                    prompt=prompt + "Choose A's first-stage strategy. ",
                    expected_index=expected,
                    expected_semantic=("USE_COMMITMENT_CHANNEL" if expected == 0 else "WAIT"),
                    expected_value=expected,
                    candidate_size=2,
                    label_namespace="credible_commitment:action",
                    matched_group_id=group,
                    certificate=certificate,
                    shared_prefix_id=prefix_id,
                )
            )
    return rows


def _acquisition_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    count = int(config["dataset"]["acquisition_games_per_split"])
    for split, index, uncertainty, surface, report in product(
        SPLITS,
        range(count),
        ACQUISITION_UNCERTAINTIES,
        SURFACES,
        (False, True),
    ):
        game = _acquisition_game(config, split, index, uncertainty)
        mapping = _choice_mapping(4, game.game_id, "information_acquisition:action")
        prompt, certificate = _acquisition_prefix(game, surface, mapping, report=report)
        factors = {"uncertainty_type": game.uncertainty, "surface": surface}
        group = _stable_id(STUDY_ID, "acquisition_pair", game.game_id, surface)
        prefix_id = _stable_id(
            STUDY_ID, "acquisition_prefix", game.game_id, uncertainty, surface
        )
        if report:
            rows.extend(
                _report_rows(
                    family="information_acquisition",
                    game_id=game.game_id,
                    split=split,
                    concepts=game.concepts,
                    factors=factors,
                    prompt=prompt,
                    certificate=certificate,
                    matched_group_id=group,
                    shared_prefix_id=prefix_id,
                )
            )
        else:
            expected, _best, values = game.optimal_query()
            certificate = {
                **certificate,
                "regret_by_index": [max(values.values()) - values.get(i, 0) for i in range(4)],
            }
            prompt = prompt + "Choose the information-acquisition option now. "
            rows.append(
                _row(
                    family="information_acquisition",
                    game_id=game.game_id,
                    split=split,
                    concepts=game.concepts,
                    task_kind="action",
                    factors=factors,
                    prompt=prompt,
                    expected_index=expected,
                    expected_semantic=(
                        ("ACT_NOW", "INSPECT_WORLD", "INSPECT_B_BELIEF", "INSPECT_B_POLICY")[
                            expected
                        ]
                    ),
                    expected_value=expected,
                    candidate_size=4,
                    label_namespace="information_acquisition:action",
                    matched_group_id=group,
                    certificate=certificate,
                    shared_prefix_id=prefix_id,
                )
            )
    return rows


def _family_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        _publicity_rows(config) + _commitment_rows(config) + _acquisition_rows(config),
        key=lambda row: row["condition_id"],
    )


def expected_row_count(config: dict[str, Any]) -> int:
    per_family = (
        int(config["dataset"]["publicity_games_per_split"]) * len(PUBLICITY_LEVELS)
        + int(config["dataset"]["commitment_games_per_split"]) * len(COMMITMENT_MODALITIES)
        + int(config["dataset"]["acquisition_games_per_split"]) * len(ACQUISITION_UNCERTAINTIES)
    )
    return len(SPLITS) * len(SURFACES) * per_family * (1 + len(REPORT_TARGETS))


def dataset_payload(config: dict[str, Any]) -> dict[str, Any]:
    rows = _family_rows(config)
    body = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "status": "generated_before_model_execution",
        "config_sha256": canonical_sha256(config),
        "rows": rows,
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def _contains_expected_target(row: dict[str, Any]) -> bool:
    prompt = row["prompt"]
    if row["task_kind"] == "report":
        target = row.get("report_target")
        target_markers = {
            "world_state": ("physical", "opportunity"),
            "own_information": (
                "A knows",
                "known to A",
                "A privately observes",
                "currently relevant target",
            ),
            "opponent_information": (
                "B receives",
                "B also",
                "B's belief",
                "B observes",
                "B does not observe",
                "whether B has",
                "what B knows",
            ),
            "opponent_belief": (
                "B knows A",
                "B's belief",
                "B's response equals",
                "whether A knows B's belief",
            ),
            "opponent_policy": (
                "B follows",
                "B uses",
                "B's policy",
                "B stays out",
                "NO_RESPONSE",
            ),
            "predicted_opponent_action": (
                "B chooses",
                "B produces",
                "B's response",
                "B takes no response",
                "determine B's response",
                "action B will take",
            ),
            "final_strategy": (
                "Choose",
                "maximize",
                "highest exact",
                "optimal information",
            ),
        }
        markers = target_markers.get(str(target), ())
        return (
            "requested diagnostic field" in prompt
            and "Options:" in prompt
            and "Execution mapping:" in prompt
            and any(marker.lower() in prompt.lower() for marker in markers)
            and "Auditor ledger" not in prompt
        )
    family = row["experiment_family"]
    required = {
        "publicity_threshold": (
            "SAFE gives A and B",
            "B's policy is REQUIRE_BOTH_AGENTS_TO_KNOW",
            "simultaneously with B",
            "Options:",
        ),
        "credible_commitment": (
            "two-stage entry game",
            "commitment channel has no immediate payoff",
            "B stays out exactly",
            "Options:",
        ),
        "information_acquisition": (
            "inspection cost is 1 point",
            "perfectly reveals only its named target",
            "highest exact expected final payoff",
            "Options:",
        ),
    }[family]
    return "Execution mapping:" in prompt and all(
        marker.lower() in prompt.lower() for marker in required
    )


def _pair_groups(
    rows: list[dict[str, Any]], *keys: str
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[key] for key in keys)
        groups.setdefault(key, []).append(row)
    return groups


def control_audit(payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    rows = payload["rows"]
    failures: list[str] = []
    if payload.get("study_id") != STUDY_ID:
        failures.append("wrong study ID")
    if payload.get("config_sha256") != canonical_sha256(config):
        failures.append("config hash mismatch")
    if len(rows) != expected_row_count(config):
        failures.append(f"row count {len(rows)} != {expected_row_count(config)}")
    if {row["experiment_family"] for row in rows} != set(FAMILIES):
        failures.append("family set mismatch")
    for row in rows:
        mapping = row["choice_mapping"]
        labels = row["candidate_labels"]
        if set(mapping) != set(labels) or sorted(mapping.values()) != list(range(len(labels))):
            failures.append(f"invalid mapping: {row['condition_id']}")
        if not _contains_expected_target(row):
            failures.append(f"target inputs missing: {row['condition_id']}")
        if row["expected_choice"] != _expected_choice(mapping, row["expected_index"]):
            failures.append(f"expected choice mismatch: {row['condition_id']}")
    action_rows = [row for row in rows if row["task_kind"] == "action"]
    if {row["surface"] for row in action_rows} != set(SURFACES):
        failures.append("surface controls incomplete")
    for group in _pair_groups(action_rows, "experiment_family", "game_id", "surface").values():
        mappings = {json.dumps(row["choice_mapping"], sort_keys=True) for row in group}
        if len(mappings) != 1:
            failures.append(f"treatment changed label mapping: {group[0]['matched_group_id']}")
        if group and group[0]["experiment_family"] == "publicity_threshold":
            levels = {row["condition_factors"]["epistemic_level"] for row in group}
            if levels != set(PUBLICITY_LEVELS):
                failures.append("publicity levels incomplete")
            by_level = {row["condition_factors"]["epistemic_level"]: row for row in group}
            if by_level.get("shared_private", {}).get("expected_index") != 1:
                failures.append("shared-private publicity certificate is not SAFE")
            if by_level.get("public_announcement", {}).get("expected_index") != 0:
                failures.append("public publicity certificate is not COMMIT")
            if by_level.get("common_knowledge", {}).get("expected_index") != 0:
                failures.append("common-knowledge publicity certificate is not COMMIT")
        if group and group[0]["experiment_family"] == "credible_commitment":
            modalities = {row["condition_factors"]["commitment_modality"] for row in group}
            if modalities != set(COMMITMENT_MODALITIES):
                failures.append("commitment modalities incomplete")
            by_modality = {
                row["condition_factors"]["commitment_modality"]: row for row in group
            }
            if by_modality.get("public_irreversible", {}).get("expected_index") != 0:
                failures.append("public commitment certificate is not USE")
            if by_modality.get("public_tool", {}).get("expected_index") != 0:
                failures.append("public tool certificate is not USE")
            if by_modality.get("private_irreversible", {}).get("expected_index") != 1:
                failures.append("private commitment certificate is not WAIT")
            if by_modality.get("cheap_promise", {}).get("expected_index") != 1:
                failures.append("promise certificate is not WAIT")
        if group and group[0]["experiment_family"] == "information_acquisition":
            uncertainties = {row["condition_factors"]["uncertainty_type"] for row in group}
            if not uncertainties.issubset(set(ACQUISITION_UNCERTAINTIES)):
                failures.append("unknown acquisition uncertainty")
    for group in _pair_groups(
        [row for row in action_rows if row["experiment_family"] == "information_acquisition"],
        "split",
        "surface",
    ).values():
        uncertainties = {row["condition_factors"]["uncertainty_type"] for row in group}
        if uncertainties != set(ACQUISITION_UNCERTAINTIES):
            failures.append("acquisition uncertainties incomplete")
        by_uncertainty = {
            row["condition_factors"]["uncertainty_type"]: row for row in group
        }
        expected_queries = {
            "world_state": 1,
            "world_state_known": 0,
            "opponent_belief": 2,
            "opponent_belief_known": 0,
            "opponent_policy": 3,
            "opponent_policy_known": 0,
        }
        if any(
            by_uncertainty.get(key, {}).get("expected_index") != value
            for key, value in expected_queries.items()
        ):
            failures.append("acquisition target certificates are not identifying")
    # Reports are separate static forks, never sampled assistant continuations.
    for group in _pair_groups(rows, "shared_prefix_id").values():
        if {row["task_kind"] for row in group} != {"action", "report"}:
            failures.append(f"missing direct/report fork: {group[0]['shared_prefix_id']}")
        if any(row.get("trajectory_condition") for row in group):
            failures.append(f"trajectory contamination: {group[0]['shared_prefix_id']}")
    return {
        "status": "cpu_semantic_controls_passed" if not failures else "failed",
        "passed": not failures,
        "row_count": len(rows),
        "expected_row_count": expected_row_count(config),
        "failure_count": len(failures),
        "failures": failures[:50],
        "gates": {
            "dataset_hash": payload.get("content_sha256")
            == canonical_sha256(
                {key: value for key, value in payload.items() if key != "content_sha256"}
            ),
            "semantic_prompt_contract": not failures,
            "surface_controls": {row["surface"] for row in action_rows} == set(SURFACES),
            "static_diagnostic_forks": not any(row.get("trajectory_condition") for row in rows),
        },
    }


def verify_dataset_payload(payload: dict[str, Any], config: dict[str, Any]) -> None:
    _require(payload.get("schema_version") == SCHEMA_VERSION, "dataset schema changed")
    _require(payload.get("study_id") == STUDY_ID, "dataset study ID changed")
    _require(
        payload.get("status") == "generated_before_model_execution", "dataset is not frozen"
    )
    _require(
        payload.get("config_sha256") == canonical_sha256(config), "dataset config mismatch"
    )
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    _require(payload.get("content_sha256") == canonical_sha256(body), "dataset hash mismatch")
    _require(
        len(payload.get("rows", [])) == expected_row_count(config), "dataset row count mismatch"
    )
    audit = control_audit(payload, config)
    _require(audit["passed"], f"dataset semantic audit failed: {audit['failures'][:3]}")


def trajectory_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    return [{"role": "user", "content": row["prompt"]}]


def result_summary(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    by_family: dict[str, dict[str, Any]] = {}
    for family in FAMILIES:
        selected = [
            row
            for row in records
            if row["experiment_family"] == family and row["task_kind"] == "action"
        ]
        by_family[family] = {
            "n_action_rows": len(selected),
            "accuracy": sum(bool(row.get("correct")) for row in selected) / len(selected)
            if selected
            else 0.0,
        }
    return {
        "study_id": STUDY_ID,
        "locked_action_rows": sum(
            row["task_kind"] == "action" and row["split"] == "locked" for row in records
        ),
        "family_action_summary": by_family,
        "report_rows": sum(row["task_kind"] == "report" for row in records),
    }


def commitment_tool_transition(modality: str, used: bool) -> dict[str, Any]:
    """Model-independent future-option transition for the later tool stage."""

    _require(modality in COMMITMENT_MODALITIES, f"unknown modality {modality}")
    if not used:
        return {"future_options": ["X", "Y"], "publicly_observed": False, "binding": False}
    binding = modality in {"public_irreversible", "private_irreversible", "public_tool"}
    observed = modality in {"public_irreversible", "public_tool"}
    return {
        "future_options": ["X"] if binding else ["X", "Y"],
        "publicly_observed": observed,
        "binding": binding,
    }
