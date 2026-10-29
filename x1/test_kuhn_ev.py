from fractions import Fraction as F
import kuhn_ev as k
def test_nash_value():
    # Known: Kuhn game value for player 2 is +1/18 at equilibrium (alpha in [0,1/3]); a=1/4 policy for P1.
    pol = k.POLICIES["nash_a025"]; tot = F(0)
    for c2 in k.CARDS:
        for obs in ("bet", "check"):
            q = k.action_values(c2, obs, pol)
            if q is None: continue
            # P2 best response expected value, weighted by P(c2) * P(obs | c2)
            pobs = sum(F(1, 2) * (pol["b"][c1] if obs == "bet" else 1 - pol["b"][c1]) for c1 in k.CARDS if c1 != c2)
            tot += F(1, 3) * pobs * max(q.values())
    assert tot == F(1, 18), tot
def test_posterior_sums_to_one():
    for pol in k.POLICIES.values():
        for c2 in k.CARDS:
            for obs in ("bet", "check"):
                p = k.posterior(c2, obs, pol)
                if p: assert sum(p.values()) == 1
def test_fold_is_minus_one_and_king_calls():
    q = k.action_values(3, "bet", k.POLICIES["nash_a025"])
    assert q["fold"] == -1 and q["call"] > 0
def test_label_consistency():
    for r in k.dataset():
        assert r["best_label"] in ("A", "B") and r["labels"][r["best_label"]] == r["best"]
def test_prompt_no_opponent_card_leak():
    for r in k.dataset()[:50]:
        assert "opponent holds" not in r["prompt"].lower()
