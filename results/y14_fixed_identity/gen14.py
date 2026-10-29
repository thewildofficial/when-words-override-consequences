import sys, json
sys.path.insert(0, "/tmp/wwoc/src"); sys.path.insert(0, "/tmp/wwoc/results/y12_tool_balanced")
import gen12
from jspace_policy.inverse_evidence_games import _make_row
def build():
    out = []
    for b in range(328, 360):
        for hist in ("none", "redundant"):
            for mp in ("prose", "table"):
                for cond in ("fixed_worded_desc", "fixed_blank_desc"):
                    r = _make_row(b, "strategic", "opposed", "assertion", hist, mp)
                    idx_of = {lab: int(i) for i, lab in r["message_labels"].items()}
                    for swap in (0, 1):
                        tools, user, names, order = gen12.spec(r, "desc_named", swap)
                        # fixed identity: tool_alpha is bound to concept index 0 of the game, tool_beta to index 1, in both list orders
                        new = {lab: ["tool_alpha", "tool_beta"][idx_of[lab]] for lab in order}
                        for lab in order: user = user.replace(names[lab], new[lab])
                        for t, lab in zip(tools, order):
                            t["function"]["name"] = new[lab]
                            if cond == "fixed_blank_desc": t["function"]["description"] = f"Call {new[lab]}."
                        names = new; exp = r["expected_action"]
                        out.append(dict(fmt="tool_" + cond, base=r["base_game_id"], history=hist, mapping=mp, swap=swap, expected_action=exp, expected_pos=order.index(exp), gold_tool=names[exp], target=r["target_response"], response_per_action=r["response_per_action"], tools=tools, user=user, names=names, max_new_tokens=400))
    return out
if __name__ == "__main__":
    o = build(); json.dump(o, open("/tmp/wwoc/results/y14_fixed_identity/y14_rows.json", "w"))
    from collections import Counter; print(len(o), Counter((x["fmt"], x["expected_pos"]) for x in o), Counter((x["fmt"], x["gold_tool"]) for x in o))
    for x in [o[0], o[1], [r for r in o if r["fmt"] == "tool_fixed_blank_desc"][0]]: print(x["user"][:600]); print([(t["function"]["name"], t["function"]["description"]) for t in x["tools"]], x["gold_tool"], x["expected_pos"]); print('--')
