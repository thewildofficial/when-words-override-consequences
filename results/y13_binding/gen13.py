import sys, json
sys.path.insert(0, "/tmp/wwoc/src"); sys.path.insert(0, "/tmp/wwoc/results/y12_tool_balanced")
import gen12
from jspace_policy.inverse_evidence_games import _make_row
GENERIC = "\nBefore calling, write one sentence restating the receiver policy and the payoff rule in your own words, then call the tool you choose."
RESTATE = "\nBefore calling, write one sentence stating which tool leads the receiver to the target response, then call that tool."
def build():
    out = []
    for b in range(296, 328):
        for hist in ("none", "redundant"):
            for mp in ("prose", "table"):
                for cond in ("desc", "desc_restate", "desc_generic", "optok_worded"):
                    r = _make_row(b, "strategic", "opposed", "assertion", hist, mp)
                    for swap in (0, 1):
                        tools, user, names, order = gen12.spec(r, "desc", swap)
                        if cond == "optok_worded":
                            idx_of = {lab: int(i) for i, lab in r["message_labels"].items()}
                            new = {lab: "send_" + r["tokens"][idx_of[lab]].lower() for lab in order}
                            for lab in order:
                                user = user.replace(names[lab], new[lab])
                            for t, lab in zip(tools, order): t["function"]["name"] = new[lab]
                            names = new
                        if cond == "desc_restate": user = user + RESTATE
                        if cond == "desc_generic": user = user + GENERIC
                        exp = r["expected_action"]
                        out.append(dict(fmt="tool_" + cond, base=r["base_game_id"], history=hist, mapping=mp, swap=swap, expected_action=exp, expected_pos=order.index(exp), target=r["target_response"], response_per_action=r["response_per_action"], tools=tools, user=user, names=names, max_new_tokens=400))
    return out
if __name__ == "__main__":
    o = build(); json.dump(o, open("/tmp/wwoc/results/y13_binding/y13_rows.json", "w"))
    from collections import Counter; print(len(o), Counter((x["fmt"], x["expected_pos"]) for x in o))
    for c in ("tool_desc_restate", "tool_optok_worded"):
        x = [r for r in o if r["fmt"] == c and r["mapping"] == "prose"][0]; print(x["user"]); print([t["function"] for t in x["tools"]], x["expected_pos"])
