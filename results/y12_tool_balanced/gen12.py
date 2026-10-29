"""Y12: label-balanced tool-WWOC. FRESH games 264-295. Each (game,history,mapping,condition) is run in BOTH tool-list orders (expected tool first / second)."""
import sys, json
sys.path.insert(0, "/tmp/wwoc/src")
from jspace_policy.inverse_evidence_games import _make_row
def spec(r, variant, swap):
    ml = r["message_labels"]; surf = r["message_surfaces"]
    idx_of = {lab: int(i) for i, lab in ml.items()}
    order = ["A", "B"] if not swap else ["B", "A"]  # tool-list order
    names = {}
    for pos, lab in enumerate(order):
        i = idx_of[lab]
        if variant == "opaque": names[lab] = "send_" + r["tokens"][i].lower()
        elif variant == "assertion_name": names[lab] = "claim_state_" + r["concepts"][i].lower()
        else: names[lab] = ["tool_one", "tool_two"][pos]
    tools = []
    for lab in order:
        i = idx_of[lab]
        d = f"Send opaque token {r['tokens'][i]}." if variant == "opaque" else f"Claim that the state is {r['concepts'][i]}."
        tools.append({"type": "function", "function": {"name": names[lab], "description": d, "parameters": {"type": "object", "properties": {}, "required": []}}})
    sc = r["scenario"]; head = sc.split("\n\nAvailable actions:")[0]
    for lab in ("A", "B"):
        s = surf[lab]
        head = head.replace(f"Option {lab} ({s})", f"Tool {names[lab]} ({s})")
        if variant == "desc_named":
            head = head.replace(f"if {s},", f"if you call {names[lab]} ({s}),")
    payoff = sc.split("\n\n")[-1]
    user = f"{head}\n\nYour available actions are the two tools provided.\n\n{payoff}\nCall the single payoff-maximizing tool."
    return tools, user, names, order
def build():
    out = []
    for b in range(264, 296):
        for hist in ("none", "redundant"):
            for mp in ("prose", "table"):
                for variant in ("opaque", "assertion_name", "desc", "desc_named"):
                    r = _make_row(b, "strategic", "opposed", "opaque_token" if variant == "opaque" else "assertion", hist, mp)
                    for swap in (0, 1):
                        tools, user, names, order = spec(r, variant, swap)
                        exp = r["expected_action"]
                        out.append(dict(fmt="tool_" + variant, base=r["base_game_id"], history=hist, mapping=mp, swap=swap, expected_action=exp, expected_pos=order.index(exp), target=r["target_response"], response_per_action=r["response_per_action"], tools=tools, user=user, names=names, max_new_tokens=400))
    return out
if __name__ == "__main__":
    o = build(); json.dump(o, open("/tmp/wwoc/results/y12_tool_balanced/y12_rows.json", "w"))
    from collections import Counter
    print(len(o), Counter((x["fmt"], x["expected_pos"]) for x in o))
    x = [r for r in o if r["fmt"] == "tool_desc_named" and r["mapping"] == "prose" and r["swap"] == 1][0]; print(x["user"]); print([t["function"]["name"] for t in x["tools"]], x["expected_pos"])
