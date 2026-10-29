"""Y11: confirmatory tool-WWOC, FRESH games 232-263, uniform 400-token cap for all rows."""
import sys, json
sys.path.insert(0, "/tmp/wwoc/src"); sys.path.insert(0, "/tmp/wwoc/results/y7_tool_wwoc")
import gen7
from jspace_policy.inverse_evidence_games import _make_row
def build():
    out = []
    for b in range(232, 264):
        for hist in ("none", "redundant"):
            for mp in ("prose", "table"):
                for variant in ("opaque", "assertion_name", "assertion_desc", "desc_named"):
                    r = _make_row(b, "strategic", "opposed", "opaque_token" if variant == "opaque" else "assertion", hist, mp)
                    v = "assertion_desc" if variant == "desc_named" else variant
                    tools, user, names = gen7.tool_spec(r, v)
                    if variant == "desc_named" and mp == "prose":
                        for lab in ("A", "B"):
                            s = r["message_surfaces"][lab]
                            user = user.replace(f"if {s},", f"if you call {names[lab]} ({s}),")
                    out.append(dict(fmt="tool_" + variant, base=r["base_game_id"], history=hist, mapping=mp, expected_action=r["expected_action"], target=r["target_response"], response_per_action=r["response_per_action"], tools=tools, user=user, names=names, max_new_tokens=400))
    return out
if __name__ == "__main__":
    o = build(); json.dump(o, open("/tmp/wwoc/results/y11_tool_uniform/y11_rows.json", "w"))
    from collections import Counter; print(len(o), Counter(x["fmt"] for x in o))
    x = [r for r in o if r["fmt"] == "tool_desc_named" and r["mapping"] == "prose"][0]; print(x["user"])
