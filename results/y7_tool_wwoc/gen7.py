"""Y7: WWOC contrarian game rendered as (a) the frozen RBG-4 A/B prompt and (b) tool calls (name-carries-words / description-only / opaque). Uses the frozen RBG-4 row generator with FRESH base games."""
import sys, json, re
sys.path.insert(0, "/tmp/wwoc/src")
from jspace_policy.inverse_evidence_games import _make_row, CONCEPT_PAIRS
BASES = range(200, 232)  # fresh (RBG-4 dataset used base games < 24; check config)
def tool_spec(row, variant):
    """Return (tools list, user text, {label: tool_name}) for a row; actions A/B map to the tool endpoints."""
    ml = row["message_labels"]  # semantic index -> label
    surf = row["message_surfaces"]  # label -> surface text
    names = {}; tools = []
    for label in ("A", "B"):
        idx = [i for i, l in ml.items() if l == label][0]
        if variant == "opaque":
            nm = "send_" + row["tokens"][int(idx)].lower(); desc = f"Send opaque token {row['tokens'][int(idx)]}."
        elif variant == "assertion_name":
            nm = "claim_state_" + row["concepts"][int(idx)].lower(); desc = f"Claim that the state is {row['concepts'][int(idx)]}."
        elif variant == "assertion_desc":
            nm = f"action_{label.lower()}"; desc = f"Claim that the state is {row['concepts'][int(idx)]}."
        names[label] = nm
        tools.append({"type": "function", "function": {"name": nm, "description": desc, "parameters": {"type": "object", "properties": {}, "required": []}}})
    # text: replace the "Available actions" block with a tool pointer, and rewrite option references (A:/B:) in table/prose to tool names
    sc = row["scenario"]
    head = sc.split("\n\nAvailable actions:")[0]
    for label in ("A", "B"):
        s = surf[label]
        head = head.replace(f"Option {label} ({s})", f"Tool {names[label]} ({s})")
    payoff = sc.split("\n\n")[-1]
    user = f"{head}\n\nYour available actions are the two tools provided.\n\n{payoff}\nCall the single payoff-maximizing tool."
    return tools, user, names
def build():
    rows = []
    for b in BASES:
        for hist in ("none", "redundant"):
            for mp in ("prose", "table"):
                for surf in ("assertion", "opaque_token"):
                    r = _make_row(b, "strategic", "opposed", surf, hist, mp); rows.append(("ab", surf, r))
                for variant in ("assertion_name", "assertion_desc", "opaque"):
                    base_surf = "opaque_token" if variant == "opaque" else "assertion"
                    r = _make_row(b, "strategic", "opposed", base_surf, hist, mp); rows.append(("tool_" + variant, base_surf, r))
    out = []
    for fmt, surf, r in rows:
        d = dict(fmt=fmt, base=r["base_game_id"], history=r["history"], mapping=r["mapping_format"], surface=surf, expected_action=r["expected_action"], target=r["target_response"], response_per_action=r["response_per_action"])
        if fmt == "ab":
            d["prompt"] = r["prompt"]
        else:
            tools, user, names = tool_spec(r, fmt[5:]); d.update(tools=tools, user=user, names=names, expected_tool=names[r["expected_action"]])
        out.append(d)
    return out
if __name__ == "__main__":
    o = build(); json.dump(o, open("/tmp/wwoc/y7/y7_rows.json", "w"))
    from collections import Counter; print(len(o), Counter(x["fmt"] for x in o))
    t = [x for x in o if x["fmt"] == "tool_assertion_name" and x["history"] == "none" and x["mapping"] == "table"][0]
    print(t["user"]); print(json.dumps(t["tools"])[:400]); print(t["names"], t["expected_tool"])
