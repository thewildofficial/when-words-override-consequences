import json, random
from gen import fmt
from gen2 import items
def build(n=120,seed=23):
    rnd=random.Random(seed); its=[m for m in items(rnd,n*2) if abs(round(100*m["eq"])-round(100*m["thr"]))>=4]
    # rebalance
    c=[m for m in its if m["gold"]=="Call"][:n//2]; f=[m for m in its if m["gold"]=="Fold"][:n//2]; its=c+f; rnd.shuffle(its)
    rows=[]
    for i,m in enumerate(its):
        o,u,P,B=m["outs"],m["unseen"],m["P"],m["B"]; eqp=round(100*m["eq"]); thrp=round(100*m["thr"]); g=m["gold"]; w="Fold" if g=="Call" else "Call"
        base="Poker, Texas Hold'em, one card to come (the river). "
        ask=" Should you call or fold to maximise expected value?"
        qs={
         "D1_both_stated":base+f"You will win the pot about {eqp}% of the time if you call, and calling breaks even at a win rate of {thrp}%."+ask,
         "D2_equity_stated_thr_derived":base+f"The pot is ${P} and your opponent bets ${B} (you must put in ${B} to call). You will win the pot about {eqp}% of the time if you call."+ask,
         "D3_equity_derived_thr_stated":base+f"You have a draw with {o} outs among the {u} unseen cards; any out wins the pot and no other card does. Calling breaks even at a win rate of {thrp}%."+ask,
         "D4_both_derived":base+f"You have a draw with {o} outs among the {u} unseen cards; any out wins the pot for you and no other card does. The pot is ${P} and your opponent bets ${B} (you must put in ${B} to call)."+ask,
        }
        for k,q in qs.items():
            t,l=fmt(q,g,w,rnd); rows.append(dict(task=k,id=f"{k}-{i}",prompt=t,gold_letter=l,correct=g,meta=m))
    return rows
if __name__=="__main__":
    r=build(); json.dump(r,open("y3_rows.json","w")); from collections import Counter
    print(len(r),Counter((x["task"],x["correct"]) for x in r)); print(r[0]["prompt"])
