import json, random
from gen import fmt
def items(rnd, n):
    out=[]; want={"Call":n//2,"Fold":n-n//2}
    while sum(want.values())>0:
        outs=rnd.choice([2,3,4,6,8,9,10,12,15]); unseen=rnd.choice([46,47]); P=rnd.choice([40,60,80,100,120,150,200]); B=rnd.choice([10,20,30,40,50,60,80,100])
        eq=outs/unseen; thr=B/(P+B)
        if abs(eq-thr)<0.05: continue
        g="Call" if eq>thr else "Fold"
        if want[g]<=0: continue
        want[g]-=1; out.append(dict(outs=outs,unseen=unseen,P=P,B=B,eq=eq,thr=thr,gold=g))
    return out
def build(n=120,seed=11):
    rnd=random.Random(seed); its=items(rnd,n); rows=[]
    for i,m in enumerate(its):
        o,u,P,B=m["outs"],m["unseen"],m["P"],m["B"]; eqp=round(100*m["eq"])
        qs={
         "C1_poker_outs":(f"Poker, Texas Hold'em, one card to come (the river). You have a draw with {o} outs among the {u} unseen cards; any out wins the pot for you and no other card does. The pot is ${P} and your opponent bets ${B} (you must put in ${B} to call). Should you call or fold to maximise expected value?",m["gold"],"Fold" if m["gold"]=="Call" else "Call"),
         "C2_urn_outs":(f"A game: a box holds {u} balls, {o} of them gold. You will draw one ball at random; if it is gold you win the prize pool of ${P}, otherwise you lose the ${B} entry stake you must pay to play. Should you play or pass to maximise expected value?",
                        "Play" if m["gold"]=="Call" else "Pass","Pass" if m["gold"]=="Call" else "Play"),
         "C3_poker_stated":(f"Poker. The pot is ${P}. Your opponent bets ${B} (you must put in ${B} to call). You will win the pot about {eqp}% of the time if you call. Should you call or fold to maximise expected value?",m["gold"],"Fold" if m["gold"]=="Call" else "Call"),
        }
        for k,(q,a,b) in qs.items():
            t,l=fmt(q,a,b,rnd); rows.append(dict(task=k,id=f"{k}-{i}",prompt=t,gold_letter=l,correct=a,meta=m))
        # C4: derivation only: is hit probability above the bet-fraction threshold (given as percent) ?
        thrp=round(100*m["thr"])
        if abs(100*m["eq"]-thrp)>=3:
            q=f"A box holds {u} balls, {o} of them gold. You draw one ball at random. Is the probability of drawing a gold ball higher or lower than {thrp}%?"
            hi=m["eq"]*100>thrp
            t,l=fmt(q,"Higher" if hi else "Lower","Lower" if hi else "Higher",rnd)
            rows.append(dict(task="C4_prob_vs_threshold",id=f"C4-{i}",prompt=t,gold_letter=l,correct="Higher" if hi else "Lower",meta=m))
    return rows
if __name__=="__main__":
    r=build(); json.dump(r,open("y2_rows.json","w"))
    from collections import Counter; print(Counter(x["task"] for x in r), Counter((x["task"],x["correct"]) for x in r))
    print(r[1]["prompt"])
