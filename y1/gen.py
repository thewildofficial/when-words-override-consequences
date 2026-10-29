"""Y1 competence battery: no-CoT single-token game decisions with exact EV ground truth."""
import json, random, itertools
def fmt(q, a, b, rnd):
    flip = rnd.random() < 0.5
    first, second = (b, a) if flip else (a, b)
    text = f"{q}\n\nA) {first}\nB) {second}\n\nAnswer with a single letter."
    return text, ("A" if not flip else "B")  # letter of option `a` (the correct one)
def pot_odds_stated(rnd):
    while True:
        P=rnd.choice([40,60,80,100,120,150,200]); B=rnd.choice([10,20,30,40,50,60,80,100]); p=rnd.randrange(8,60)/100
        thr=B/(P+B)
        if abs(p-thr)>=0.07: break
    ev=p*P-(1-p)*B
    q=f"Poker. The pot is ${P}. Your opponent bets ${B} (you must put in ${B} to call). You know you will win the pot {int(p*100)}% of the time if you call. Should you call or fold to maximise expected value?"
    return q,("Call" if ev>0 else "Fold"),("Fold" if ev>0 else "Call"),dict(P=P,B=B,p=p,ev=ev)
def outs_draw(rnd):
    while True:
        outs=rnd.choice([2,3,4,6,8,9,10,12,15]); unseen=rnd.choice([46,47]); P=rnd.choice([40,60,80,100,120,150,200]); B=rnd.choice([10,20,30,40,50,60,80,100])
        eq=outs/unseen; thr=B/(P+B)
        if abs(eq-thr)>=0.05: break
    ev=eq*P-(1-eq)*B
    q=f"Poker, Texas Hold'em, one card to come (the river). You have a draw with {outs} outs among the {unseen} unseen cards; any out wins the pot for you and no other card does. The pot is ${P} and your opponent bets ${B} (you must put in ${B} to call). Should you call or fold to maximise expected value?"
    return q,("Call" if ev>0 else "Fold"),("Fold" if ev>0 else "Call"),dict(outs=outs,unseen=unseen,P=P,B=B,eq=eq,ev=ev)
def gamble(rnd):
    while True:
        x=rnd.choice([20,30,50,80,100,150,200]); p=rnd.randrange(10,90)//5*5/100; y=rnd.randrange(5,int(x*0.9)); 
        if abs(p*x-y)>=0.12*x: break
    q=f"Choose the option with the higher expected value in dollars. Option 1 is a sure ${y}. Option 2 pays ${x} with probability {int(p*100)}% and $0 otherwise."
    return q,("Option 2" if p*x>y else "Option 1"),("Option 1" if p*x>y else "Option 2"),dict(x=x,p=p,y=y)
def bj(rnd):
    # infinite deck exact EV: hit-once-then-stand vs stand? use full optimal: compute EV(stand) and EV(hit then optimal)
    import functools
    probs={v:(4/13 if v==10 else 1/13) for v in range(1,11)}
    def dealer_dist(up):
        @functools.lru_cache(None)
        def go(tot,soft):
            t=tot+(10 if soft and tot+10<=21 else 0)
            if t>=17: return {t:1.0} if t<=21 else {'bust':1.0}
            out={}
            for v,pv in probs.items():
                nt=tot+v; ns=soft or v==1
                for k,w in go(nt,ns).items(): out[k]=out.get(k,0)+pv*w
            return out
        return go(up,up==1)
    def stand_ev(tot,soft,up):
        t=tot+(10 if soft and tot+10<=21 else 0); d=dealer_dist(up); e=0
        for k,w in d.items():
            e+= w*(1 if k=='bust' or t>k else (0 if t==k else -1))
        return e
    @functools.lru_cache(None)
    def best(tot,soft,up):
        t=tot+(10 if soft and tot+10<=21 else 0)
        if tot>21: return -1.0
        s=stand_ev(tot,soft,up); h=0
        for v,pv in probs.items(): h+=pv*best(tot+v,soft or v==1,up)
        return max(s,h)
    while True:
        cards=[rnd.randrange(2,11),rnd.randrange(2,11)]; up=rnd.randrange(2,11)
        tot=sum(cards)
        if tot>=20 or tot<=8: continue
        s=stand_ev(tot,False,up); h=sum(pv*best(tot+v,v==1,up) for v,pv in probs.items())
        if abs(s-h)>=0.08: break
    q=f"Blackjack (infinite deck, dealer stands on all 17s). Your hand: {cards[0]} and {cards[1]} (total {tot}, no aces). Dealer shows {up}. Choose the action with the higher expected value."
    return q,("Hit" if h>s else "Stand"),("Stand" if h>s else "Hit"),dict(cards=cards,up=up,ev_stand=s,ev_hit=h)
TASKS=dict(pot_odds_stated=pot_odds_stated,outs_draw=outs_draw,gamble=gamble,blackjack=bj)
def build(n=100,seed=7):
    rnd=random.Random(seed); rows=[]
    for name,fn in TASKS.items():
        for i in range(n):
            q,a,b,meta=fn(rnd); text,letter=fmt(q,a,b,rnd)
            rows.append(dict(task=name,id=f"{name}-{i}",prompt=text,gold_letter=letter,correct=a,meta=meta))
    return rows
if __name__=="__main__":
    rows=build(); json.dump(rows,open("y1_rows.json","w"))
    from collections import Counter
    print(len(rows),Counter((r["task"],r["gold_letter"]) for r in rows)); print(rows[0]["prompt"]); print(rows[100]["prompt"])
