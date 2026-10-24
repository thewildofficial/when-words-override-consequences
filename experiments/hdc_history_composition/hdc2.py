"""HDC-2 (follow-up to HDC-1; exploratory, motivated by HDC-1 output - not preregistered).
Hypothesis H-default-goal: with the private-state sentence present, models treat the private true state as the default payoff target, ignoring 'opposed'.
Predictions: removing/neutralizing the state sentence raises opposed-assertion accuracy; restating the payoff does not; moving payoff before the state sentence helps little.
"""
import json,modal,sys
sys.path.insert(0,'/tmp/hdc') if False else None
import hdc
from hdc import make_row,MODELS,SYSTEM
def variants(nbase=24,K=4):
    out=[]
    for b in range(nbase):
        for inc in ("aligned","opposed"):
            base=make_row(b,inc,"assertion","balanced",K); p=base["prompt"]
            lines=p.split("\n"); si=[i for i,l in enumerate(lines) if l.startswith("The private true state is")][0]
            pay=[i for i,l in enumerate(lines) if l.startswith("You earn 10 points")][0]
            def mk(name,L): out.append(dict(base_row=base,variant=name,base=b,incentive=inc,surface="assertion",htype="balanced",K=K,expected=base["expected"],prompt="\n".join(L)))
            mk("baseline",lines)
            mk("state_removed",lines[:si]+lines[si+1:])
            mk("state_neutral_word",lines[:si]+["The private true state is ORCHID."]+lines[si+1:])
            mk("payoff_restated",lines[:pay+1]+["(Reminder: "+lines[pay]+")"]+lines[pay+1:])
            L=[l for i,l in enumerate(lines) if i not in (si,pay)]; j=[i for i,l in enumerate(L) if l.startswith("You are an informed")][0]
            mk("payoff_before_state",L[:j+1]+[lines[pay],lines[si]]+L[j+1:])
    for o in out: o.pop("base_row")
    return out
app=modal.App("hdc2"); resv=modal.Volume.from_name("hdc1-results",create_if_missing=True); cache=modal.Volume.from_name("jspace-hf-cache",create_if_missing=True)
IMG=hdc.image.add_local_python_source("hdc")
@app.function(image=IMG,gpu="A100-80GB",volumes={"/cache":cache,"/res":resv},timeout=1800)
def run(k,data):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    mid=MODELS[k]; tok=AutoTokenizer.from_pretrained(mid); tok.padding_side="left"
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    m=AutoModelForCausalLM.from_pretrained(mid,dtype=torch.bfloat16,device_map="cuda").eval()
    ia,ib=tok.encode("A",add_special_tokens=False)[0],tok.encode("B",add_special_tokens=False)[0]
    texts=[tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":d["prompt"]}],tokenize=False,add_generation_prompt=True) for d in data]
    out=[]
    for i in range(0,len(texts),24):
        enc=tok(texts[i:i+24],return_tensors="pt",padding=True,add_special_tokens=False).to("cuda")
        with torch.no_grad(): lg=m(**enc).logits[:,-1,:].float()
        for j in range(lg.shape[0]):
            d=dict(data[i+j]); d["pred"]="A" if lg[j,ia]>lg[j,ib] else "B"; d["correct"]=int(d["pred"]==d["expected"]); out.append(d)
    json.dump(dict(model=mid,rows=out),open(f"/res/hdc2_{k}.json","w")); resv.commit()
@app.local_entrypoint()
def main():
    data=variants(); print(len(data))
    for k in MODELS: run.spawn(k,data)
