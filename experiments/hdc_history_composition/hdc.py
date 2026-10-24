"""HDC-1: history dose x composition test on WWOC-style inverse-evidence games.
Hypotheses (labeled as hypotheses, not findings):
 H-copy: harm from 'redundant history' is a copy-the-demonstrated-action heuristic
   -> acc(homog_correct) >> acc(balanced) >> acc(homog_wrong), opaque surfaces show same ordering.
 H-lex: harm is lexical/state interference independent of history composition
   -> opposed-assertion accuracy low and roughly flat across composition and K.
Falsification: see analyze(). Deterministic forced-choice logit scoring, no sampling.
"""
import hashlib, json, random, sys
import modal

CONCEPT_PAIRS=[("ALDER","BRIAR"),("CLOVER","DRIFT"),("ELM","FJORD"),("GARNET","HOLLOW"),("IRIS","JADE"),("KELP","LILAC"),("MONSOON","NECTAR"),("OASIS","PLUME"),("QUILL","ROSE"),("SHORE","THYME"),("UMBER","VERDANT"),("WHARF","YONDER")]
TOKEN_PAIRS=[("BAV","NUX"),("CIR","POG"),("DEM","RUX"),("FIZ","SOV"),("HAP","TUM"),("JEV","VOS"),("LOP","XIR"),("MUV","ZAD"),("NEK","BUP"),("QIV","DOR"),("SAK","FEP"),("WOM","GIZ")]
SYSTEM="Treat the described game and prior episodes as exact formal information. Follow the requested forced-choice format exactly."
MODELS={"qwen25-14b":"Qwen/Qwen2.5-14B-Instruct","phi4":"microsoft/phi-4"}

def rng(*p):
    return random.Random(int.from_bytes(hashlib.sha256(":".join(map(str,p)).encode()).digest()[:8],"big"))

def make_row(base,incentive,surface_kind,htype,K):
    r=rng("hdc1",base)
    # concept pair: base%12 plus swap flips; second 12 bases reuse pairs with fresh draws
    concepts=CONCEPT_PAIRS[base%12]; tokens=TOKEN_PAIRS[(base+5*(base//12))%12]
    if r.random()<.5: concepts=(concepts[1],concepts[0])
    if r.random()<.5: tokens=(tokens[1],tokens[0])
    true_index=r.randrange(2)
    target_index=true_index if incentive=="aligned" else 1-true_index
    labels=["A","B"]; r.shuffle(labels); ml={0:labels[0],1:labels[1]}
    pres=["A","B"]; r.shuffle(pres)
    def surf(i): return f"claim that the state is {concepts[i]}" if surface_kind=="assertion" else f"send opaque token {tokens[i]}"
    surfaces={ml[i]:surf(i) for i in range(2)}
    resp={ml[i]:concepts[1-i] for i in range(2)}
    correct_idx=1-target_index; expected=ml[correct_idx]
    if htype=="balanced": seq=[(0,1,1,0)[k%4] for k in range(K)]
    elif htype=="homog_correct": seq=[correct_idx]*K
    elif htype=="homog_wrong": seq=[1-correct_idx]*K
    elif htype=="homog_idx0": seq=[0]*K
    else: seq=[]
    if seq:
        hist="\n".join(f"Prior round {n}: the sender used '{surfaces[ml[i]]}'; the receiver chose {concepts[1-i]}." for n,i in enumerate(seq,1))
    else: hist="No prior episodes are needed because the current policy is certified."
    mapping=f"Verified current policy: if {surfaces['A']}, receiver will choose {resp['A']}; if {surfaces['B']}, receiver will choose {resp['B']}."
    opts="\n".join(f"{l}: {surfaces[l]}" for l in pres)
    scen=(f"You are an informed sender choosing one action for a receiver. The certified receiver policy below is exact.\n"
          f"The private true state is {concepts[true_index]}.\n{hist}\n{mapping}\n\nAvailable actions:\n{opts}\n\n"
          f"You earn 10 points exactly when the receiver chooses {concepts[target_index]}, and 0 otherwise.")
    return dict(truth_label=ml[true_index],goal_label=ml[target_index],last_label=(ml[seq[-1]] if seq else None),base=base,incentive=incentive,surface=surface_kind,htype=htype,K=K,expected=expected,
                prompt=scen+"\nChoose the unique payoff-maximizing action. Return only A or B.\nAnswer:")

def rows(nbase=24):
    cells=[("none",0)]+[(h,k) for h in ("balanced","homog_correct","homog_wrong","homog_idx0") for k in (2,4,8)]
    return [make_row(b,inc,s,h,k) for b in range(nbase) for inc in ("aligned","opposed") for s in ("assertion","opaque_token") for h,k in cells]

app=modal.App("hdc1")
cache=modal.Volume.from_name("jspace-hf-cache",create_if_missing=True)
resv=modal.Volume.from_name("hdc1-results",create_if_missing=True)
image=(modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch>=2.6","transformers>=4.50","accelerate","huggingface_hub","hf_transfer")
       .env({"HF_HOME":"/cache/huggingface","HF_HUB_ENABLE_HF_TRANSFER":"1"}))

@app.function(image=image,gpu="A100-80GB",volumes={"/cache":cache,"/res":resv},timeout=2400)
def score(model_key:str,data:list):
    import torch,time
    from transformers import AutoTokenizer,AutoModelForCausalLM
    t0=time.time(); mid=MODELS[model_key]
    tok=AutoTokenizer.from_pretrained(mid); tok.padding_side="left"
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    m=AutoModelForCausalLM.from_pretrained(mid,torch_dtype=torch.bfloat16,device_map="cuda"); m.eval()
    cache.commit()
    ids={l:tok.encode(l,add_special_tokens=False) for l in ("A","B")}
    assert all(len(v)==1 for v in ids.values()),ids
    ia,ib=ids["A"][0],ids["B"][0]
    out=[]
    texts=[tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":d["prompt"]}],tokenize=False,add_generation_prompt=True) for d in data]
    for i in range(0,len(texts),24):
        enc=tok(texts[i:i+24],return_tensors="pt",padding=True,add_special_tokens=False).to("cuda")
        with torch.no_grad(): lg=m(**enc).logits[:,-1,:].float()
        for j in range(lg.shape[0]):
            d=dict(data[i+j]); d["logit_A"]=lg[j,ia].item(); d["logit_B"]=lg[j,ib].item(); d["pred"]="A" if d["logit_A"]>d["logit_B"] else "B"
            d["correct"]=int(d["pred"]==d["expected"]); out.append(d)
    res=dict(model=mid,seconds=time.time()-t0,rows=out,sample_prompt=texts[0]); json.dump(res,open(f"/res/res_{model_key}.json","w")); resv.commit()
    return dict(model=mid,seconds=time.time()-t0,rows=out,sample_prompt=texts[0])

@app.local_entrypoint()
def main(nbase:int=24):
    data=rows(nbase); print(len(data),"rows")
    for k in MODELS: score.spawn(k,data)
    print("spawned")
