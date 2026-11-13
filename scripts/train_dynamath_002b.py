"""DynaMath 002b: final continuation of 002a weights and optimizer on SAME branch.

No topology, loss, tokenizer, or training-stream changes; 002b is the final training label.
"""
import hashlib, json, math, platform, random, sys, time
from pathlib import Path
import torch
from torch.nn import functional as F
from dynamath.v002 import Model
from dynamath.v002.__main__ import train_language, generate, streams
from dynamath.korean_data import Tokenizer, fetch, load, EOS, SILENCE

ROOT=Path("docs/dynamath/experiment-002b")
OUT=Path("models/dynamath/002b")
STAGES=(30000,45000,60000,72000)
VARIANTS=("model1","model3-cross")
SOURCE_SHA="287eb129695b577321c80ad397bb3c2279164d4ca577874d129fd3db5b30afe2"

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def data():
    csv=Path("data/dynamath/001a/ChatbotData.csv")
    if not csv.exists():
        csv.parent.mkdir(parents=True,exist_ok=True)
        csv.write_bytes(fetch("ChatbotData.csv"))
    if digest(csv)!=SOURCE_SHA: raise ValueError("Pinned Song Ys CSV mismatch")
    return load(csv.parent)

def stream_coverage(rows,tok,batch,x):
    counts=[];fully=0
    for lane in range(batch):
        cumulative=0
        for row in rows[lane::batch]:
            n=len(tok.pairs(row)[0])
            cumulative+=n
            if cumulative<=len(x):fully+=1
        counts.append(cumulative)
    return dict(per_lane_positions=counts,usable_per_lane=len(x),
        discarded_positions=sum(c-len(x) for c in counts),
        complete_pairs_in_original_stream=fully,total_pairs=len(rows))

@torch.no_grad()
def full_eval(model,rows,tok,batch=8):
    """True full-holdout teacher-forced evaluation (no min-lane truncation)."""
    lanes=[]
    for k in range(min(batch,len(rows))):
        a=[];b=[]
        for row in rows[k::batch]:
            xx,yy=tok.pairs(row);a+=xx;b+=yy
        lanes.append((a,b))
    length=max(len(a) for a,b in lanes);width=len(lanes)
    x=torch.full((length,width),tok.ids[EOS],dtype=torch.long)
    y=torch.full_like(x,tok.ids[SILENCE])
    active=torch.zeros_like(x,dtype=torch.bool)
    for j,(a,b) in enumerate(lanes):
        x[:len(a),j]=torch.tensor(a)
        y[:len(b),j]=torch.tensor(b)
        active[:len(a),j]=True
    state,mem=model.zero(width)
    loss=0.;correct=total=omissions=silent_checks=false_emissions=0
    for pos in range(length):
        state=model.inject(state,x[pos])
        state,mem,_=model.step(state,mem,pos*3)
        state,mem,_=model.step(state,mem,pos*3+1)
        first=model.logits(state)
        state,mem,_=model.step(state,mem,pos*3+2)
        second=model.logits(state)
        mask=active[pos];response=mask&(y[pos]!=tok.ids[SILENCE])
        if response.any():
            ce=F.cross_entropy(first,y[pos],reduction="none")
            loss+=float(ce[response].sum());total+=int(response.sum())
            pred=first.argmax(1)
            correct+=int(((pred==y[pos])&response).sum())
            emitted=(first.max(1).values/model.config.gain>=.1)&(pred!=tok.ids[SILENCE])
            omissions+=int((response&~emitted).sum())
        for logits,quiet in ((first,mask&(y[pos]==tok.ids[SILENCE])),(second,mask)):
            pred=logits.argmax(1)
            emitted=(logits.max(1).values/model.config.gain>=.1)&(pred!=tok.ids[SILENCE])
            silent_checks+=int(quiet.sum());false_emissions+=int((quiet&emitted).sum())
    ce=loss/total
    return dict(pairs=len(rows),response_tokens=total,response_CE=ce,
                response_PPL=math.exp(ce),response_accuracy=correct/total,
                omission_rate=omissions/total,off_target_emissions=false_emissions,
                silent_target_checks=silent_checks,
                false_emission_rate=false_emissions/silent_checks,
                all_pairs_included=True)

def examples(model,rows,ticks):
    prompts=[r["Q"] for r in rows[:12]]+[
        "안녕하세요","고마워","시험 때문에 걱정돼","배고파","잠이 안 와"]
    return [generate(model,p,max_ticks=ticks) for p in prompts]

def main():
    ROOT.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path("configs/dynamath/experiment-002.json").read_text())
    torch.set_num_threads(cfg["threads"]);torch.use_deterministic_algorithms(True)
    splits,tok=data();x,y=streams(splits["train"],tok,cfg["batch"])
    provenance={name:json.loads((Path("docs/dynamath/experiment-002a")/(name+"-progress.json")).read_text())["stages"][-1]["sha256"] for name in VARIANTS}
    coverage=stream_coverage(splits["train"],tok,cfg["batch"],x)
    baseline=json.loads(Path("docs/dynamath/experiment-002a/summary.json").read_text())
    save(ROOT/"plan-and-environment.json",dict(version="002b",branch="feature/dynamath-002",
        starting_steps=18000,step_targets=STAGES,variants=VARIANTS,
        source_sha256=SOURCE_SHA,training_stream_coverage=coverage,
        limitation="Continuing original 002 truncated per-lane stream; some train pairs excluded.",
        pytorch=torch.__version__,python=sys.version,cpu_platform=platform.platform(),
        threads=cfg["threads"],heldout_test="evaluate only once per finished variant",
        original_002a_full_test_CE={n:baseline[n]["final_test_CE"] for n in VARIANTS}))
    all_results={}
    for name in VARIANTS:
        source=Path("models/dynamath/002a")/(name+"-step18000.pt")
        if digest(source)!=provenance[name]: raise ValueError("Original 002a checkpoint changed "+str(source))
        model,payload=Model.load(source)
        if model.config.frozen: raise ValueError("Final 002b candidates must have writable scalars")
        restart=payload["restart"]
        completed=restart.get("completed_steps",restart["t"]//(3*cfg["tbptt_slots"]))
        if completed!=18000: raise ValueError("002a base checkpoint has unexpected steps")
        random.seed(cfg["seed"]);torch.manual_seed(cfg["seed"])
        entry=dict(variant=name,source=str(source),source_sha256=digest(source),
                   base_steps=completed,stages=[],finished=False)
        save(ROOT/(name+"-progress.json"),entry)
        for target in STAGES:
            # train_language resumes optimizer, model, S, m, tick, stream position and RNG.
            segment_cfg={**cfg,"steps":target,"seconds_cap_per_run":2100}
            t0=time.perf_counter()
            training,restart=train_language(model,x,y,segment_cfg,resume=restart)
            path=OUT/(name+"-step"+str(training["steps"])+".pt")
            if path.exists(): raise FileExistsError(str(path))
            model.save(path,training=training,restart=restart,
                       provenance=dict(previous_002a_sha256=digest(source),version="002b"))
            val=full_eval(model,splits["validation"],tok)
            entry["stages"].append(dict(target=target,actual_steps=training["steps"],
                training_seconds=training["seconds"],wall_seconds=time.perf_counter()-t0,
                checkpoint=str(path),sha256=digest(path),validation=val,
                recorded_training_loss=training["history"]))
            save(ROOT/(name+"-progress.json"),entry)
            print(json.dumps(dict(model=name,step=training["steps"],
                                  validation_CE=val["response_CE"])),flush=True)
            if training["steps"]!=target:
                entry["reason"]="per-segment wall time cap";break
        entry["finished"]=bool(entry["stages"] and entry["stages"][-1]["actual_steps"]==STAGES[-1])
        if entry["finished"]:
            entry["test_full"]=full_eval(model,splits["test"],tok)
            entry["free_generation"]=examples(model,splits["test"],cfg["generation_ticks"])
            entry["generation_summary"]=dict(samples=len(entry["free_generation"]),
                eos=sum(bool(r["eos"]) for r in entry["free_generation"]),
                question_events=sum(len(r["question_events"]) for r in entry["free_generation"]),
                raw_output=[r["text"] for r in entry["free_generation"]])
        save(ROOT/(name+"-progress.json"),entry)
        all_results[name]=entry
    save(ROOT/"summary.json",{n:dict(steps=v["stages"][-1]["actual_steps"],
        complete=v["finished"],validation_CE=v["stages"][-1]["validation"]["response_CE"],
        final_test_CE=v.get("test_full",{}).get("response_CE"),
        final_test_PPL=v.get("test_full",{}).get("response_PPL"),
        eos_count=v.get("generation_summary",{}).get("eos")) for n,v in all_results.items()})
    report=["# DynaMath 002b — final measured continuation","",
        "Original 002a 18,000-update checkpoints were resumed, not reinitialized. Same nine laws,",
        "same formula parameterization and supervision; only the update budget changed.","",
        "| Model | Cumulative updates | Validation CE | Full-test CE | Full-test PPL | EOS / 17 |",
        "|---|---:|---:|---:|---:|---:|"]
    for name,v in all_results.items():
        last=v["stages"][-1]
        test=v.get("test_full")
        report.append("| {} | {} | {:.4f} | {} | {} | {} |".format(
            name,last["actual_steps"],last["validation"]["response_CE"],
            ("{:.4f}".format(test["response_CE"]) if test else "not evaluated"),
            ("{:.3f}".format(test["response_PPL"]) if test else "—"),
            (str(v["generation_summary"]["eos"]) if test else "—")))
    report+=["", "## Comparison to 002a on the same full test split",
        "| Model | 002a CE | 002b CE | CE change |",
        "|---|---:|---:|---:|"]
    for name,v in all_results.items():
        test=v.get("test_full")
        previous=baseline[name]["final_test_CE"]
        if test:
            report.append("| {} | {:.4f} | {:.4f} | {:+.4f} |".format(
                name,previous,test["response_CE"],test["response_CE"]-previous))
        else:
            report.append("| {} | {:.4f} | not evaluated | — |".format(name,previous))
    report+=["","## Interpretation limits",
        "- This trains coefficients of a fixed family of formulas, NOT symbolic formula discovery.",
        "- Model1 has 9 laws and 1 shared scalar; Model3-cross has 9 laws and 3 cross-readable scalars.",
        "- Original min-lane stream truncation retained to preserve exact optimizer continuation.",
        "- 002a and 002b final test use the same full split, objective and readout schedule.",
        "- Single seed; no held-out test selection; initial and new model files preserved.",
        "- Teacher-forced PPL is not proof of coherent closed-loop dialogue.",
        "- EOS and scalar writes alone are not proof of useful memory or response meaning.",
        "", "Detailed stage records, checkpoint hashes and unedited generated text are in per-model progress JSON."]
    (ROOT/"report.md").write_text("\n".join(report)+"\n")
    print("002b completed all variants:",all(v["finished"] for v in all_results.values()),flush=True)

if __name__=="__main__": main()
