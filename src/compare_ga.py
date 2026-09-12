"""compare_ga.py — GA方式を複数回走らせて平均順位で比較する"""
import json, os, random, sys, statistics as st
import codesign_v2 as V
from ga_resample import ResampleGA

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS = os.path.join(_ROOT, "results")
os.makedirs(_RESULTS, exist_ok=True)

recs=[json.loads(l) for l in open(os.path.join(_RESULTS, "v2_exhaustive_9000.jsonl"))]
recs.sort(key=lambda r:-r['fitness'])
RANK={tuple(r['genome']):i for i,r in enumerate(recs,1)}
FIT={tuple(r['genome']):r['fitness'] for r in recs}

def simple_ga(walls,hz,seed,n_seeds=1,pop=24,gens=12):
    ga=V.GA(walls,hz,pop_size=pop,generations=gens,seed=seed,n_seeds=n_seeds)
    (bf,bg,br),_=ga.run(verbose=False)
    return bg, ga.evals*n_seeds

def resample_ga(walls,hz,seed,pop=24,gens=12):
    ga=ResampleGA(walls,hz,pop_size=pop,generations=gens,seed=seed)
    (bf,bg),_=ga.run(verbose=False)
    return bg, ga.evals

if __name__=="__main__":
    which=sys.argv[1]; seed=int(sys.argv[2])
    walls,hz=V.random_map(10,random.Random(9000))
    if which=="simple1": g,e=simple_ga(walls,hz,seed,n_seeds=1)
    elif which=="simple3": g,e=simple_ga(walls,hz,seed,n_seeds=3)
    else: g,e=resample_ga(walls,hz,seed)
    print(json.dumps({"method":which,"seed":seed,"genome":list(g),
                      "rank":RANK.get(tuple(g)),"true_fitness":FIT.get(tuple(g)),
                      "evals":e}, ensure_ascii=False), flush=True)
    with open(os.path.join(_RESULTS, "ga_compare.jsonl"), "a") as f:
        f.write(json.dumps({"method":which,"seed":seed,"genome":list(g),
                            "rank":RANK.get(tuple(g)),
                            "true_fitness":FIT.get(tuple(g)),"evals":e},
                           ensure_ascii=False)+"\n")
