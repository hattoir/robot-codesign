"""rebuild_truth.py — 信頼できる正解表を作る
3シード版の上位N件を、12シードで測り直して確定ランキングを作る。
(下位は誤差があっても順位に影響しないので上位のみ精査する = 二段階選抜)"""
import json, os, random, sys, statistics as st
import codesign_v2 as V

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS = os.path.join(_ROOT, "results")
os.makedirs(_RESULTS, exist_ok=True)

TOP_N = int(sys.argv[1]) if len(sys.argv)>1 else 60
A = int(sys.argv[2]) if len(sys.argv)>2 else 0
B = int(sys.argv[3]) if len(sys.argv)>3 else TOP_N
N_SEEDS = 12

walls,hz = V.random_map(10, random.Random(9000))
recs=[json.loads(l) for l in open(os.path.join(_RESULTS, "v2_exhaustive_9000.jsonl"))]
recs.sort(key=lambda r:-r['fitness'])
cands = recs[:TOP_N]

out=open(os.path.join(_RESULTS, "v2_truth_9000.jsonl"), "a")
for i,r in enumerate(cands[A:B]):
    g=tuple(r['genome'])
    srs=[V.evaluate_genome(g,walls,hz,seed=50000+k*7919)['success']
         for k in range(N_SEEDS)]
    m=st.mean(srs); sd=st.pstdev(srs)
    out.write(json.dumps({"genome":list(g),"success":m,"success_sd":sd,
                          "cost":V.genome_cost(g),
                          "fitness":m*100-6.0*V.genome_cost(g),
                          "prelim_rank":A+i+1,"n_seeds":N_SEEDS},
                         ensure_ascii=False)+"\n")
    if (i+1)%10==0: out.flush(); print(f"  {A+i+1}/{TOP_N}", flush=True)
out.close()
print("done")
