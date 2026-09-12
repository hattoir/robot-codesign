"""
ga_v25.py — 3軸適応度(性能・コスト・学習しやすさ)でGAを回す

v2 のGAは「最終性能 − コスト」だけで選抜していた。
v2.5 では learnability.py の知見を組み込み、以下を適応度に入れる:
  - 最終性能(平均)
  - 学習の安定性(到達率)
  - 学習速度(忍耐予算 patience を超えた分だけ減点)
  - 部品コスト

これにより「役割(開発時間の制約)ごとに、選ばれる身体が変わる」ことを
GA自身に発見させる。
"""
import json, os, random, sys, statistics as st
from collections import defaultdict
import codesign_v2 as V
from learnability import learn_curve, time_to_threshold

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS = os.path.join(_ROOT, "results")
os.makedirs(_RESULTS, exist_ok=True)

class GA25:
    def __init__(self, walls, hazards, patience=1500, impatience=0.02,
                 pop_size=24, generations=12, elite=3, tourn=3,
                 mut_rate=0.15, seed=0, cost_weight=6.0, n_seeds=3,
                 thresh=0.8, episodes=3000):
        self.walls, self.hazards = walls, hazards
        self.patience, self.impatience = patience, impatience
        self.pop_size, self.generations = pop_size, generations
        self.elite, self.tourn, self.mut_rate = elite, tourn, mut_rate
        self.cost_weight, self.n_seeds = cost_weight, n_seeds
        self.thresh, self.episodes = thresh, episodes
        self.rnd = random.Random(seed)
        self.cache = {}
        self.evals = 0
        self.history = []

    def profile(self, g):
        if g in self.cache: return self.cache[g]
        finals, times, reached = [], [], 0
        for k in range(self.n_seeds):
            curve, final = learn_curve(g, self.walls, self.hazards,
                                       seed=(abs(hash(g))+k*7919) % 100000,
                                       episodes=self.episodes)
            finals.append(final)
            t = time_to_threshold(curve, self.thresh)
            if t is not None: times.append(t); reached += 1
            self.evals += 1
        p = {"final_mean": st.mean(finals), "final_sd": st.pstdev(finals),
             "reach_rate": reached/self.n_seeds,
             "time_mean": (st.mean(times) if times else None),
             "cost": V.genome_cost(g)}
        self.cache[g] = p
        return p

    def fitness(self, g):
        p = self.profile(g)
        base = (p["final_mean"]*100 - self.cost_weight*p["cost"]) * p["reach_rate"]
        if p["time_mean"] is None: return base - 50.0
        return base - self.impatience * max(0.0, p["time_mean"] - self.patience)

    def random_genome(self):
        return tuple(self.rnd.choice(list(V.COMPONENTS[k].keys())) for k in V.KEYS)
    def select(self, pool):
        return max((self.rnd.choice(pool) for _ in range(self.tourn)),
                   key=lambda x: x[0])[1]
    def crossover(self, a, b):
        return tuple(a[i] if self.rnd.random()<0.5 else b[i]
                     for i in range(len(V.KEYS)))
    def mutate(self, g):
        g=list(g)
        for i,k in enumerate(V.KEYS):
            if self.rnd.random()<self.mut_rate:
                g[i]=self.rnd.choice(list(V.COMPONENTS[k].keys()))
        return tuple(g)

    def run(self, verbose=True):
        pop=[self.random_genome() for _ in range(self.pop_size)]
        best=None
        for gen in range(self.generations):
            scored=sorted(((self.fitness(g),g) for g in pop), key=lambda x:-x[0])
            if best is None or scored[0][0]>best[0]: best=scored[0]
            dist={}
            for i,k in enumerate(V.KEYS):
                c=defaultdict(int)
                for g in pop: c[g[i]]+=1
                dist[k]=dict(c)
            p=self.profile(scored[0][1])
            self.history.append({"gen":gen,"best":scored[0][0],
                "mean":st.mean(s[0] for s in scored),
                "best_genome":list(scored[0][1]),"dist":dist,"evals":self.evals,
                "best_time":p["time_mean"],"best_final":p["final_mean"]})
            if verbose:
                t=f'{p["time_mean"]:.0f}' if p["time_mean"] else "--"
                print(f'  gen {gen:2d} best={scored[0][0]:6.1f} '
                      f'mean={self.history[-1]["mean"]:6.1f} '
                      f'性能{p["final_mean"]*100:5.1f}% 学習{t}ep '
                      f'{"/".join(scored[0][1])}', flush=True)
            nxt=[g for _,g in scored[:self.elite]]
            while len(nxt)<self.pop_size:
                nxt.append(self.mutate(self.crossover(
                    self.select(scored), self.select(scored))))
            pop=nxt
        return best, self.history

if __name__=="__main__":
    label=sys.argv[1]; pat=int(sys.argv[2]); imp=float(sys.argv[3])
    seed=int(sys.argv[4]) if len(sys.argv)>4 else 1
    walls,hz=V.random_map(10,random.Random(9000))
    print(f'=== GA v2.5 [{label}] patience={pat} impatience={imp} seed={seed} ===')
    ga=GA25(walls,hz,patience=pat,impatience=imp,seed=seed)
    (bf,bg),hist=ga.run()
    p=ga.profile(bg)
    print(f'\n勝者: {"/".join(bg)}')
    print(f'  適応度={bf:.1f} 性能={p["final_mean"]*100:.1f}% '
          f'学習={p["time_mean"]:.0f}ep コスト={p["cost"]:.1f}')
    with open(os.path.join(_RESULTS, "v25_ga_runs.jsonl"), "a") as f:
        f.write(json.dumps({"label":label,"patience":pat,"impatience":imp,
            "seed":seed,"genome":list(bg),"fitness":bf,
            "final":p["final_mean"],"time":p["time_mean"],"cost":p["cost"],
            "evals":ga.evals}, ensure_ascii=False)+"\n")
