"""
ga_resample.py — ノイズに強いGA(リサンプリング方式)

問題:1回の学習結果は運に大きく左右される(同じ身体で成功率1%〜100%)。
      単純なGAは「たまたま良い結果が出た個体」を選んでしまう(ノイズへの過適合)。

対策:リサンプリング。
      - 各遺伝子について、評価結果をサンプルとして蓄積していく
      - 毎世代、上位個体には追加のシードで再評価してサンプルを増やす
      - 適応度 = 蓄積した全サンプルの平均
      → まぐれ当たりは世代を追うごとに平均に回帰し、正体がバレる
"""
import os, json, random, statistics as st
from collections import defaultdict
import codesign_v2 as V

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS = os.path.join(_ROOT, "results")
os.makedirs(_RESULTS, exist_ok=True)

class ResampleGA:
    def __init__(self, walls, hazards, pop_size=30, generations=20,
                 elite=4, tourn=3, mut_rate=0.15, seed=0,
                 cost_weight=6.0, init_seeds=2, resample_top=8):
        self.walls, self.hazards = walls, hazards
        self.pop_size, self.generations = pop_size, generations
        self.elite, self.tourn, self.mut_rate = elite, tourn, mut_rate
        self.cost_weight = cost_weight
        self.init_seeds = init_seeds        # 新規個体の初回サンプル数
        self.resample_top = resample_top    # 毎世代 追加評価する上位個体数
        self.rnd = random.Random(seed)
        self.samples = defaultdict(list)    # 遺伝子 → 成功率のサンプル列
        self.evals = 0
        self.history = []

    def _sample(self, g, n=1):
        for _ in range(n):
            k = len(self.samples[g])
            r = V.evaluate_genome(g, self.walls, self.hazards,
                                  seed=(abs(hash(g)) + k*7919) % 100000)
            self.samples[g].append(r["success"])
            self.evals += 1

    def fitness(self, g):
        if not self.samples[g]:
            self._sample(g, self.init_seeds)
        return st.mean(self.samples[g])*100 - self.cost_weight*V.genome_cost(g)

    def random_genome(self):
        return tuple(self.rnd.choice(list(V.COMPONENTS[k].keys())) for k in V.KEYS)

    def select(self, pool):
        return max((self.rnd.choice(pool) for _ in range(self.tourn)),
                   key=lambda x: x[0])[1]

    def crossover(self, a, b):
        return tuple(a[i] if self.rnd.random() < 0.5 else b[i]
                     for i in range(len(V.KEYS)))

    def mutate(self, g):
        g = list(g)
        for i, k in enumerate(V.KEYS):
            if self.rnd.random() < self.mut_rate:
                g[i] = self.rnd.choice(list(V.COMPONENTS[k].keys()))
        return tuple(g)

    def run(self, verbose=True):
        pop = [self.random_genome() for _ in range(self.pop_size)]
        for gen in range(self.generations):
            scored = sorted(((self.fitness(g), g) for g in pop),
                            key=lambda x: -x[0])
            # 上位個体を追加サンプリング(まぐれをあぶり出す)
            for _, g in scored[:self.resample_top]:
                self._sample(g, 1)
            scored = sorted(((self.fitness(g), g) for g in pop),
                            key=lambda x: -x[0])

            dist = {}
            for i, k in enumerate(V.KEYS):
                c = defaultdict(int)
                for g in pop: c[g[i]] += 1
                dist[k] = dict(c)
            self.history.append({
                "gen": gen, "best": scored[0][0],
                "mean": st.mean(s[0] for s in scored),
                "best_genome": list(scored[0][1]),
                "n_samples_best": len(self.samples[scored[0][1]]),
                "dist": dist, "evals": self.evals})
            if verbose:
                print(f"  gen {gen:2d} best={scored[0][0]:6.2f} "
                      f"mean={self.history[-1]['mean']:6.2f} "
                      f"学習{self.evals:4d}回 "
                      f"(best は{len(self.samples[scored[0][1]])}回評価) "
                      f"{'/'.join(scored[0][1])}", flush=True)

            nxt = [g for _, g in scored[:self.elite]]
            while len(nxt) < self.pop_size:
                nxt.append(self.mutate(self.crossover(
                    self.select(scored), self.select(scored))))
            pop = nxt

        final = sorted(((self.fitness(g), g) for g in set(self.samples)
                        if len(self.samples[g]) >= 3), key=lambda x: -x[0])
        return final[0], self.history


if __name__ == "__main__":
    import sys
    map_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
    walls, hazards = V.random_map(10, random.Random(map_seed))
    print(f"=== リサンプリングGA (map={map_seed}) ===")
    ga = ResampleGA(walls, hazards, pop_size=30, generations=15, seed=1)
    (bf, bg), hist = ga.run()
    print(f"\nGAの答え: {'/'.join(bg)}")
    print(f"  適応度={bf:.2f}  サンプル数={len(ga.samples[bg])}  "
          f"成功率={st.mean(ga.samples[bg])*100:.1f}%")
    print(f"  学習回数 合計={ga.evals}")
    with open(os.path.join(_RESULTS, f"v2_resample_{map_seed}.json"), "w") as f:
        json.dump({"best": {"genome": list(bg), "fitness": bf,
                            "n_samples": len(ga.samples[bg])},
                   "evals": ga.evals, "history": hist}, f, ensure_ascii=False)
