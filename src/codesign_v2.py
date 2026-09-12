"""
codesign_v2.py
------------------------------------------------------------
v2:遺伝的アルゴリズム(GA)の導入

v0/v1 では部品が8通りしかなかったので「全部試して並べる」で済んでいた。
これは厳密には進化ではなく総当たり。

v2 では部品を6種類×3択 = 729通りに拡張し、GA で探索する。
  - 選抜(トーナメント) → 交叉(一様交叉) → 突然変異 → エリート保存
  - 同じ遺伝子は再評価せずキャッシュ(GAは同じ個体が何度も出るため)

検証のポイント:
  729通りは全数評価も可能なサイズ。
  → GA の答えを「総当たりの正解」と突き合わせて、GAが正しく動くか確かめられる。
------------------------------------------------------------
"""

import os, sys, json, random, itertools
from collections import deque, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) \
       if "__file__" in dir() else "."
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SIZE = 10
EPISODES = 3000
EVAL_EPS = 250
MAX_STEPS = 140
DIRS = [(-1,0),(1,0),(0,-1),(0,1)]

# ============================================================
# 部品カタログ(6種類 × 3択 = 729通り)
# ============================================================
COMPONENTS = {
    # モーター:1歩で何マス進むか / 滑る確率
    "motor":    {"torque":(1,0.00), "balanced":(2,0.10), "speed":(3,0.20)},
    # センサー精度:自己位置を見間違える確率
    "sensor":   {"precise":0.00, "mid":0.15, "cheap":0.30},
    # 安定性:行動がランダムに化ける確率
    "stability":{"stable":0.00, "mid":0.10, "wobbly":0.25},
    # バッテリー:行動できる最大ステップ数
    "battery":  {"large":140, "mid":90, "small":55},
    # 車体重量:軽いと慣性で滑る / 重いと安定だが1歩遅い
    "weight":   {"light":0.20, "mid":0.08, "heavy":0.00},
    # 演算装置:方策の粒度(粗いと状態をまとめて認識=学習は速いが精度が落ちる)
    "compute":  {"high":1, "mid":2, "low":3},
}
COST = {
    "torque":1.0, "balanced":1.5, "speed":2.0,
    "precise":2.0, "mid":1.0, "cheap":0.0,
    "stable":2.0, "wobbly":0.0,
    "large":2.0, "small":0.0,
    "light":0.0, "heavy":1.5,
    "high":2.5, "low":0.0,
}
# "mid" が複数キーで衝突するので、部品ごとのコスト表を明示的に作る
COST_TABLE = {
    "motor":    {"torque":1.0, "balanced":1.5, "speed":2.0},
    "sensor":   {"precise":2.0, "mid":1.0, "cheap":0.0},
    "stability":{"stable":2.0, "mid":1.0, "wobbly":0.0},
    "battery":  {"large":2.0, "mid":1.0, "small":0.0},
    "weight":   {"light":0.0, "mid":0.7, "heavy":1.5},
    "compute":  {"high":2.5, "mid":1.2, "low":0.0},
}
KEYS = list(COMPONENTS.keys())

def genome_cost(g):
    return sum(COST_TABLE[k][v] for k, v in zip(KEYS, g))

def all_genomes():
    return list(itertools.product(*[list(COMPONENTS[k].keys()) for k in KEYS]))


# ============================================================
# 環境
# ============================================================
def random_map(size, rnd, wall_frac=0.18, hz_frac=0.12):
    start, goal = (0,0), (size-1,size-1)
    def reachable(walls):
        seen={start}; dq=deque([start])
        while dq:
            r,c=dq.popleft()
            if (r,c)==goal: return True
            for dr,dc in DIRS:
                nr,nc=r+dr,c+dc
                if 0<=nr<size and 0<=nc<size and (nr,nc) not in walls and (nr,nc) not in seen:
                    seen.add((nr,nc)); dq.append((nr,nc))
        return False
    cells=[(r,c) for r in range(size) for c in range(size) if (r,c) not in (start,goal)]
    rnd.shuffle(cells)
    walls=set()
    for cell in cells:
        if len(walls) >= int(size*size*wall_frac): break
        walls.add(cell)
        if not reachable(walls): walls.discard(cell)
    free=[c for c in cells if c not in walls]
    rnd.shuffle(free)
    return walls, set(free[:int(size*size*hz_frac)])


def evaluate_genome(genome, walls, hazards, seed, size=SIZE,
                    episodes=EPISODES, eval_eps=EVAL_EPS,
                    track_curve=False):
    """1つの身体でRLを学習し、成績を返す。"""
    motor, sensor, stab, battery, weight, compute = genome
    step_len, slip_p = COMPONENTS["motor"][motor]
    noise_p  = COMPONENTS["sensor"][sensor]
    wobble_p = COMPONENTS["stability"][stab]
    max_steps = COMPONENTS["battery"][battery]
    drift_p  = COMPONENTS["weight"][weight]
    coarse   = COMPONENTS["compute"][compute]   # 状態を coarse マスごとに丸める

    rnd = random.Random(seed)
    rr, ri = rnd.random, rnd.randrange

    wall_flag=[[False]*size for _ in range(size)]
    hz_flag=[[False]*size for _ in range(size)]
    for (r,c) in walls: wall_flag[r][c]=True
    for (r,c) in hazards: hz_flag[r][c]=True
    goal=(size-1,size-1)

    # 状態の粒度(compute が低いと粗くまとめる)
    gs = (size + coarse - 1)//coarse
    n_states = gs*gs
    def sidx(r,c): return ((r//coarse)*gs + (c//coarse))*4

    def observe(r,c):
        if noise_p and rr() < noise_p:
            dr,dc = DIRS[ri(4)]
            nr,nc = r+dr, c+dc
            if 0<=nr<size and 0<=nc<size and not wall_flag[nr][nc]:
                return nr,nc
        return r,c

    def step(r,c,a):
        if wobble_p and rr() < wobble_p: a = ri(4)
        dr,dc = DIRS[a]
        n = step_len
        if slip_p and rr() < slip_p:
            n += 1 if rr()<0.5 else -1
        for _ in range(max(n,0)):
            nr,nc = r+dr, c+dc
            if 0<=nr<size and 0<=nc<size and not wall_flag[nr][nc]:
                r,c = nr,nc
        # 車体が軽いと慣性でもう1マス滑る
        if drift_p and rr() < drift_p:
            nr,nc = r+dr, c+dc
            if 0<=nr<size and 0<=nc<size and not wall_flag[nr][nc]:
                r,c = nr,nc
        reached = (r==goal[0] and c==goal[1])
        hz = hz_flag[r][c]
        rew = -1.0 - (15.0 if hz else 0.0) + (20.0 if reached else 0.0)
        return r,c,rew,reached,hz

    Q=[0.0]*(n_states*4)
    alpha,gamma=0.2,0.95
    curve=[]
    win_succ=0; win_n=0

    for ep in range(episodes):
        eps = 1.0 + (0.05-1.0)*(ep/episodes)
        r=c=0
        orow,ocol = observe(r,c)
        reached=False
        for _ in range(max_steps):
            base = sidx(orow,ocol)
            if rr()<eps: a=ri(4)
            else:
                q0,q1,q2,q3=Q[base],Q[base+1],Q[base+2],Q[base+3]
                a=0; best=q0
                if q1>best: a,best=1,q1
                if q2>best: a,best=2,q2
                if q3>best: a,best=3,q3
            r,c,rew,reached,hz = step(r,c,a)
            nrow,ncol = observe(r,c)
            nb = sidx(nrow,ncol)
            if reached: target = rew
            else:
                m=Q[nb]
                if Q[nb+1]>m: m=Q[nb+1]
                if Q[nb+2]>m: m=Q[nb+2]
                if Q[nb+3]>m: m=Q[nb+3]
                target = rew + gamma*m
            Q[base+a] += alpha*(target-Q[base+a])
            orow,ocol = nrow,ncol
            if reached: break
        if track_curve:
            win_succ += 1 if reached else 0; win_n += 1
            if win_n == 100:
                curve.append(win_succ/win_n); win_succ=0; win_n=0

    # 評価(貪欲方策)
    succ=0; steps_sum=0; hz_sum=0
    for _ in range(eval_eps):
        r=c=0; orow,ocol=observe(r,c); reached=False; t=0; hits=0
        for _ in range(max_steps):
            base=sidx(orow,ocol)
            q0,q1,q2,q3=Q[base],Q[base+1],Q[base+2],Q[base+3]
            a=0; best=q0
            if q1>best: a,best=1,q1
            if q2>best: a,best=2,q2
            if q3>best: a,best=3,q3
            r,c,rew,reached,hz = step(r,c,a)
            orow,ocol = observe(r,c)
            t+=1
            if hz: hits+=1
            if reached: break
        if reached: succ+=1; steps_sum+=t
        hz_sum+=hits
    sr = succ/eval_eps
    out = {"success":sr,
           "avg_steps": (steps_sum/succ) if succ else None,
           "hazard": hz_sum/eval_eps}
    if track_curve: out["curve"]=curve
    return out


def evaluate_genome_avg(genome, walls, hazards, n_seeds=3, base_seed=0, **kw):
    """同じ身体でも学習は運に大きく左右されるので、複数シードで平均を取る。
       (v2の検証で、単一シードだと成功率が1%〜100%まで振れることが判明した)"""
    import statistics as _st
    rs = [evaluate_genome(genome, walls, hazards,
                          seed=base_seed*1000+i*7919, **kw)
          for i in range(n_seeds)]
    srs = [r["success"] for r in rs]
    steps = [r["avg_steps"] for r in rs if r["avg_steps"] is not None]
    return {"success": _st.mean(srs),
            "success_sd": _st.pstdev(srs),
            "avg_steps": (_st.mean(steps) if steps else None),
            "hazard": _st.mean(r["hazard"] for r in rs),
            "n_seeds": n_seeds}


def fitness_of(res, genome, cost_weight=6.0):
    """v2ではシンプルに:成功率 − コスト(学習速度はv2.5で導入)"""
    return res["success"]*100 - cost_weight*genome_cost(genome)


# ============================================================
# 遺伝的アルゴリズム
# ============================================================
class GA:
    def __init__(self, walls, hazards, pop_size=30, generations=20,
                 elite=3, tourn=3, mut_rate=0.15, seed=0, cost_weight=6.0,
                 n_seeds=3):
        self.walls=walls; self.hazards=hazards
        self.pop_size=pop_size; self.generations=generations
        self.elite=elite; self.tourn=tourn; self.mut_rate=mut_rate
        self.cost_weight=cost_weight
        self.n_seeds=n_seeds
        self.rnd=random.Random(seed)
        self.cache={}                 # 遺伝子 → 評価結果(再評価を避ける)
        self.evals=0                  # 実際に学習を回した回数
        self.history=[]

    def random_genome(self):
        return tuple(self.rnd.choice(list(COMPONENTS[k].keys())) for k in KEYS)

    def evaluate(self, g):
        if g in self.cache: return self.cache[g]
        res = evaluate_genome_avg(g, self.walls, self.hazards,
                                  n_seeds=self.n_seeds,
                                  base_seed=abs(hash(g)) % 100000)
        f = fitness_of(res, g, self.cost_weight)
        self.cache[g] = (f, res)
        self.evals += 1
        return self.cache[g]

    def select(self, scored):
        """トーナメント選択"""
        cand = [self.rnd.choice(scored) for _ in range(self.tourn)]
        return max(cand, key=lambda x: x[0])[1]

    def crossover(self, a, b):
        """一様交叉:部品ごとにどちらの親から受け継ぐか決める"""
        return tuple(a[i] if self.rnd.random()<0.5 else b[i]
                     for i in range(len(KEYS)))

    def mutate(self, g):
        g=list(g)
        for i,k in enumerate(KEYS):
            if self.rnd.random() < self.mut_rate:
                g[i] = self.rnd.choice(list(COMPONENTS[k].keys()))
        return tuple(g)

    def run(self, verbose=True):
        pop=[self.random_genome() for _ in range(self.pop_size)]
        best_ever=None
        for gen in range(self.generations):
            scored=[]
            for g in pop:
                f,res = self.evaluate(g)
                scored.append((f,g,res))
            scored.sort(key=lambda x:-x[0])
            if best_ever is None or scored[0][0] > best_ever[0]:
                best_ever = (scored[0][0], scored[0][1], scored[0][2])

            # 遺伝子の分布を記録(淘汰の過程を見るため)
            dist={}
            for i,k in enumerate(KEYS):
                cnt=defaultdict(int)
                for g in pop: cnt[g[i]]+=1
                dist[k]=dict(cnt)
            self.history.append({
                "gen":gen,
                "best":scored[0][0],
                "mean":sum(s[0] for s in scored)/len(scored),
                "best_genome":list(scored[0][1]),
                "dist":dist,
                "evals":self.evals,
            })
            if verbose:
                print(f"  gen {gen:2d}  best={scored[0][0]:7.2f}  "
                      f"mean={self.history[-1]['mean']:7.2f}  "
                      f"評価回数={self.evals}  best={'/'.join(scored[0][1])}",
                      flush=True)

            # 次世代
            pair=[(f,g) for f,g,_ in scored]
            nxt=[g for _,g,_ in scored[:self.elite]]      # エリート保存
            while len(nxt) < self.pop_size:
                p1=self.select(pair); p2=self.select(pair)
                nxt.append(self.mutate(self.crossover(p1,p2)))
            pop=nxt
        return best_ever, self.history


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv)>1 else "ga"
    map_seed = int(sys.argv[2]) if len(sys.argv)>2 else 9000

    walls, hazards = random_map(SIZE, random.Random(map_seed))
    print(f"部品空間: {3**6} 通り  /  マップseed={map_seed}")

    if mode == "ga":
        print("\n=== GA 実行 ===")
        ga = GA(walls, hazards, pop_size=30, generations=20, seed=1)
        (bf, bg, br), hist = ga.run()
        print(f"\nGAの答え: {'/'.join(bg)}")
        print(f"  適応度={bf:.2f}  成功率={br['success']*100:.1f}%  "
              f"コスト={genome_cost(bg):.1f}")
        print(f"  実際に学習を回した回数: {ga.evals} / 729")
        with open(os.path.join(RESULTS_DIR,f"v2_ga_{map_seed}.json"),"w") as f:
            json.dump({"best":{"genome":list(bg),"fitness":bf,**br},
                       "evals":ga.evals,"history":hist}, f, ensure_ascii=False)
        print(f"  → results/v2_ga_{map_seed}.json に保存")

    elif mode == "exhaustive":
        # 分割実行: python3 codesign_v2.py exhaustive <map_seed> <start> <end>
        a = int(sys.argv[3]) if len(sys.argv)>3 else 0
        b = int(sys.argv[4]) if len(sys.argv)>4 else 729
        print(f"\n=== 全数評価 {a}〜{b} ===")
        path = os.path.join(RESULTS_DIR, f"v2_exhaustive_{map_seed}.jsonl")
        gs = all_genomes()[a:b]
        fh = open(path, "a")
        for i,g in enumerate(gs):
            res = evaluate_genome_avg(g, walls, hazards, n_seeds=3,
                                      base_seed=abs(hash(g))%100000)
            f = fitness_of(res, g)
            fh.write(json.dumps({"genome":list(g),"fitness":f,
                                 "cost":genome_cost(g),**res}, ensure_ascii=False)+"\n")
            if (i+1)%50==0: fh.flush(); print(f"  {a+i+1}/729 ...", flush=True)
        fh.close()
        print(f"  → {path} に追記")
