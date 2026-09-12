"""
roles.py — 「役割」ごとに部品構成を選ぶ

これまでは課題が1つ(速くゴールへ)だけだった。
ここでは3つの役割を用意し、役割ごとに
  - タスク報酬 R(エージェントが感じるもの)
  - 適応度 F(淘汰の基準)
の両方を変える。

役割:
  speed   … 速度重視。とにかく速くゴールへ。ハザードは軽視
  safety  … 安全重視。ハザードを絶対に踏まない。多少遅くてもよい
  energy  … 省エネ重視。消費ステップ数を最小に。バッテリーが小さくても回るように

狙い:
  同じ部品カタログから、役割ごとに別の推奨構成が出るはず。
  →「役割ごとの推奨部品構成表」を作る。
"""
import json, os, random, sys, statistics as st
from collections import deque, defaultdict
import codesign_v2 as V

SIZE = 10
DIRS = [(-1,0),(1,0),(0,-1),(0,1)]
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS, exist_ok=True)

# ============================================================
# 役割の定義
# ============================================================
ROLES = {
    # 名前:  報酬(step, hazard, goal),  適応度の重み
    "speed":  {"r_step":-1.0, "r_hazard":-3.0,  "r_goal":20.0,
               "w_time":1.0, "w_hazard":2.0,  "cost_weight":6.0,
               "desc":"速度重視(速くゴールへ。ハザードは軽視)"},
    "safety": {"r_step":-0.3, "r_hazard":-40.0, "r_goal":20.0,
               "w_time":0.15, "w_hazard":30.0, "cost_weight":6.0,
               "desc":"安全重視(ハザードを踏まない。遅くてもよい)"},
    "energy": {"r_step":-2.5, "r_hazard":-10.0, "r_goal":20.0,
               "w_time":2.5, "w_hazard":5.0,   "cost_weight":10.0,
               "desc":"省エネ重視(消費ステップ最小。コストにも厳しい)"},
}


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


def run_role(genome, walls, hazards, role, seed,
             episodes=3000, eval_eps=200, size=SIZE, track_curve=False):
    """役割ごとの報酬でRLを学習し、役割ごとの指標で評価する"""
    R = ROLES[role]
    motor, sensor, stab, battery, weight, compute = genome
    step_len, slip_p = V.COMPONENTS["motor"][motor]
    noise_p  = V.COMPONENTS["sensor"][sensor]
    wobble_p = V.COMPONENTS["stability"][stab]
    max_steps= V.COMPONENTS["battery"][battery]
    drift_p  = V.COMPONENTS["weight"][weight]
    coarse   = V.COMPONENTS["compute"][compute]

    rnd = random.Random(seed); rr, ri = rnd.random, rnd.randrange
    wall_flag=[[False]*size for _ in range(size)]
    hz_flag=[[False]*size for _ in range(size)]
    for (r,c) in walls: wall_flag[r][c]=True
    for (r,c) in hazards: hz_flag[r][c]=True
    goal=(size-1,size-1)
    gs=(size+coarse-1)//coarse
    def sidx(r,c): return ((r//coarse)*gs + (c//coarse))*4

    def observe(r,c):
        if noise_p and rr()<noise_p:
            dr,dc=DIRS[ri(4)]; nr,nc=r+dr,c+dc
            if 0<=nr<size and 0<=nc<size and not wall_flag[nr][nc]: return nr,nc
        return r,c

    def step(r,c,a):
        if wobble_p and rr()<wobble_p: a=ri(4)
        dr,dc=DIRS[a]; n=step_len
        if slip_p and rr()<slip_p: n += 1 if rr()<0.5 else -1
        for _ in range(max(n,0)):
            nr,nc=r+dr,c+dc
            if 0<=nr<size and 0<=nc<size and not wall_flag[nr][nc]: r,c=nr,nc
        if drift_p and rr()<drift_p:
            nr,nc=r+dr,c+dc
            if 0<=nr<size and 0<=nc<size and not wall_flag[nr][nc]: r,c=nr,nc
        reached=(r==goal[0] and c==goal[1]); hz=hz_flag[r][c]
        rew = R["r_step"] + (R["r_hazard"] if hz else 0.0) + (R["r_goal"] if reached else 0.0)
        return r,c,rew,reached,hz

    Q=[0.0]*(gs*gs*4); alpha,gamma=0.2,0.95
    curve=[]; wsucc=0; wn=0
    for ep in range(episodes):
        eps=1.0+(0.05-1.0)*(ep/episodes)
        r=c=0; orow,ocol=observe(r,c); reached=False
        for _ in range(max_steps):
            base=sidx(orow,ocol)
            if rr()<eps: a=ri(4)
            else:
                q0,q1,q2,q3=Q[base],Q[base+1],Q[base+2],Q[base+3]
                a=0; best=q0
                if q1>best: a,best=1,q1
                if q2>best: a,best=2,q2
                if q3>best: a,best=3,q3
            r,c,rew,reached,hz=step(r,c,a)
            nrow,ncol=observe(r,c); nb=sidx(nrow,ncol)
            if reached: target=rew
            else:
                m=Q[nb]
                if Q[nb+1]>m: m=Q[nb+1]
                if Q[nb+2]>m: m=Q[nb+2]
                if Q[nb+3]>m: m=Q[nb+3]
                target=rew+gamma*m
            Q[base+a]+=alpha*(target-Q[base+a])
            orow,ocol=nrow,ncol
            if reached: break
        if track_curve:
            wsucc += 1 if reached else 0; wn+=1
            if wn==100: curve.append(wsucc/wn); wsucc=0; wn=0

    succ=0; steps_ok=0; hz_total=0
    for _ in range(eval_eps):
        r=c=0; orow,ocol=observe(r,c); reached=False; t=0; hits=0
        for _ in range(max_steps):
            base=sidx(orow,ocol)
            q0,q1,q2,q3=Q[base],Q[base+1],Q[base+2],Q[base+3]
            a=0; best=q0
            if q1>best: a,best=1,q1
            if q2>best: a,best=2,q2
            if q3>best: a,best=3,q3
            r,c,rew,reached,hz=step(r,c,a)
            orow,ocol=observe(r,c); t+=1
            if hz: hits+=1
            if reached: break
        if reached: succ+=1; steps_ok+=t
        hz_total+=hits
    out={"success":succ/eval_eps,
         "avg_steps":(steps_ok/succ) if succ else None,
         "hazard":hz_total/eval_eps}
    if track_curve: out["curve"]=curve
    return out


def role_fitness(res, genome, role):
    """役割ごとの適応度。
       共通: 成功率が土台。そこから役割ごとの重みでペナルティを引く。"""
    R = ROLES[role]
    if res["success"] == 0:
        return -100.0 - R["cost_weight"]*V.genome_cost(genome)
    base = res["success"]*100
    base -= R["w_time"] * (res["avg_steps"] or 0)      # 時間ペナルティ
    base -= R["w_hazard"] * res["hazard"]              # ハザードペナルティ
    base -= R["cost_weight"] * V.genome_cost(genome)
    return base


def profile_role(genome, walls, hazards, role, n_seeds=3, episodes=3000):
    rs=[run_role(genome, walls, hazards, role, seed=70000+k*7919,
                 episodes=episodes) for k in range(n_seeds)]
    srs=[r["success"] for r in rs]
    steps=[r["avg_steps"] for r in rs if r["avg_steps"] is not None]
    mean={"success":st.mean(srs), "success_sd":st.pstdev(srs),
          "avg_steps":(st.mean(steps) if steps else None),
          "hazard":st.mean(r["hazard"] for r in rs)}
    return mean


# ============================================================
# 役割ごとのGA
# ============================================================
class RoleGA:
    def __init__(self, walls, hazards, role, pop_size=24, generations=12,
                 elite=3, tourn=3, mut_rate=0.15, seed=0, n_seeds=2):
        self.walls, self.hazards, self.role = walls, hazards, role
        self.pop_size, self.generations = pop_size, generations
        self.elite, self.tourn, self.mut_rate = elite, tourn, mut_rate
        self.n_seeds = n_seeds
        self.rnd = random.Random(seed)
        self.cache = {}; self.evals = 0; self.history = []

    def fitness(self, g):
        if g in self.cache: return self.cache[g][0]
        p = profile_role(g, self.walls, self.hazards, self.role,
                         n_seeds=self.n_seeds)
        self.evals += self.n_seeds
        f = role_fitness(p, g, self.role)
        self.cache[g] = (f, p)
        return f

    def random_genome(self):
        return tuple(self.rnd.choice(list(V.COMPONENTS[k].keys())) for k in V.KEYS)
    def select(self, pool):
        return max((self.rnd.choice(pool) for _ in range(self.tourn)),
                   key=lambda x:x[0])[1]
    def crossover(self,a,b):
        return tuple(a[i] if self.rnd.random()<0.5 else b[i] for i in range(len(V.KEYS)))
    def mutate(self,g):
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
            p=self.cache[scored[0][1]][1]
            self.history.append({"gen":gen,"best":scored[0][0],
                "mean":st.mean(s[0] for s in scored),
                "best_genome":list(scored[0][1]),"dist":dist,"evals":self.evals})
            if verbose:
                stp=f'{p["avg_steps"]:.0f}' if p["avg_steps"] else "--"
                print(f'  gen {gen:2d} best={scored[0][0]:7.1f} '
                      f'成功{p["success"]*100:5.1f}% 歩数{stp} '
                      f'被弾{p["hazard"]:.2f} {"/".join(scored[0][1])}', flush=True)
            nxt=[g for _,g in scored[:self.elite]]
            while len(nxt)<self.pop_size:
                nxt.append(self.mutate(self.crossover(self.select(scored),
                                                      self.select(scored))))
            pop=nxt
        return best, self.history


if __name__ == "__main__":
    role = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv)>2 else 1
    map_seed = int(sys.argv[3]) if len(sys.argv)>3 else 9000
    walls,hz = random_map(SIZE, random.Random(map_seed))
    print(f'=== 役割: {role} ({ROLES[role]["desc"]}) seed={seed} ===')
    ga = RoleGA(walls, hz, role, seed=seed)
    (bf,bg), hist = ga.run()
    p = ga.cache[bg][1]
    print(f'\n勝者: {"/".join(bg)}')
    print(f'  適応度={bf:.1f} 成功率={p["success"]*100:.1f}% '
          f'歩数={p["avg_steps"]:.1f} 被弾={p["hazard"]:.2f} '
          f'コスト={V.genome_cost(bg):.1f}')
    with open(f"{RESULTS}/role_ga_runs.jsonl","a") as f:
        f.write(json.dumps({"role":role,"seed":seed,"map_seed":map_seed,
            "genome":list(bg),"fitness":bf,"success":p["success"],
            "avg_steps":p["avg_steps"],"hazard":p["hazard"],
            "cost":V.genome_cost(bg),"evals":ga.evals},
            ensure_ascii=False)+"\n")
