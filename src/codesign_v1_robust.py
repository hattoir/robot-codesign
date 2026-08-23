"""
codesign_v1_robust.py
------------------------------------------------------------
v1 の検証:「torque全滅」は本物か、それとも1枚のマップの偶然か?

方法:
  - ランダムに壁とハザードを配置したマップを N 枚 生成する。
  - 各マップで、8つの部品構成すべてに強化学習を回して評価する。
  - マップをまたいで集計:
      * 各構成の「平均成功率」と「勝者になった回数」
      * モーター種別(torque / speed)ごとの平均成功率
  => 1枚のマップに依存しない結論を出す。
------------------------------------------------------------
"""

import numpy as np
import itertools
from collections import defaultdict

COMPONENTS = {
    "motor":     ["torque", "speed"],
    "sensor":    ["precise", "noisy"],
    "stability": ["stable", "wobbly"],
}
COST = {
    ("motor","torque"):1.0, ("motor","speed"):2.0,
    ("sensor","precise"):2.0, ("sensor","noisy"):0.0,
    ("stability","stable"):2.0, ("stability","wobbly"):0.0,
}
def body_cost(b): return sum(COST[(k,v)] for k,v in b.items())


def random_map(size, rng, n_walls=None, n_hazards=None):
    """スタート(0,0)からゴール(size-1,size-1)へ到達可能な、
       ランダムな壁・ハザード配置を作る(BFSで到達性を保証)。"""
    start, goal = (0,0), (size-1, size-1)
    n_walls = n_walls or int(size*size*0.18)
    n_hazards = n_hazards or int(size*size*0.12)

    def reachable(walls):
        from collections import deque
        seen = {start}; dq = deque([start])
        while dq:
            r,c = dq.popleft()
            if (r,c) == goal: return True
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc = r+dr, c+dc
                if 0<=nr<size and 0<=nc<size and (nr,nc) not in walls and (nr,nc) not in seen:
                    seen.add((nr,nc)); dq.append((nr,nc))
        return False

    cells = [(r,c) for r in range(size) for c in range(size)
             if (r,c) not in (start, goal)]
    # 壁を置いては到達性を確認(壊したら戻す)
    walls = set()
    rng.shuffle(cells)
    for cell in cells:
        if len(walls) >= n_walls: break
        walls.add(cell)
        if not reachable(walls):
            walls.discard(cell)
    # ハザードは壁でない所に置く
    free = [c for c in cells if c not in walls]
    rng.shuffle(free)
    hazards = set(free[:n_hazards])
    return walls, hazards


class Grid:
    def __init__(self, size, body, walls, hazards, max_steps=140):
        self.size=size; self.body=body; self.walls=walls
        self.hazards=hazards; self.max_steps=max_steps
        self.goal=(size-1,size-1)
    def reset(self): self.pos=(0,0); self.t=0; return self._obs(self.pos)
    def _clamp(self,r,c): return (min(max(r,0),self.size-1), min(max(c,0),self.size-1))
    def _obs(self,p):
        if self.body["sensor"]=="noisy" and self._rng()<0.25:
            r,c=p; dr,dc=[(-1,0),(1,0),(0,-1),(0,1)][int(self._ri(4))]
            cand=self._clamp(r+dr,c+dc)
            if cand not in self.walls: return cand
        return p
    # 乱数はグローバルの rng を使う
    def _rng(self): return rng.random()
    def _ri(self,n): return rng.integers(0,n)
    def _move1(self,r,c,dr,dc):
        nr,nc=self._clamp(r+dr,c+dc)
        return (r,c) if (nr,nc) in self.walls else (nr,nc)
    def step(self,a):
        if self.body["stability"]=="wobbly" and rng.random()<0.20:
            a=int(rng.integers(0,4))
        dr,dc=[(-1,0),(1,0),(0,-1),(0,1)][a]
        if self.body["motor"]=="torque": step_len=1
        else:
            step_len=2
            if rng.random()<0.15: step_len+=rng.choice([-1,1])
        r,c=self.pos
        for _ in range(max(step_len,0)): r,c=self._move1(r,c,dr,dc)
        self.pos=(r,c); self.t+=1
        reached=(self.pos==self.goal); hz=(self.pos in self.hazards)
        done=reached or (self.t>=self.max_steps)
        rew=-1.0-(15.0 if hz else 0.0)+(20.0 if reached else 0.0)
        return self._obs(self.pos), rew, done, {"reached":reached,"hazard":hz}


def train(env, episodes=2000, alpha=0.2, gamma=0.95):
    n=env.size; Q=np.zeros((n,n,4))
    for ep in range(episodes):
        eps=1.0+(0.05-1.0)*(ep/episodes)
        o=env.reset(); done=False
        while not done:
            r,c=o
            a=int(rng.integers(0,4)) if rng.random()<eps else int(np.argmax(Q[r,c]))
            no,rew,done,_=env.step(a); nr,nc=no
            Q[r,c,a]+=alpha*(rew+(0 if done else gamma*np.max(Q[nr,nc]))-Q[r,c,a])
            o=no
    return Q

def evaluate(env, Q, episodes=150):
    succ=0
    for _ in range(episodes):
        o=env.reset(); done=False
        while not done:
            r,c=o; o,_,done,info=env.step(int(np.argmax(Q[r,c])))
        if info["reached"]: succ+=1
    return succ/episodes


def run(size=10, n_maps=12, seed=0):
    global rng
    keys=list(COMPONENTS.keys())
    bodies=[dict(zip(keys,combo)) for combo in itertools.product(*COMPONENTS.values())]

    succ_by_body=defaultdict(list)
    wins=defaultdict(int)
    succ_by_motor=defaultdict(list)

    for m in range(n_maps):
        rng=np.random.default_rng(1000+m)          # マップ生成用
        walls,hazards=random_map(size,rng)
        rng=np.random.default_rng(seed*100+m)      # 学習・評価用(再現性)
        scored=[]
        for b in bodies:
            env=Grid(size,b,walls,hazards)
            Q=train(env); sr=evaluate(env,Q)
            key=f'{b["motor"]}/{b["sensor"]}/{b["stability"]}'
            succ_by_body[key].append(sr)
            succ_by_motor[b["motor"]].append(sr)
            fitness=sr*100 - 6.0*body_cost(b)      # 成功率ベースの適応度
            scored.append((fitness, key))
        scored.sort(reverse=True)
        wins[scored[0][1]]+=1
        print(f'  map {m+1:2d}/{n_maps}: 勝者 = {scored[0][1]}')

    return bodies, succ_by_body, wins, succ_by_motor


if __name__ == "__main__":
    N=5
    print("="*70)
    print(f"v1 検証:ランダムマップ {N} 枚で「torque全滅」は再現するか")
    print("="*70)
    bodies, succ_by_body, wins, succ_by_motor = run(size=10, n_maps=N)

    print("\n--- 構成ごとの平均成功率(マップ{}枚平均)---".format(N))
    rows=sorted(succ_by_body.items(),
                key=lambda kv: np.mean(kv[1]), reverse=True)
    for key,vals in rows:
        bar="█"*int(np.mean(vals)*20)
        print(f'  {key:<26} {np.mean(vals)*100:5.0f}%  {bar}')

    print("\n--- モーター種別の平均成功率 ---")
    for mot in ["speed","torque"]:
        v=succ_by_motor[mot]
        print(f'  {mot:<8} {np.mean(v)*100:5.1f}%  (試行 {len(v)})')

    print("\n--- 勝者になった回数 ---")
    for key,cnt in sorted(wins.items(), key=lambda kv:-kv[1]):
        print(f'  {key:<26} {cnt} / {N} 回')
