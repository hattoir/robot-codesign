"""
codesign_large.py — 大規模検証(高速版)

高速化:
  - numpy のスカラー演算をやめ、素の Python + random モジュール
  - Q テーブルはフラットな list(numpy インデックスより速い)
  - 1マップ分を1バッチとして JSONL に逐次追記 → 何回かに分けて回せる

使い方:
  python3 codesign_large.py <開始マップID> <終了マップID>
"""

import os, sys, json, random
from collections import deque

SIZE = 10
EPISODES = 5000        # 学習エピソード数(本番設定に戻した)
EVAL_EPS = 400         # 評価エピソード数
MAX_STEPS = 140
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "large_results.jsonl")

MOTORS = ["torque", "speed"]
SENSORS = ["precise", "noisy"]
STABS = ["stable", "wobbly"]
COST = {"torque":1.0, "speed":2.0, "precise":2.0, "noisy":0.0,
        "stable":2.0, "wobbly":0.0}

DIRS = [(-1,0),(1,0),(0,-1),(0,1)]


def random_map(size, rnd, wall_frac=0.18, hz_frac=0.12):
    start, goal = (0,0), (size-1,size-1)
    n_walls = int(size*size*wall_frac)
    n_hz = int(size*size*hz_frac)

    def reachable(walls):
        seen = {start}; dq = deque([start])
        while dq:
            r,c = dq.popleft()
            if (r,c) == goal: return True
            for dr,dc in DIRS:
                nr,nc = r+dr, c+dc
                if 0<=nr<size and 0<=nc<size and (nr,nc) not in walls and (nr,nc) not in seen:
                    seen.add((nr,nc)); dq.append((nr,nc))
        return False

    cells = [(r,c) for r in range(size) for c in range(size) if (r,c) not in (start,goal)]
    rnd.shuffle(cells)
    walls = set()
    for cell in cells:
        if len(walls) >= n_walls: break
        walls.add(cell)
        if not reachable(walls): walls.discard(cell)
    free = [c for c in cells if c not in walls]
    rnd.shuffle(free)
    return walls, set(free[:n_hz])


def make_sim(size, motor, sensor, stab, walls, hazards, rnd):
    """1ステップ関数を閉じ込めて返す(属性アクセスを減らして高速化)"""
    goal = (size-1, size-1)
    wall_flag = [[False]*size for _ in range(size)]
    hz_flag   = [[False]*size for _ in range(size)]
    for (r,c) in walls: wall_flag[r][c] = True
    for (r,c) in hazards: hz_flag[r][c] = True
    noisy = (sensor == "noisy")
    wobbly = (stab == "wobbly")
    fast = (motor == "speed")
    rr = rnd.random; ri = rnd.randrange

    def observe(r, c):
        if noisy and rr() < 0.25:
            dr, dc = DIRS[ri(4)]
            nr, nc = r+dr, c+dc
            if 0 <= nr < size and 0 <= nc < size and not wall_flag[nr][nc]:
                return nr, nc
        return r, c

    def step(r, c, a):
        if wobbly and rr() < 0.20:
            a = ri(4)
        dr, dc = DIRS[a]
        n = 1
        if fast:
            n = 2
            if rr() < 0.15:
                n += 1 if rr() < 0.5 else -1
        for _ in range(n):
            nr, nc = r+dr, c+dc
            if 0 <= nr < size and 0 <= nc < size and not wall_flag[nr][nc]:
                r, c = nr, nc
        reached = (r == goal[0] and c == goal[1])
        hz = hz_flag[r][c]
        rew = -1.0 - (15.0 if hz else 0.0) + (20.0 if reached else 0.0)
        return r, c, rew, reached, hz

    return step, observe


def train_and_eval(size, motor, sensor, stab, walls, hazards, seed):
    rnd = random.Random(seed)
    step, observe = make_sim(size, motor, sensor, stab, walls, hazards, rnd)
    rr = rnd.random; ri = rnd.randrange

    n_states = size*size
    Q = [0.0]*(n_states*4)
    alpha, gamma = 0.2, 0.95

    for ep in range(EPISODES):
        eps = 1.0 + (0.05-1.0)*(ep/EPISODES)
        r = c = 0
        orow, ocol = observe(r, c)
        for _ in range(MAX_STEPS):
            base = (orow*size + ocol)*4
            if rr() < eps:
                a = ri(4)
            else:
                q0,q1,q2,q3 = Q[base],Q[base+1],Q[base+2],Q[base+3]
                a = 0; best = q0
                if q1 > best: a, best = 1, q1
                if q2 > best: a, best = 2, q2
                if q3 > best: a, best = 3, q3
            r, c, rew, reached, hz = step(r, c, a)
            nrow, ncol = observe(r, c)
            nb = (nrow*size + ncol)*4
            if reached:
                target = rew
            else:
                m = Q[nb]
                if Q[nb+1] > m: m = Q[nb+1]
                if Q[nb+2] > m: m = Q[nb+2]
                if Q[nb+3] > m: m = Q[nb+3]
                target = rew + gamma*m
            Q[base+a] += alpha*(target - Q[base+a])
            orow, ocol = nrow, ncol
            if reached: break

    # 評価(貪欲方策)
    succ = 0; steps_ok = 0; hz_total = 0
    for _ in range(EVAL_EPS):
        r = c = 0
        orow, ocol = observe(r, c)
        reached = False; t = 0; hits = 0
        for _ in range(MAX_STEPS):
            base = (orow*size + ocol)*4
            q0,q1,q2,q3 = Q[base],Q[base+1],Q[base+2],Q[base+3]
            a = 0; best = q0
            if q1 > best: a, best = 1, q1
            if q2 > best: a, best = 2, q2
            if q3 > best: a, best = 3, q3
            r, c, rew, reached, hz = step(r, c, a)
            orow, ocol = observe(r, c)
            t += 1
            if hz: hits += 1
            if reached: break
        if reached:
            succ += 1; steps_ok += t
        hz_total += hits
    sr = succ/EVAL_EPS
    return {"success": sr,
            "avg_steps": (steps_ok/succ) if succ else None,
            "hazard": hz_total/EVAL_EPS}


def main(m_start, m_end):
    with open(OUT, "a") as f:
        for m in range(m_start, m_end):
            walls, hazards = random_map(SIZE, random.Random(9000+m))
            for motor in MOTORS:
                for sensor in SENSORS:
                    for stab in STABS:
                        res = train_and_eval(SIZE, motor, sensor, stab,
                                             walls, hazards, seed=m*97+7)
                        rec = {"map": m, "motor": motor, "sensor": sensor,
                               "stab": stab,
                               "cost": COST[motor]+COST[sensor]+COST[stab],
                               **res}
                        f.write(json.dumps(rec)+"\n")
            f.flush()
            print(f"map {m} done", flush=True)


if __name__ == "__main__":
    a = int(sys.argv[1]); b = int(sys.argv[2])
    main(a, b)
