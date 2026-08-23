"""
codesign_v1.py
------------------------------------------------------------
進化 × 強化学習によるロボット協調設計 —— v1(課題を難しくする)

v0 の問題:6x6 のまっすぐ歩くだけの課題では、8通り全部が成功率100%。
           部品の差が成績に出ず、実質コストだけで順位が決まった(天井効果)。

v1 の狙い:課題を厳しくして「安い部品が破綻する場所」を作る。
  - グリッドを広げる(10x10)
  - ペナルティのマス(hazard)を配置:踏むと大きく減点
  - 狭い通路:壁で囲み、位置を間違えると壁にぶつかって進めない
  => ノイズありセンサー / 動作ブレ / 行き過ぎるスピード型 が
     ここで初めて「失敗」に繋がるはず。

設計の背骨は v0 と同じ:
  R(エージェントが感じる報酬)と F(客観的な適応度)を分ける。
------------------------------------------------------------
"""

import numpy as np
import itertools

rng = np.random.default_rng(0)

# ============================================================
# 部品カタログとコスト(v0 と同じ)
# ============================================================
COMPONENTS = {
    "motor":     ["torque", "speed"],
    "sensor":    ["precise", "noisy"],
    "stability": ["stable", "wobbly"],
}
COST = {
    ("motor", "torque"): 1.0, ("motor", "speed"): 2.0,
    ("sensor", "precise"): 2.0, ("sensor", "noisy"): 0.0,
    ("stability", "stable"): 2.0, ("stability", "wobbly"): 0.0,
}
def body_cost(body):
    return sum(COST[(k, v)] for k, v in body.items())


# ============================================================
# 環境 v1:壁 + ハザード付きグリッド
# ============================================================
def build_map(size=10):
    """壁(#)とハザード(!)を配置したマップを作る。
       スタート(0,0) 左上 → ゴール(size-1,size-1) 右下。
       途中に、狭い通路と、その脇のハザード地帯を置く。"""
    walls = set()
    hazards = set()

    mid = size // 2
    # 横方向の壁で上下を仕切り、真ん中に1マスだけ通路を開ける(狭い通路)
    gap = mid  # 通路の列
    for c in range(size):
        if c != gap:
            walls.add((mid, c))

    # 通路の両脇(壁の手前)にハザードを敷く:通路を外すと減点地帯に入る
    for c in range(size):
        if c not in (gap,):
            if (mid - 1, c) not in walls:
                hazards.add((mid - 1, c))

    # ゴール手前にもハザードを少し(精密に止まれないと踏む)
    hazards.add((size - 1, size - 2))
    hazards.add((size - 2, size - 1))

    # スタート/ゴールは安全にする
    for cell in [(0, 0), (size - 1, size - 1)]:
        walls.discard(cell); hazards.discard(cell)
    return walls, hazards


class GridWorldV1:
    def __init__(self, size, body, walls, hazards, max_steps=120):
        self.size = size
        self.body = body
        self.walls = walls
        self.hazards = hazards
        self.max_steps = max_steps
        self.goal = (size - 1, size - 1)

    def reset(self):
        self.pos = (0, 0)
        self.t = 0
        return self._observe(self.pos)

    def _clamp(self, r, c):
        return (min(max(r, 0), self.size - 1), min(max(c, 0), self.size - 1))

    def _observe(self, true_pos):
        if self.body["sensor"] == "noisy" and rng.random() < 0.25:
            r, c = true_pos
            dr, dc = rng.choice([(-1,0),(1,0),(0,-1),(0,1)])
            cand = self._clamp(r + dr, c + dc)
            if cand not in self.walls:
                return cand
        return true_pos

    def _move_one(self, r, c, dr, dc):
        """1マス移動。壁があれば進めない(その場に留まる)。"""
        nr, nc = self._clamp(r + dr, c + dc)
        if (nr, nc) in self.walls:
            return r, c
        return nr, nc

    def step(self, action):
        if self.body["stability"] == "wobbly" and rng.random() < 0.20:
            action = int(rng.integers(0, 4))
        dr, dc = [(-1,0),(1,0),(0,-1),(0,1)][action]

        if self.body["motor"] == "torque":
            step_len = 1
        else:
            step_len = 2
            if rng.random() < 0.15:
                step_len += rng.choice([-1, 1])

        r, c = self.pos
        for _ in range(max(step_len, 0)):          # 1マスずつ進めて壁を判定
            r, c = self._move_one(r, c, dr, dc)
        self.pos = (r, c)
        self.t += 1

        reached = (self.pos == self.goal)
        in_hazard = (self.pos in self.hazards)
        done = reached or (self.t >= self.max_steps)

        # 固定の課題報酬:1歩-1、ハザード-15、ゴール+20
        reward = -1.0
        if in_hazard:
            reward -= 15.0
        if reached:
            reward += 20.0

        obs = self._observe(self.pos)
        return obs, reward, done, {"reached": reached, "hazard": in_hazard}


# ============================================================
# 強化学習:テーブル型Q学習(v0 と同じ仕組み)
# ============================================================
def train_q(env, episodes=6000, alpha=0.2, gamma=0.95,
            eps_start=1.0, eps_end=0.05):
    n = env.size
    Q = np.zeros((n, n, 4))
    for ep in range(episodes):
        eps = eps_start + (eps_end - eps_start) * (ep / episodes)
        obs = env.reset(); done = False
        while not done:
            r, c = obs
            a = int(rng.integers(0,4)) if rng.random() < eps else int(np.argmax(Q[r,c]))
            nobs, reward, done, _ = env.step(a)
            nr, nc = nobs
            target = reward + (0.0 if done else gamma * np.max(Q[nr, nc]))
            Q[r, c, a] += alpha * (target - Q[r, c, a])
            obs = nobs
    return Q


def evaluate(env, Q, episodes=400):
    scores, succ, steps_ok, hazard_hits = [], 0, [], []
    for _ in range(episodes):
        obs = env.reset(); done = False; hits = 0
        while not done:
            r, c = obs
            a = int(np.argmax(Q[r, c]))
            obs, _, done, info = env.step(a)
            if info["hazard"]:
                hits += 1
        if info["reached"]:
            succ += 1
            steps_ok.append(env.t)
            scores.append(100.0 - env.t - 5.0 * hits)
        else:
            scores.append(-80.0 - 5.0 * hits)
        hazard_hits.append(hits)
    return (np.mean(scores), succ/episodes,
            np.mean(steps_ok) if steps_ok else float("nan"),
            np.mean(hazard_hits))


# ============================================================
# 協調設計ループ:全構成を評価して淘汰
# ============================================================
def run(size=10, cost_weight=6.0):
    walls, hazards = build_map(size)
    keys = list(COMPONENTS.keys())
    bodies = [dict(zip(keys, combo)) for combo in itertools.product(*COMPONENTS.values())]
    out = []
    for body in bodies:
        env = GridWorldV1(size, body, walls, hazards)
        Q = train_q(env)
        perf, sr, avg_steps, hz = evaluate(env, Q)
        cost = body_cost(body)
        out.append({"body": body, "perf": perf, "success": sr,
                    "avg_steps": avg_steps, "hazard": hz,
                    "cost": cost, "fitness": perf - cost_weight*cost})
    out.sort(key=lambda x: x["fitness"], reverse=True)
    return out, (walls, hazards)


if __name__ == "__main__":
    results, (walls, hazards) = run(size=10, cost_weight=6.0)

    print("=" * 82)
    print("協調設計 v1 — 課題を難しくした(10x10 + 狭い通路 + ハザード地帯)")
    print("=" * 82)
    print(f'{"順位":<4}{"motor":<8}{"sensor":<9}{"stability":<10}'
          f'{"成功率":>7}{"平均歩数":>8}{"被弾":>7}{"コスト":>7}{"適応度":>9}')
    print("-" * 82)
    for i, r in enumerate(results, 1):
        b = r["body"]
        steps = f'{r["avg_steps"]:.1f}' if r["avg_steps"]==r["avg_steps"] else "--"
        print(f'{i:<4}{b["motor"]:<8}{b["sensor"]:<9}{b["stability"]:<10}'
              f'{r["success"]*100:>6.0f}%{steps:>8}{r["hazard"]:>7.1f}'
              f'{r["cost"]:>7.0f}{r["fitness"]:>9.1f}')
    print("-" * 82)
    w = results[0]["body"]
    print(f'v1 の勝者: {w["motor"]}/{w["sensor"]}/{w["stability"]}  '
          f'(成功率 {results[0]["success"]*100:.0f}% / コスト {results[0]["cost"]:.0f})')

    # 成功率の散らばり(=天井効果が崩れたか)を確認
    srs = [r["success"] for r in results]
    print(f'\n成功率の範囲: {min(srs)*100:.0f}% 〜 {max(srs)*100:.0f}% '
          f'(v0 は全部100%だった)')
