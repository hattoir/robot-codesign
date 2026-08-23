"""
codesign_v0.py
------------------------------------------------------------
進化 × 強化学習によるロボット協調設計 —— 抽象版 v0

考え方:
  - 課題(タスク)と報酬は固定。 => 「速くゴールに着け」
  - "身体" = どの部品を積むか(離散の組み合わせ)。これが遺伝子。
  - 各身体ごとに、その身体の動かし方を強化学習(RL)が学習する。
  - できあがった挙動を、客観的な適応度(fitness)で評価する。
  - v0 では部品の組み合わせが少ないので「全数評価」= 淘汰の1世代分。
    (組み合わせが多すぎて全部試せなくなったら、次段でGA=進化に切り替える)

ポイント:
  - RL が最適化するのは「エージェントが感じる報酬」(ノイズありセンサ越し)。
  - fitness が測るのは「客観的な結果」(真の状態で測る)。
    この2つを分けるのが設計の背骨。
------------------------------------------------------------
"""

import numpy as np
import itertools

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FIG = os.path.join(_ROOT, "figures")
os.makedirs(_FIG, exist_ok=True)


rng = np.random.default_rng(0)

# ============================================================
# 部品カタログ(遺伝子の候補)とコスト
# ============================================================
COMPONENTS = {
    "motor":     ["torque", "speed"],     # トルク型(1歩・確実) / スピード型(2歩・稀に滑る)
    "sensor":    ["precise", "noisy"],    # 高精度 / 安いがノイズ多め
    "stability": ["stable", "wobbly"],    # 安定 / たまに動作がブレる
}

# 部品コスト(高性能ほど高い)。fitness で performance からこれを引く。
COST = {
    ("motor", "torque"): 1.0, ("motor", "speed"): 2.0,
    ("sensor", "precise"): 2.0, ("sensor", "noisy"): 0.0,
    ("stability", "stable"): 2.0, ("stability", "wobbly"): 0.0,
}

def body_cost(body):
    return sum(COST[(k, v)] for k, v in body.items())

# ============================================================
# 環境:小さなグリッドワールド
#   部品の選択が「ダイナミクス(動きの物理)」を変える。
# ============================================================
class GridWorld:
    def __init__(self, size, body, max_steps=50):
        self.size = size
        self.body = body
        self.max_steps = max_steps
        self.goal = (size - 1, size - 1)

    def reset(self):
        self.pos = (0, 0)
        self.t = 0
        return self._observe(self.pos)

    def _clamp(self, r, c):
        return (min(max(r, 0), self.size - 1), min(max(c, 0), self.size - 1))

    def _observe(self, true_pos):
        """センサ部品の効果:noisy だと稀に隣のマスに見間違える。"""
        if self.body["sensor"] == "noisy" and rng.random() < 0.25:
            r, c = true_pos
            dr, dc = rng.choice([(-1,0),(1,0),(0,-1),(0,1)])
            return self._clamp(r + dr, c + dc)
        return true_pos

    def step(self, action):
        # action: 0=up 1=down 2=left 3=right
        # stability の効果:wobbly だと稀に行動がランダムに化ける
        if self.body["stability"] == "wobbly" and rng.random() < 0.20:
            action = int(rng.integers(0, 4))

        dr, dc = [(-1,0),(1,0),(0,-1),(0,1)][action]

        # motor の効果:歩幅と滑り
        if self.body["motor"] == "torque":
            step_len = 1
        else:  # speed:2歩進むが、稀に1 or 3 歩(滑り→行き過ぎ/届かず)
            step_len = 2
            if rng.random() < 0.15:
                step_len += rng.choice([-1, 1])

        r, c = self.pos
        self.pos = self._clamp(r + dr * step_len, c + dc * step_len)
        self.t += 1

        reached = (self.pos == self.goal)
        done = reached or (self.t >= self.max_steps)

        # 固定の課題報酬:1歩ごとに-1、ゴールで+10(=速く着くほど良い)
        reward = 10.0 if reached else -1.0

        obs = self._observe(self.pos)          # エージェントが「感じる」状態
        return obs, reward, done, {"true_pos": self.pos, "reached": reached}


# ============================================================
# 強化学習(制御の学習):テーブル型 Q学習
#   その身体に合った動かし方を、身体ごとに学ぶ。
# ============================================================
def train_q(env, episodes=2500, alpha=0.2, gamma=0.95,
            eps_start=1.0, eps_end=0.05):
    n = env.size
    Q = np.zeros((n, n, 4))
    for ep in range(episodes):
        eps = eps_start + (eps_end - eps_start) * (ep / episodes)
        obs = env.reset()
        done = False
        while not done:
            r, c = obs
            if rng.random() < eps:
                a = int(rng.integers(0, 4))
            else:
                a = int(np.argmax(Q[r, c]))
            nobs, reward, done, _ = env.step(a)
            nr, nc = nobs
            target = reward + (0.0 if done else gamma * np.max(Q[nr, nc]))
            Q[r, c, a] += alpha * (target - Q[r, c, a])   # 感じた状態で更新
            obs = nobs
    return Q


def evaluate(env, Q, episodes=300):
    """客観的な適応度を測る。
       行動は身体(ノイズありセンサ)越しに選ぶが、結果は真の状態で測る。"""
    scores, successes, steps_on_success = [], 0, []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            r, c = obs
            a = int(np.argmax(Q[r, c]))     # 学習済み方策(貪欲)
            obs, _, done, info = env.step(a)
        if info["reached"]:
            successes += 1
            steps_on_success.append(env.t)
            scores.append(100.0 - env.t)    # 速いほど高得点
        else:
            scores.append(-50.0)            # 失敗はペナルティ
    success_rate = successes / episodes
    avg_steps = np.mean(steps_on_success) if steps_on_success else float("nan")
    return np.mean(scores), success_rate, avg_steps


# ============================================================
# 協調設計ループ:全部品構成を評価して淘汰(ランキング)
# ============================================================
def run(size=6, cost_weight=6.0):
    keys = list(COMPONENTS.keys())
    all_bodies = [dict(zip(keys, combo))
                  for combo in itertools.product(*COMPONENTS.values())]

    results = []
    for body in all_bodies:
        env = GridWorld(size=size, body=body)
        Q = train_q(env)
        perf, sr, avg_steps = evaluate(env, Q)
        cost = body_cost(body)
        fitness = perf - cost_weight * cost
        results.append({
            "body": body, "perf": perf, "success": sr,
            "avg_steps": avg_steps, "cost": cost, "fitness": fitness,
        })

    results.sort(key=lambda x: x["fitness"], reverse=True)
    return results


def fmt_body(b):
    return f'{b["motor"]:<7} {b["sensor"]:<8} {b["stability"]:<7}'


if __name__ == "__main__":
    results = run(size=6, cost_weight=6.0)

    print("=" * 74)
    print("協調設計 v0 — 全部品構成を強化学習 → 客観適応度でランキング")
    print("(課題:6x6グリッドを速くゴールへ。身体も動きも人間は設計しない)")
    print("=" * 74)
    header = f'{"順位":<4}{"motor":<8}{"sensor":<9}{"stability":<10}' \
             f'{"成功率":>7}{"平均歩数":>9}{"コスト":>7}{"適応度":>9}'
    print(header)
    print("-" * 74)
    for i, r in enumerate(results, 1):
        b = r["body"]
        steps = f'{r["avg_steps"]:.1f}' if r["avg_steps"] == r["avg_steps"] else "--"
        print(f'{i:<4}{b["motor"]:<8}{b["sensor"]:<9}{b["stability"]:<10}'
              f'{r["success"]*100:>6.0f}%{steps:>9}{r["cost"]:>7.0f}{r["fitness"]:>9.1f}')
    print("-" * 74)
    win = results[0]["body"]
    print(f'淘汰の勝者: {fmt_body(win)}  '
          f'(コスト {results[0]["cost"]:.0f} / 適応度 {results[0]["fitness"]:.1f})')

    # ---- 可視化:適応度ランキングと「性能 vs コスト」 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [f'{r["body"]["motor"][:3]}/{r["body"]["sensor"][:3]}/'
                  f'{r["body"]["stability"][:3]}' for r in results]
        fit = [r["fitness"] for r in results]
        perf = [r["perf"] for r in results]
        cost = [r["cost"] for r in results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        colors = ["#2a9d8f" if i == 0 else "#8ecae6" for i in range(len(results))]
        ax1.barh(range(len(results)), fit, color=colors)
        ax1.set_yticks(range(len(results)))
        ax1.set_yticklabels(labels, fontsize=9)
        ax1.invert_yaxis()
        ax1.set_xlabel("fitness")
        ax1.set_title("Selection ranking of body (component) combos")
        ax1.axvline(0, color="#999", lw=0.8)

        ax2.scatter(cost, perf, s=90, c="#e76f51", zorder=3)
        for r, x, y in zip(results, cost, perf):
            lab = f'{r["body"]["motor"][:3]}/{r["body"]["sensor"][:3]}/{r["body"]["stability"][:3]}'
            ax2.annotate(lab, (x, y), fontsize=8,
                         xytext=(4, 4), textcoords="offset points")
        ax2.set_xlabel("cost (pricier parts ->)")
        ax2.set_ylabel("performance")
        ax2.set_title("Performance vs Cost trade-off")
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(_FIG, "codesign_v0_result.png"), dpi=130)
        print("\n[chart] figures/ に保存しました")
    except Exception as e:
        print("可視化はスキップ:", e)
