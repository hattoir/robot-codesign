"""
learnability.py — 「学習しやすさ」を測る

3つの軸で身体を評価する:
  1. 最終性能  final   … 学習後の成功率(従来の指標)
  2. 学習速度  speed   … 成功率が閾値に到達するまでのエピソード数
  3. 学習安定性 reliability … 複数回学習したときの結果のばらつき

「学習速度」の淘汰圧は、現実では何で決まるか:
  - 開発者の忍耐(何回まで試行錯誤に付き合えるか)
  - 周囲の技術革新の速さ(遅いと市場に置いていかれる)
  → これを patience(忍耐予算)というパラメータで表現し、
    予算内に学習が終わらない身体にペナルティを与える。
"""
import json, os, random, sys, statistics as st
import codesign_v2 as V

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS = os.path.join(_ROOT, "results")
os.makedirs(_RESULTS, exist_ok=True)

def learn_curve(genome, walls, hazards, seed, episodes=3000, window=100):
    """学習曲線(100エピソードごとの成功率)を返す"""
    r = V.evaluate_genome(genome, walls, hazards, seed=seed,
                          episodes=episodes, eval_eps=150, track_curve=True)
    return r["curve"], r["success"]

def time_to_threshold(curve, thresh=0.8, window=100):
    """成功率が thresh を初めて超えたエピソード数。到達しなければ None"""
    for i, v in enumerate(curve):
        if v >= thresh:
            return (i+1)*window
    return None

def profile(genome, walls, hazards, n_seeds=8, thresh=0.8, episodes=3000):
    finals, times, reached = [], [], 0
    for k in range(n_seeds):
        curve, final = learn_curve(genome, walls, hazards,
                                   seed=70000+k*7919, episodes=episodes)
        finals.append(final)
        t = time_to_threshold(curve, thresh)
        if t is not None:
            times.append(t); reached += 1
    return {
        "genome": list(genome),
        "final_mean": st.mean(finals),
        "final_sd": st.pstdev(finals),
        "reach_rate": reached/n_seeds,                  # 閾値に到達した割合
        "time_mean": (st.mean(times) if times else None),
        "cost": V.genome_cost(genome),
        "n_seeds": n_seeds,
    }

def fitness_v25(p, cost_weight=6.0, patience=1500, impatience=0.02):
    """学習しやすさを織り込んだ適応度。

    patience   … 許容できる学習エピソード数(忍耐予算)
    impatience … 予算超過1エピソードあたりの減点(技術革新の速さ)

    - 閾値に到達できない身体は、そもそも大きく減点(reach_rate)
    - 予算内に終われば減点なし、超過すると比例して減点
    """
    base = p["final_mean"]*100 - cost_weight*p["cost"]
    base *= p["reach_rate"]                     # 到達できない回は価値ゼロ扱い
    if p["time_mean"] is None:
        return base - 50.0
    over = max(0.0, p["time_mean"] - patience)
    return base - impatience*over

if __name__ == "__main__":
    a = int(sys.argv[1]) if len(sys.argv)>1 else 0
    b = int(sys.argv[2]) if len(sys.argv)>2 else 12
    walls, hz = V.random_map(10, random.Random(9000))
    truth=[json.loads(l) for l in open(os.path.join(_RESULTS, "v2_truth_9000.jsonl"))]
    truth.sort(key=lambda r:-r['fitness'])
    out=open(os.path.join(_RESULTS, "v25_learnability.jsonl"), "a")
    for r in truth[a:b]:
        p = profile(tuple(r["genome"]), walls, hz)
        p["old_fitness"]=r["fitness"]
        out.write(json.dumps(p, ensure_ascii=False)+"\n"); out.flush()
        t = f'{p["time_mean"]:.0f}' if p["time_mean"] else "未到達"
        print(f'{"/".join(p["genome"]):44s} 最終{p["final_mean"]*100:5.1f}%'
              f'±{p["final_sd"]*100:4.1f} 到達率{p["reach_rate"]*100:3.0f}% '
              f'学習{t}ep', flush=True)
    out.close()
