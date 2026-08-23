"""analyze_large.py — 大規模検証(48マップ)の集計"""
import os, json, statistics as st
from collections import defaultdict

recs = [json.loads(l) for l in open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "large_results.jsonl"))]
N_MAPS = len({r["map"] for r in recs})
COST_W = 6.0

def key(r): return f'{r["motor"]}/{r["sensor"]}/{r["stab"]}'

print("="*76)
print(f"大規模検証:ランダムマップ {N_MAPS} 枚 × 8構成 = {len(recs)} 試行")
print("(学習5000エピソード / 評価400エピソード)")
print("="*76)

# --- 構成ごとの平均成功率 ---
by_body = defaultdict(list)
for r in recs: by_body[key(r)].append(r["success"])

print("\n【構成ごとの平均成功率】")
rows = sorted(by_body.items(), key=lambda kv: st.mean(kv[1]), reverse=True)
for k, v in rows:
    m = st.mean(v); sd = st.pstdev(v)
    print(f"  {k:<26} {m*100:5.1f}%  ±{sd*100:4.1f}  {'█'*int(m*22)}")

# --- 部品ごとの効果(他を平均して比較)---
print("\n【部品ごとの平均成功率(その部品を積んだ全構成の平均)】")
for field, opts in [("motor",["torque","speed"]),
                    ("sensor",["precise","noisy"]),
                    ("stab",["stable","wobbly"])]:
    print(f"  -- {field} --")
    means = {}
    for o in opts:
        v = [r["success"] for r in recs if r[field]==o]
        means[o] = st.mean(v)
        print(f"     {o:<9} {st.mean(v)*100:5.1f}%  (n={len(v)})")
    diff = abs(means[opts[0]]-means[opts[1]])*100
    print(f"     → 差: {diff:4.1f} ポイント")

# --- 適応度(コスト込み)での勝者 ---
print("\n【適応度 = 成功率×100 − コスト×6 での勝者(マップごと)】")
wins = defaultdict(int)
per_map_best = {}
for m in sorted({r["map"] for r in recs}):
    sub = [r for r in recs if r["map"]==m]
    scored = sorted(sub, key=lambda r: r["success"]*100 - COST_W*r["cost"],
                    reverse=True)
    wins[key(scored[0])] += 1
    per_map_best[m] = key(scored[0])
for k, c in sorted(wins.items(), key=lambda kv:-kv[1]):
    print(f"  {k:<26} {c:2d}/{N_MAPS} 回 ({c/N_MAPS*100:4.1f}%)  {'▇'*c}")
print(f"\n  → 優勝したことがある構成: {len(wins)} / 8 種類")

# --- 「常に最強の身体」は存在するか ---
print("\n【成功率だけで見た場合のマップごとのベスト(コスト無視)】")
wins2 = defaultdict(int)
for m in sorted({r["map"] for r in recs}):
    sub = [r for r in recs if r["map"]==m]
    best = max(sub, key=lambda r: r["success"])
    wins2[key(best)] += 1
for k, c in sorted(wins2.items(), key=lambda kv:-kv[1]):
    print(f"  {k:<26} {c:2d}/{N_MAPS} 回")
print(f"\n  → ベストになったことがある構成: {len(wins2)} / 8 種類")

# --- 環境の難しさで層別 ---
print("\n【マップの難易度別:torque と speed の成功率】")
diff_by_map = {}
for m in sorted({r["map"] for r in recs}):
    sub = [r for r in recs if r["map"]==m]
    diff_by_map[m] = st.mean([r["success"] for r in sub])
easy = [m for m,v in diff_by_map.items() if v >= 0.8]
hard = [m for m,v in diff_by_map.items() if v < 0.5]
mid  = [m for m,v in diff_by_map.items() if 0.5 <= v < 0.8]
for name, ms in [("易しい(平均成功率80%以上)", easy),
                 ("中間(50〜80%)", mid),
                 ("難しい(50%未満)", hard)]:
    if not ms: continue
    tq = st.mean([r["success"] for r in recs if r["map"] in ms and r["motor"]=="torque"])
    sp = st.mean([r["success"] for r in recs if r["map"] in ms and r["motor"]=="speed"])
    cheap = st.mean([r["success"] for r in recs if r["map"] in ms and r["cost"]<=2])
    rich  = st.mean([r["success"] for r in recs if r["map"] in ms and r["cost"]>=5])
    print(f"  {name}  マップ数={len(ms)}")
    print(f"     torque {tq*100:5.1f}%  /  speed {sp*100:5.1f}%")
    print(f"     安い構成(コスト≦2) {cheap*100:5.1f}%  /  高い構成(コスト≧5) {rich*100:5.1f}%")

# --- v1単発マップとの比較 ---
tq_zero = sum(1 for r in recs if r["motor"]=="torque" and r["success"]==0)
tq_all  = sum(1 for r in recs if r["motor"]=="torque")
print(f"\n【前回v1の「torque全滅」は再現したか】")
print(f"  torque構成が成功率0%だった割合: {tq_zero}/{tq_all} "
      f"({tq_zero/tq_all*100:.1f}%)")
print(f"  → 前回の1枚のマップでは torque の 3/4 が 0% だった")
