"""make_figures_v1.py — 第3回(v1 + 大規模検証)用の図"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FIG = os.path.join(_ROOT, "figures")
os.makedirs(_FIG, exist_ok=True)


plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

INK, SUB = "#22303b", "#7b8794"
ACC, ACC2, LIGHT = "#2a9d8f", "#e76f51", "#bcd7e8"
BG = "#ffffff"


# ============================================================
# 図4:v1単発マップの劇的な結果
# ============================================================
def fig4():
    labels = ["speed\n高精度/不安定", "speed\n安ｾﾝｻ/安定", "speed\n高精度/安定",
              "torque\n高精度/安定", "speed\n安ｾﾝｻ/不安定",
              "torque\n安ｾﾝｻ/不安定", "torque\n安ｾﾝｻ/安定", "torque\n高精度/不安定"]
    succ = [100, 100, 100, 100, 93, 0, 0, 0]
    colors = [ACC if s == 100 else (LIGHT if s > 0 else ACC2) for s in succ]

    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.bar(range(len(succ)), succ, color=colors, edgecolor="white", width=0.7)
    ax.set_xticks(range(len(succ)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("成功率(%)", fontsize=11.5)
    ax.set_ylim(0, 122)
    ax.set_title("v1:課題を難しくしたら、成功率が0〜100%に割れた",
                 fontsize=15, fontweight="bold", pad=16)
    ax.text(6, 30, "v0の勝者が\nここで全滅", fontsize=12, color=ACC2,
            ha="center", fontweight="bold", linespacing=1.5)
    ax.annotate("", xy=(5.6, 8), xytext=(6, 22),
                arrowprops=dict(arrowstyle="-|>", color=ACC2, lw=2))
    ax.text(1.5, 108, "v0では8通り全部が100%だった(天井効果)",
            fontsize=11, color=SUB, ha="center")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(_FIG, "fig4_v1_drama.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    plt.close()


# ============================================================
# 図5:部品ごとの効き目(48マップ平均)
# ============================================================
def fig5():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4),
                                   gridspec_kw={"width_ratios": [1, 1.05]})

    # 左:部品ごとの差
    parts = ["モーター\n(speed vs torque)", "センサー\n(高精度 vs 安い)",
             "安定性\n(安定 vs 不安定)"]
    diffs = [22.4, 14.0, 2.2]
    cols = [ACC, ACC, LIGHT]
    bars = ax1.barh(range(3), diffs, color=cols, edgecolor="white", height=0.6)
    ax1.set_yticks(range(3)); ax1.set_yticklabels(parts, fontsize=11)
    ax1.invert_yaxis()
    ax1.set_xlabel("成功率の差(ポイント)", fontsize=11.5)
    ax1.set_xlim(0, 27)
    ax1.set_title("どの部品にお金を払う価値があるか",
                  fontsize=14, fontweight="bold", pad=14)
    for b, d in zip(bars, diffs):
        ax1.text(d + 0.6, b.get_y() + b.get_height()/2, f"{d:.1f}pt",
                 va="center", fontsize=11, fontweight="bold", color=INK)
    ax1.text(4.5, 2.0, "  ← ほとんど効かない", fontsize=10.5,
             color=SUB, ha="left", va="center")
    for s in ["top", "right"]:
        ax1.spines[s].set_visible(False)
    ax1.grid(axis="x", alpha=0.25); ax1.set_axisbelow(True)

    # 右:平均とばらつき
    names = ["speed/高精度/不安定", "speed/安ｾﾝｻ/不安定", "torque/高精度/安定",
             "speed/高精度/安定", "speed/安ｾﾝｻ/安定", "torque/高精度/不安定",
             "torque/安ｾﾝｻ/安定", "torque/安ｾﾝｻ/不安定"]
    means = [98.7, 97.8, 93.8, 88.7, 83.7, 70.9, 62.4, 52.3]
    sds   = [5.0, 5.1, 24.2, 29.5, 34.7, 43.0, 48.0, 46.3]
    y = np.arange(len(names))
    cols2 = [ACC if "speed" in n else ACC2 for n in names]
    ax2.errorbar(means, y, xerr=sds, fmt="o", color="none",
                 ecolor=SUB, elinewidth=1.6, capsize=4, zorder=2)
    ax2.scatter(means, y, s=110, c=cols2, zorder=3,
                edgecolor="white", lw=1.4)
    ax2.set_yticks(y); ax2.set_yticklabels(names, fontsize=9.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("平均成功率(%) ± ばらつき", fontsize=11.5)
    ax2.set_xlim(-5, 125)
    ax2.set_title("torque型は「100%か0%か」に割れる",
                  fontsize=14, fontweight="bold", pad=14)
    ax2.text(60, 6.9, "ばらつきが極端に大きい", fontsize=10.5,
             color=ACC2, fontweight="bold", ha="center")
    for s in ["top", "right"]:
        ax2.spines[s].set_visible(False)
    ax2.grid(axis="x", alpha=0.25); ax2.set_axisbelow(True)

    fig.suptitle("大規模検証:ランダムマップ48枚 × 8構成 = 384試行",
                 fontsize=15, fontweight="bold", y=1.03, color=INK)
    plt.tight_layout()
    plt.savefig(os.path.join(_FIG, "fig5_components.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    plt.close()


# ============================================================
# 図6:「最強の身体」と「選ばれる身体」のねじれ + 難易度別
# ============================================================
def fig6():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # 左:ねじれ
    strong_labels = ["torque/高精度/安定", "speed/高精度/不安定", "その他"]
    strong_vals = [45, 2, 1]
    sel_labels = ["speed/安ｾﾝｻ/不安定", "torque/安ｾﾝｻ/不安定", "その他"]
    sel_vals = [26, 19, 3]

    x = np.array([0, 1])
    bottom = np.zeros(2)
    palette = [ACC, LIGHT, "#e9edf0"]
    for i in range(3):
        vals = np.array([strong_vals[i], sel_vals[i]])
        ax1.bar(x, vals, bottom=bottom, color=palette[i],
                edgecolor="white", width=0.55)
        for xi, v, b in zip(x, vals, bottom):
            if v >= 2:
                lab = strong_labels[i] if xi == 0 else sel_labels[i]
                ax1.text(xi, b + v/2, f"{lab}\n{v}回", ha="center",
                         va="center", fontsize=9.5,
                         color="white" if i == 0 else INK,
                         fontweight="bold" if i == 0 else "normal",
                         linespacing=1.4)
        bottom += vals
    ax1.set_xticks(x)
    ax1.set_xticklabels(["成績が一番良い身体\n(コスト無視)",
                         "淘汰が選ぶ身体\n(コスト込み)"], fontsize=11.5)
    ax1.set_ylabel("48マップ中の回数", fontsize=11.5)
    ax1.set_title("「一番強い身体」と「選ばれる身体」は別物",
                  fontsize=14, fontweight="bold", pad=14)
    for s in ["top", "right"]:
        ax1.spines[s].set_visible(False)
    ax1.grid(axis="y", alpha=0.25); ax1.set_axisbelow(True)

    # 右:難易度別
    cats = ["易しい\n(29枚)", "中間\n(15枚)", "難しい\n(4枚)"]
    tq = [91.3, 44.0, 11.0]
    sp = [99.0, 84.3, 72.6]
    xx = np.arange(3); w = 0.36
    ax2.bar(xx - w/2, tq, w, label="torque(安い)", color=ACC2,
            edgecolor="white")
    ax2.bar(xx + w/2, sp, w, label="speed(高い)", color=ACC,
            edgecolor="white")
    for i, (t, s) in enumerate(zip(tq, sp)):
        ax2.text(i - w/2, t + 2, f"{t:.0f}", ha="center", fontsize=10)
        ax2.text(i + w/2, s + 2, f"{s:.0f}", ha="center", fontsize=10)
        ax2.text(i, 110, f"差 {s-t:.0f}pt", fontsize=11, color=INK,
                 ha="center", fontweight="bold")
    ax2.set_xticks(xx); ax2.set_xticklabels(cats, fontsize=11)
    ax2.set_ylabel("成功率(%)", fontsize=11.5)
    ax2.set_ylim(0, 124)
    ax2.set_title("環境が厳しいほど、良い部品の価値が上がる",
                  fontsize=14, fontweight="bold", pad=14)
    ax2.legend(fontsize=10.5, frameon=False, loc="lower left",
               bbox_to_anchor=(0.0, 0.02))
    for s in ["top", "right"]:
        ax2.spines[s].set_visible(False)
    ax2.grid(axis="y", alpha=0.25); ax2.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(_FIG, "fig6_twist.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    plt.close()


if __name__ == "__main__":
    fig4(); fig5(); fig6()
    print("saved fig4 / fig5 / fig6")
