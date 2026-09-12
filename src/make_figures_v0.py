"""
make_figures.py — note記事用の図を3枚生成する
  fig1: 研究の仕組み(身体=淘汰 / 動き=強化学習)の概念図
  fig2: v0の淘汰ランキング結果
  fig3: 天井効果(=課題が簡単すぎた証拠)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FIG = os.path.join(_ROOT, "figures")
os.makedirs(_FIG, exist_ok=True)


# 日本語フォント:環境にあるものを上から順に採用する
# (Ubuntu では Noto CJK、Windows では Yu Gothic などが選ばれる)
import matplotlib.font_manager as _fm
_JP_CANDIDATES = ["Noto Sans CJK JP", "Yu Gothic", "Meiryo",
                  "IPAexGothic", "Noto Sans JP", "MS Gothic"]
_AVAIL = {f.name for f in _fm.fontManager.ttflist}
_JP = [n for n in _JP_CANDIDATES if n in _AVAIL]
if not _JP:
    print("[warn] 日本語フォントが見つかりません。図のラベルが文字化けします。")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = _JP + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 配色
INK   = "#22303b"
SUB   = "#7b8794"
ACC   = "#2a9d8f"   # 勝者・強調
ACC2  = "#e76f51"   # 対比
LIGHT = "#bcd7e8"
BG    = "#ffffff"

# v0 の実測結果
results = [
    # label, motor, sensor, stab, success, steps, cost, perf, fitness
    ("トルク/安ｾﾝｻ/不安定",  100, 12.5, 1, 87.5, 81.5),
    ("スピード/安ｾﾝｻ/不安定", 100,  7.3, 2, 92.7, 80.7),
    ("トルク/高精度ｾﾝｻ...",   100, 10.1, 3, 89.9, 71.9),
    ("スピード/安ｾﾝｻ/安定",  100,  5.8, 4, 94.2, 70.2),
    ("トルク/高精度/不安定",  100, 12.3, 3, 87.7, 69.7),
    ("スピード/高精度/不安定",100,  7.0, 4, 93.1, 69.0),
    ("トルク/高精度/安定",    100, 10.0, 5, 90.0, 60.0),
    ("スピード/高精度/安定",  100,  5.7, 6, 94.3, 58.3),
]
labels   = [r[0] for r in results]
success  = [r[1] for r in results]
steps    = [r[2] for r in results]
costs    = [r[3] for r in results]
perfs    = [r[4] for r in results]
fitness  = [r[5] for r in results]


# ============================================================
# 図1:仕組みの概念図
# ============================================================
def fig1():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=12, tc=INK, bold=False):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1.2",
                           fc=fc, ec=ec, lw=1.8, zorder=2)
        ax.add_patch(b)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fs, color=tc, zorder=3,
                fontweight="bold" if bold else "normal", linespacing=1.5)

    def arrow(x1, y1, x2, y2, color=SUB, style="-|>"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=18, lw=1.8,
                                     color=color, zorder=1))

    ax.text(50, 56, "人間が与えるのは「課題」だけ",
            ha="center", fontsize=15, fontweight="bold", color=INK)

    # 人間の入力
    box(36, 45, 28, 7, "課題:速くゴールへ", "#fdf1e7", ACC2, 12, INK, True)

    # 左:淘汰(身体)
    box(4, 26, 34, 11,
        "【淘汰】\nどの部品を積むか(身体)", "#e8f4f1", ACC, 12, INK, True)
    # 右:強化学習(動き)
    box(62, 26, 34, 11,
        "【強化学習】\nその身体をどう動かすか", "#e8f0f7", "#4a7ba7", 12, INK, True)

    arrow(45, 45, 26, 38)
    arrow(55, 45, 76, 38)

    # 中央:シミュレーション
    box(30, 12, 40, 9, "シミュレーションで実行 → 客観的な成績", BG, SUB, 12)
    arrow(21, 26, 40, 22)
    arrow(79, 26, 60, 22)

    # 結果 → 淘汰に戻る
    box(33, 1, 34, 7, "適応度 = 性能 − 部品コスト", "#fdf1e7", ACC2, 12, INK, True)
    arrow(50, 12, 50, 8.5)

    # フィードバックの矢印(適応度 → 淘汰)
    ax.add_patch(FancyArrowPatch((33, 4.5), (10, 4.5),
                                 arrowstyle="-|>", mutation_scale=18,
                                 lw=1.8, color=ACC, zorder=1))
    ax.add_patch(FancyArrowPatch((10, 4.5), (10, 26),
                                 arrowstyle="-|>", mutation_scale=18,
                                 lw=1.8, color=ACC, zorder=1))
    ax.text(11.5, 15, "良い身体が\n生き残る", fontsize=11, color=ACC,
            va="center", fontweight="bold", linespacing=1.4)

    ax.set_ylim(-6, 60)
    ax.text(50, -3.5, "身体も動きも、人間は設計しない",
            ha="center", fontsize=12, color=SUB, style="italic")

    plt.tight_layout()
    plt.savefig(os.path.join(_FIG, "fig1_concept.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    plt.close()


# ============================================================
# 図2:淘汰ランキング
# ============================================================
def fig2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # --- 左:適応度ランキング ---
    colors = [ACC if i == 0 else LIGHT for i in range(len(results))]
    y = np.arange(len(results))
    bars = ax1.barh(y, fitness, color=colors, edgecolor="white", height=0.72)
    ax1.set_yticks(y); ax1.set_yticklabels(labels, fontsize=10.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("適応度(性能 − コスト)", fontsize=11.5)
    ax1.set_title("淘汰の結果:一番『安い』構成が勝った",
                  fontsize=14, fontweight="bold", pad=14)
    ax1.set_xlim(0, 95)
    for i, (b, f, c) in enumerate(zip(bars, fitness, costs)):
        ax1.text(f + 1.5, b.get_y() + b.get_height()/2,
                 f"{f:.1f}  (コスト{c})", va="center", fontsize=9.5,
                 color=INK if i == 0 else SUB,
                 fontweight="bold" if i == 0 else "normal")
    ax1.text(2, 0.02, "◀ 勝者", fontsize=10.5, color="white",
             fontweight="bold", va="center")
    for s in ["top", "right"]:
        ax1.spines[s].set_visible(False)
    ax1.grid(axis="x", alpha=0.25)
    ax1.set_axisbelow(True)

    # --- 右:性能 vs コスト ---
    ax2.scatter(costs, perfs, s=150, c=[ACC if i == 0 else ACC2
                for i in range(len(results))], zorder=3, edgecolor="white", lw=1.5)
    for lab, x, yv in zip(labels, costs, perfs):
        short = lab.split("/")[0]
        ax2.annotate(short, (x, yv), fontsize=9, color=SUB,
                     xytext=(6, 5), textcoords="offset points")
    ax2.set_xlabel("部品コスト(高性能ほど高い) →", fontsize=11.5)
    ax2.set_ylabel("タスク性能", fontsize=11.5)
    ax2.set_title("高い部品を積んでも、性能はほぼ横ばい",
                  fontsize=14, fontweight="bold", pad=14)
    ax2.set_ylim(84, 98)
    ax2.axhspan(87, 95, color=ACC2, alpha=0.07, zorder=0)
    ax2.text(3.5, 96.3, "性能差はわずか約7点しかない",
             fontsize=10.5, color=ACC2, ha="center", fontweight="bold")
    for s in ["top", "right"]:
        ax2.spines[s].set_visible(False)
    ax2.grid(alpha=0.25)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(_FIG, "fig2_ranking.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    plt.close()


# ============================================================
# 図3:天井効果(課題が簡単すぎた証拠)
# ============================================================
def fig3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6))

    x = np.arange(len(results))
    short = [l.split("/")[0] + "\n" + ("安ｾﾝｻ" if "安ｾﾝｻ" in l else "高精度")
             for l in labels]

    # 左:成功率が全部100%
    ax1.bar(x, success, color=LIGHT, edgecolor="white", width=0.68)
    ax1.axhline(100, color=ACC2, ls="--", lw=2, zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short, fontsize=8.5)
    ax1.set_ylabel("成功率(%)", fontsize=11.5)
    ax1.set_ylim(0, 118)
    ax1.set_title("8通り すべて成功率100%", fontsize=14,
                  fontweight="bold", pad=14)
    ax1.text(3.5, 107, "全部が天井に張り付いている = 差がつかない",
             ha="center", fontsize=11, color=ACC2, fontweight="bold")
    for s in ["top", "right"]:
        ax1.spines[s].set_visible(False)
    ax1.grid(axis="y", alpha=0.25); ax1.set_axisbelow(True)

    # 右:性能の分布がいかに狭いか
    ax2.scatter(perfs, [1]*len(perfs), s=180, c=ACC2, alpha=0.75,
                zorder=3, edgecolor="white", lw=1.5)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0.5, 1.5)
    ax2.set_yticks([])
    ax2.set_xlabel("タスク性能", fontsize=11.5)
    ax2.set_title("性能スコアも、この狭さに団子", fontsize=14,
                  fontweight="bold", pad=14)
    ax2.annotate("", xy=(87, 1.22), xytext=(95, 1.22),
                 arrowprops=dict(arrowstyle="<->", color=INK, lw=1.6))
    ax2.text(91, 1.29, "8体すべてがこの中", ha="center",
             fontsize=11, color=INK, fontweight="bold")
    ax2.text(45, 0.72, "← 本来ならここまで差が開いてほしい",
             fontsize=11, color=SUB, ha="center")
    for s in ["top", "right", "left"]:
        ax2.spines[s].set_visible(False)
    ax2.grid(axis="x", alpha=0.25); ax2.set_axisbelow(True)

    fig.suptitle("課題が簡単すぎて、部品の良し悪しが成績に出ていない",
                 fontsize=15, fontweight="bold", y=1.02, color=INK)
    plt.tight_layout()
    plt.savefig(os.path.join(_FIG, "fig3_ceiling.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    plt.close()


if __name__ == "__main__":
    fig1(); fig2(); fig3()
    print("saved: fig1_concept.png / fig2_ranking.png / fig3_ceiling.png")
