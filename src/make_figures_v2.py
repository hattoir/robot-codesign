"""make_figures_v2.py — v2/v2.5 の記事用の図"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, json, os, statistics as st
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
plt.rcParams["axes.unicode_minus"]=False

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS = os.path.join(_ROOT, "results")
os.makedirs(_RESULTS, exist_ok=True)
_FIG = os.path.join(_ROOT, "figures")
os.makedirs(_FIG, exist_ok=True)
INK,SUB="#22303b","#7b8794"; ACC,ACC2,LIGHT="#2a9d8f","#e76f51","#bcd7e8"

# ---- 図7:評価ノイズ(同じ身体でも結果が振れる) ----
def fig7():
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13.5,5.2))
    data={"1回":[92,3,18,7,18],"3回平均":[61,36,25,11,23],
          "5回平均":[47,23,37,10,30],"10回平均":[39,17,29,19,27]}
    xs=list(data.keys())
    for i,(k,v) in enumerate(data.items()):
        ax1.scatter([i]*len(v),v,s=110,c=ACC2,alpha=.75,zorder=3,
                    edgecolor="white",lw=1.3)
        ax1.plot([i-.18,i+.18],[st.mean(v)]*2,color=INK,lw=2.2,zorder=4)
        ax1.annotate("",xy=(i+.3,min(v)),xytext=(i+.3,max(v)),
                     arrowprops=dict(arrowstyle="<->",color=SUB,lw=1.3))
        ax1.text(i+.36,(min(v)+max(v))/2,f"{max(v)-min(v)}pt",fontsize=10,
                 color=SUB,va="center")
    ax1.set_xticks(range(len(xs))); ax1.set_xticklabels(xs,fontsize=11)
    ax1.set_ylabel("推定した成功率(%)",fontsize=11.5); ax1.set_ylim(-5,105)
    ax1.set_title("同じ身体なのに、測るたび結果が変わる",fontsize=14,
                  fontweight="bold",pad=14)
    ax1.text(1.5,97,"何回平均しても、ブレは消えきらない",fontsize=10.5,
             color=ACC2,ha="center",fontweight="bold")
    for s in["top","right"]: ax1.spines[s].set_visible(False)
    ax1.grid(axis="y",alpha=.25); ax1.set_axisbelow(True)

    # 安定 vs 不安定な身体
    rs=[json.loads(l) for l in open(os.path.join(_RESULTS, "v25_learnability.jsonl"))]
    stable=sorted([r for r in rs if r['final_sd']<0.05],
                  key=lambda r:-r['final_mean'])[:5]
    unstable=sorted([r for r in rs if r['final_sd']>0.2],
                    key=lambda r:-r['final_sd'])[:5]
    names=[ "/".join(r['genome'][:3]) for r in stable+unstable]
    means=[r['final_mean']*100 for r in stable+unstable]
    sds=[r['final_sd']*100 for r in stable+unstable]
    cols=[ACC]*len(stable)+[ACC2]*len(unstable)
    y=np.arange(len(names))
    ax2.errorbar(means,y,xerr=sds,fmt="o",color="none",ecolor=SUB,
                 elinewidth=1.6,capsize=4,zorder=2)
    ax2.scatter(means,y,s=110,c=cols,zorder=3,edgecolor="white",lw=1.3)
    ax2.set_yticks(y); ax2.set_yticklabels(names,fontsize=9)
    ax2.invert_yaxis(); ax2.set_xlim(0,130)
    ax2.set_xlabel("最終成功率(%) ± ばらつき",fontsize=11.5)
    ax2.set_title("「毎回学習できる身体」と「運任せの身体」",fontsize=14,
                  fontweight="bold",pad=14)
    ax2.text(112,1.5,"安定",fontsize=11,color=ACC,fontweight="bold")
    ax2.text(112,6.5,"運任せ",fontsize=11,color=ACC2,fontweight="bold")
    for s in["top","right"]: ax2.spines[s].set_visible(False)
    ax2.grid(axis="x",alpha=.25); ax2.set_axisbelow(True)
    plt.tight_layout(); plt.savefig(os.path.join(_FIG, "fig7_noise.png"),dpi=150,
                                    bbox_inches="tight",facecolor="white")
    plt.close()

# ---- 図8:学習速度は独立した軸 ----
def fig8():
    rs=[json.loads(l) for l in open(os.path.join(_RESULTS, "v25_learnability.jsonl"))]
    xs=[r['time_mean'] for r in rs]; ys=[r['final_mean']*100 for r in rs]
    fig,ax=plt.subplots(figsize=(9.5,5.8))
    ax.scatter(xs,ys,s=120,c=ACC2,alpha=.8,zorder=3,edgecolor="white",lw=1.4)
    fast=min(rs,key=lambda r:r['time_mean']); slow=max(rs,key=lambda r:r['time_mean'])
    for r,lab,dx in [(fast,"最速",10),(slow,"最遅",-10)]:
        ax.annotate("/".join(r['genome'][:3]),
                    (r['time_mean'],r['final_mean']*100),fontsize=10,
                    xytext=(dx,14),textcoords="offset points",color=INK,
                    ha="left" if dx>0 else "right",fontweight="bold",
                    arrowprops=dict(arrowstyle="-",color=SUB,lw=1))
    ax.set_xlabel("学習に要したエピソード数 →(遅い)",fontsize=11.5)
    ax.set_ylabel("最終成功率(%)",fontsize=11.5)
    ax.set_title("最終性能が同じでも、学習の速さは2.4倍違う",
                 fontsize=15,fontweight="bold",pad=16)
    ax.axhspan(90,101,color=ACC,alpha=.07,zorder=0)
    ax.text(1500,102,"この帯はどれも「性能はほぼ同じ」",fontsize=10.5,
            color=ACC,ha="center",fontweight="bold")
    ax.set_ylim(70,106)
    for s in["top","right"]: ax.spines[s].set_visible(False)
    ax.grid(alpha=.25); ax.set_axisbelow(True)
    plt.tight_layout(); plt.savefig(os.path.join(_FIG, "fig8_speed.png"),dpi=150,
                                    bbox_inches="tight",facecolor="white")
    plt.close()

# ---- 図9:忍耐予算で答えが変わる ----
def fig9():
    scen=[("じっくり\n(予算3000ep)","torque",4.5,2800,92.9),
          ("標準\n(予算1500ep)","balanced",3.5,1367,95.6),
          ("超短期\n(予算700ep)","speed",7.5,667,97.3)]
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13.5,5.4))
    labs=[s[0] for s in scen]; costs=[s[2] for s in scen]; times=[s[3] for s in scen]
    x=np.arange(3)
    b=ax1.bar(x,costs,color=[ACC,LIGHT,ACC2],edgecolor="white",width=.55)
    for i,(s,c) in enumerate(zip(scen,costs)):
        ax1.text(i,c+.2,f"{s[1]}\nコスト{c}",ha="center",fontsize=11,
                 fontweight="bold",color=INK,linespacing=1.4)
    ax1.set_xticks(x); ax1.set_xticklabels(labs,fontsize=10.5)
    ax1.set_ylabel("選ばれた身体の部品コスト",fontsize=11.5); ax1.set_ylim(0,9.5)
    ax1.set_title("時間がないほど、高い部品が選ばれる",fontsize=14,
                  fontweight="bold",pad=14)
    for s in["top","right"]: ax1.spines[s].set_visible(False)
    ax1.grid(axis="y",alpha=.25); ax1.set_axisbelow(True)

    ax2.bar(x,times,color=[ACC,LIGHT,ACC2],edgecolor="white",width=.55)
    for i,t in enumerate(times):
        ax2.text(i,t+60,f"{t}ep",ha="center",fontsize=11,fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(labs,fontsize=10.5)
    ax2.set_ylabel("学習に要したエピソード数",fontsize=11.5)
    ax2.set_ylim(0,3300)
    ax2.set_title("進化は「速く学べる身体」を選び直した",fontsize=14,
                  fontweight="bold",pad=14)
    for s in["top","right"]: ax2.spines[s].set_visible(False)
    ax2.grid(axis="y",alpha=.25); ax2.set_axisbelow(True)
    fig.suptitle("同じ課題・同じ部品カタログでも、開発時間の制約が変われば答えが変わる",
                 fontsize=14.5,fontweight="bold",y=1.03,color=INK)
    plt.tight_layout(); plt.savefig(os.path.join(_FIG, "fig9_patience.png"),dpi=150,
                                    bbox_inches="tight",facecolor="white")
    plt.close()

if __name__=="__main__":
    fig7(); fig8(); fig9(); print("saved fig7 / fig8 / fig9")
