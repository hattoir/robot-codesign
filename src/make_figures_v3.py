"""make_figures_v3.py — 役割別・学習しやすさの図"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, json, os, statistics as st
from collections import defaultdict
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

# 図10:役割ごとの推奨構成
def fig10():
    roles=[("速度重視","balanced/cheap/wobbly",16.9,1.30,4.0,"3/3"),
           ("安全重視","torque/precise/stable",23.9,0.00,7.5,"3/4"),
           ("省エネ","balanced/cheap/wobbly",17.8,1.33,4.0,"3/3")]
    fig,axes=plt.subplots(1,3,figsize=(14,5))
    x=np.arange(3); labs=[r[0] for r in roles]
    cols=[LIGHT,ACC2,LIGHT]
    for ax,(vals,title,ylab,fmt) in zip(axes,[
        ([r[2] for r in roles],"歩数(短いほど速い)","平均歩数","{:.1f}"),
        ([r[3] for r in roles],"被弾(ハザードを踏んだ回数)","1エピソードあたり","{:.2f}"),
        ([r[4] for r in roles],"部品コスト","コスト","{:.1f}")]):
        ax.bar(x,vals,color=cols,edgecolor="white",width=.55)
        for i,v in enumerate(vals):
            ax.text(i,v+max(vals)*0.03,fmt.format(v),ha="center",
                    fontsize=12,fontweight="bold",color=INK)
        ax.set_xticks(x); ax.set_xticklabels(labs,fontsize=11)
        ax.set_ylabel(ylab,fontsize=11); ax.set_title(title,fontsize=13,
                      fontweight="bold",pad=12)
        ax.set_ylim(0,max(vals)*1.25 if max(vals)>0 else 1)
        for s in["top","right"]: ax.spines[s].set_visible(False)
        ax.grid(axis="y",alpha=.25); ax.set_axisbelow(True)
    axes[1].text(1,0.35,"0.00\n一度も踏まない",ha="center",fontsize=11,
                 color=ACC2,fontweight="bold",linespacing=1.4)
    fig.suptitle("同じ部品カタログでも、役割が変われば選ばれる身体が変わる",
                 fontsize=15,fontweight="bold",y=1.02,color=INK)
    plt.tight_layout()
    plt.savefig(os.path.join(_FIG, "fig10_roles.png"),dpi=150,
                bbox_inches="tight",facecolor="white"); plt.close()

# 図11:部品が学習しやすさに与える影響
def fig11():
    rs=[json.loads(l) for l in open(os.path.join(_RESULTS, "v25_learnability.jsonl"))]
    KEYS=['motor','sensor','stability','battery','weight']
    NAMES={'motor':'モーター','sensor':'センサー','stability':'安定性',
           'battery':'バッテリー','weight':'車体重量'}
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13.5,5.6))
    # 学習速度
    ypos=0; yt=[]; yl=[]
    for i,k in enumerate(KEYS):
        d=defaultdict(list)
        for r in rs:
            if r['time_mean']: d[r['genome'][i]].append(r['time_mean'])
        items=sorted(d.items(),key=lambda kv:st.mean(kv[1]))
        for v,arr in items:
            ax1.barh(ypos,st.mean(arr),color=ACC if v==items[0][0] else LIGHT,
                     edgecolor="white",height=.7)
            yt.append(ypos); yl.append(f'{NAMES[k]}:{v}')
            ypos+=1
        ypos+=0.6
    ax1.set_yticks(yt); ax1.set_yticklabels(yl,fontsize=9)
    ax1.invert_yaxis(); ax1.set_xlabel("学習に要したエピソード数(短いほど良い)",fontsize=11)
    ax1.set_title("どの部品が「速く学べる」か",fontsize=13.5,fontweight="bold",pad=12)
    for s in["top","right"]: ax1.spines[s].set_visible(False)
    ax1.grid(axis="x",alpha=.25); ax1.set_axisbelow(True)
    # 安定性
    ypos=0; yt=[]; yl=[]
    for i,k in enumerate(KEYS):
        d=defaultdict(list)
        for r in rs: d[r['genome'][i]].append(r['final_sd']*100)
        items=sorted(d.items(),key=lambda kv:st.mean(kv[1]))
        for v,arr in items:
            ax2.barh(ypos,st.mean(arr),color=ACC if v==items[0][0] else LIGHT,
                     edgecolor="white",height=.7)
            yt.append(ypos); yl.append(f'{NAMES[k]}:{v}')
            ypos+=1
        ypos+=0.6
    ax2.set_yticks(yt); ax2.set_yticklabels(yl,fontsize=9)
    ax2.invert_yaxis(); ax2.set_xlabel("結果のばらつき(小さいほど毎回学習できる)",fontsize=11)
    ax2.set_title("どの部品が「安定して学べる」か",fontsize=13.5,
                  fontweight="bold",pad=12)
    for s in["top","right"]: ax2.spines[s].set_visible(False)
    ax2.grid(axis="x",alpha=.25); ax2.set_axisbelow(True)
    fig.suptitle(f"部品と「学習しやすさ」の関係({len(rs)}構成をプロファイル)",
                 fontsize=14.5,fontweight="bold",y=1.02,color=INK)
    plt.tight_layout()
    plt.savefig(os.path.join(_FIG, "fig11_components_learn.png"),dpi=150,
                bbox_inches="tight",facecolor="white"); plt.close()

if __name__=="__main__":
    fig10(); fig11(); print("saved fig10 / fig11")
