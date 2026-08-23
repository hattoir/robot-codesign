# robot-codesign

**進化 × 強化学習によるロボットの協調設計**

ロボットの「身体」（どの部品を積むか）を淘汰で選び、「動き方」を強化学習に学ばせる個人研究。
人間が与えるのは **課題（何ができたら成功か）だけ**。身体も動きも設計しない。

着想：リチャード・ドーキンス『利己的な遺伝子』の淘汰シミュレーション。

---

## 研究の構造

```
        人間が与えるのは「課題」だけ
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   【淘汰】                【強化学習】
  どの部品を積むか        その身体をどう動かすか
        └───────────┬───────────┘
                    ▼
        シミュレーションで実行
                    ▼
        適応度 = 性能 − 部品コスト
                    │
                    └──► 良い身体が生き残る
```

**設計の背骨**：エージェントが感じる報酬 `R` と、淘汰の基準になる客観的な適応度 `F` を分ける。
RL は「自分が感じた世界」で報酬を最大化するが、淘汰は「実際どうだったか」で判定する。

---

## 進捗

| 版 | 内容 | 状態 |
|---|---|---|
| v0 | 6×6グリッド・8構成の全数評価 | ✅ 完了 |
| v1 | 10×10・狭い通路・ハザード（課題を難しくする） | ✅ 完了 |
| v1-検証 | ランダムマップ48枚での大規模再検証 | ✅ 完了 |
| v2 | 遺伝的アルゴリズム（GA）の導入 | 🔜 次 |
| v3 | 物理シミュレーション（PyBullet + PPO） | 未着手 |
| v4 | 身体の自由度を開放し、創発を狙う | 未着手 |

詳細は [`docs/RESULTS.md`](docs/RESULTS.md) と [`docs/ROADMAP.md`](docs/ROADMAP.md) を参照。

---

## クイックスタート

```bash
# 1. 仮想環境
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. 依存関係
pip install -r requirements.txt

# 3. v0 を実行（数分）
python3 src/codesign_v0.py
```

詳しい実行方法は [`docs/SETUP.md`](docs/SETUP.md)。

---

## ディレクトリ構成

```
robot-codesign/
├── README.md              このファイル
├── requirements.txt       依存ライブラリ
├── src/                   実験コード
│   ├── codesign_v0.py         v0：小グリッド・全数評価
│   ├── codesign_v1.py         v1：難しい課題
│   ├── codesign_v1_robust.py  v1検証（初期版・低速）
│   ├── codesign_large.py      大規模検証（高速版・本命）
│   ├── analyze_large.py       集計スクリプト
│   ├── make_figures_v0.py     記事用の図（第2回）
│   └── make_figures_v1.py     記事用の図（第3回）
├── docs/                  ドキュメント
│   ├── SETUP.md               実行環境の作り方
│   ├── DESIGN.md              設計思想・各部品の意味
│   ├── RESULTS.md             実験結果のログ
│   └── ROADMAP.md             今後の計画
├── results/               生データ
│   └── large_results.jsonl    48マップ×8構成 = 384試行
├── figures/               生成した図
└── articles/              note記事の原稿
```

---

## これまでに分かったこと（要約）

1. **課題が簡単だと、部品の差は成績に出ない**（v0の天井効果）
2. **1枚のマップの結果は信用できない**（v1の劇的な結果は、48枚で検証したらほぼ幻だった）
3. **部品ごとに効き目が違う**：モーター22.4pt > センサー14.0pt > 安定性2.2pt
4. **「一番強い身体」と「淘汰が選ぶ身体」は別物**（コストを入れると勝者が入れ替わる）
5. **環境が厳しいほど、良い部品の価値が上がる**（易しい環境では差8pt、難しい環境では差62pt）

---

## 記事

- 第1回：導入（`articles/note_article_01.md`）
- 第2回：v0 —— 一番安いロボットが勝った（`articles/note_article_02_v0.md`）
- 第3回：v1と大規模検証 —— 派手な結果の8割は幻だった（`articles/note_article_03_v1.md`）

---

## 参考

- Aggelos Psiris et al. (2026) *Foundation Models in Robotics: A Comprehensive Review*, arXiv:2604.15395
- Richard Dawkins, *The Selfish Gene*
- 関連分野：hardware-software co-design / morphology-control co-optimization / Karl Sims "Evolving Virtual Creatures" / MAP-Elites
