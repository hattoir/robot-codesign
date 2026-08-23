# SETUP —— 実行環境

## 必要なもの

- **Python 3.10 以上**
- ライブラリは `numpy` と `matplotlib` のみ（v0〜v1の範囲）

物理シミュレーションは使っていないので、GPUは不要。ノートPCで十分動く。

---

## セットアップ手順

### 1. Python の確認

```bash
python3 --version
```

`Python 3.10.x` 以上が出ればOK。出なければ https://www.python.org/downloads/ から導入。

### 2. 仮想環境（推奨）

プロジェクト専用の環境を作ると、他と混ざらない。

```bash
cd robot-codesign
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

行頭に `(venv)` が付けば成功。

### 3. 依存関係のインストール

```bash
pip install -r requirements.txt
```

---

## 実行方法

### v0（小グリッド・全数評価）

```bash
python3 src/codesign_v0.py
```

所要時間：数分。8構成のランキング表と `codesign_v0_result.png` が出力される。

**いじってみるポイント**：ファイル末尾の

```python
results = run(size=6, cost_weight=6.0)
```

`cost_weight` を `2.0` や `12.0` に変えると、勝者が入れ替わる。
（※ `perf` は `cost_weight` に依存しないので、学習結果自体は変わらない。順位の計算だけが変わる）

---

### v1（難しい課題）

```bash
python3 src/codesign_v1.py
```

所要時間：10〜20分程度。10×10グリッド、狭い通路、ハザードあり。

---

### 大規模検証（本命）

```bash
# マップ 0〜47 を実行（結果は results/large_results.jsonl に追記される）
python3 src/codesign_large.py 0 48

# 続きから足す場合
python3 src/codesign_large.py 48 100
```

**JSONL に追記される方式**なので、途中で止めても大丈夫。何回かに分けて回せる。

集計：

```bash
python3 src/analyze_large.py
```

---

### 図の生成

```bash
python3 src/make_figures_v0.py     # 第2回の図
python3 src/make_figures_v1.py     # 第3回の図
```

---

## 性能に関するメモ（重要）

**素のPython + list が、numpy より圧倒的に速い。**

最初は numpy で書いていたが、Q学習の内側ループのようにスカラー演算を大量に回す処理では、
numpy の呼び出しオーバーヘッドが致命的だった。

素のPython（`random` モジュール + フラットな `list`）に書き直したら **20倍以上高速化**。
1構成あたり 約0.7秒、1マップ（8構成）で 約5.5秒 になった。

→ `codesign_large.py` がこの高速版。`codesign_v1_robust.py` は初期の低速版（記録として保存）。

numpy が有利なのは「大きな配列をまとめて演算する」場合であって、
「小さな値を大量に読み書きする」場合は素のPythonの方が速いことがある。

---

## 日本語フォント（図の文字化け対策）

matplotlib で日本語ラベルを使う場合、日本語フォントが必要。

```python
plt.rcParams["font.family"] = "Noto Sans CJK JP"
```

フォントが無い環境では□（豆腐）になる。その場合：

- **macOS**：`"Hiragino Sans"` を指定
- **Windows**：`"Yu Gothic"` または `"MS Gothic"` を指定
- **Linux**：`sudo apt install fonts-noto-cjk` で導入

または `pip install japanize-matplotlib` して `import japanize_matplotlib` するだけでも解決する。

---

## トラブルシューティング

| 症状 | 原因と対処 |
|---|---|
| `command not found: python3` | Python 未インストール |
| `pip: command not found` | 仮想環境の有効化を忘れている |
| 図の日本語が□になる | 上記のフォント設定を参照 |
| 実行が終わらない | `EPISODES` や マップ数を減らす |
