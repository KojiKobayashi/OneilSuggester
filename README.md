# OneilSuggester

O'Neil 流のスクリーニング基準を使い、日本株の買いシグナルをスコアリングするツールです。

## Pages

| ブランチ | URL | 更新タイミング |
|---|---|---|
| `main` | https://kojikobayashi.github.io/OneilSuggester/ | 毎朝自動実行（スケジュール） |
| `develop` | https://kojikobayashi.github.io/OneilSuggester/develop/ | `develop` ブランチへのプッシュ時 |

---

## 機能

- 日本株のOHLCVデータを取得し、日本版ロング・CANSLIM-lite ロング・ショートを並行スコアリング
- 各銘柄をスコアリングし、上位銘柄を `docs/data/latest.json` に出力
- GitHub Actions で毎朝自動実行（`main`）、および `develop` ブランチ更新時に自動実行

---

## 開発環境のセットアップ（uv）

本プロジェクトはパッケージマネージャーとして [uv](https://docs.astral.sh/uv/) を使用します。

### 1. uv のインストール

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

または pip 経由でインストール:

```bash
pip install uv
```

インストール後、`uv --version` でバージョンを確認してください。

### 2. 依存パッケージのインストール

```bash
# 通常の依存パッケージ + 開発用パッケージをインストール
uv sync --dev
```

これにより `.venv/` 仮想環境が自動で作成され、`uv.lock` に記録された依存パッケージが再現性を持ってインストールされます。

### 3. スクリプトの実行

```bash
# バッチ処理の実行
uv run python batch/run_daily.py

# テストの実行
uv run pytest
```

### 4. 依存パッケージの追加・更新

```bash
# 依存パッケージを追加する場合
uv add <package>

# 開発用パッケージを追加する場合
uv add --dev <package>

# ロックファイルを更新する場合
uv lock --upgrade
```

---

## 設定ファイル

| ファイル | 説明 |
|---|---|
| `config.yaml` | スクリーニング設定（`top_n`, `lookback_days`, `min_avg_dollar_volume` など） |
| `data/stock_list.csv` | スクリーニング対象の銘柄リスト |
| `docs/data/latest.json` | バッチ処理の出力（スクリーニング結果） |

---

## プロジェクト構成

```
OneilSuggester/
├── .github/workflows/
│   ├── daily.yml            # GitHub Actions（毎朝自動実行・main ブランチ用）
│   ├── develop-pages.yml    # GitHub Actions（develop ブランチ更新時）
│   └── test.yml             # GitHub Actions（テスト自動実行）
├── batch/
│   └── run_daily.py             # バッチ処理エントリーポイント
├── src/
│   ├── fetcher.py               # OHLCVデータ取得
│   ├── indicators.py            # テクニカル指標
│   ├── scoring.py               # 日本版ロング / CANSLIM-lite / ショートの集約
│   └── patterns/                # パターン検出ロジック
├── tests/
│   ├── test_indicators.py       # indicators.py のユニットテスト
│   ├── test_scoring.py          # scoring.py のユニットテスト
│   └── patterns/
│       ├── test_cup_with_handle.py  # カップウィズハンドルパターンのテスト
│       └── test_short_sell.py       # ショートセルパターンのテスト
├── docs/
│   ├── index.html               # フロントエンド（main 用）
│   ├── develop/                 # develop ブランチ用 Pages（自動生成）
│   └── data/latest.json         # スクリーニング結果（自動更新）
├── data/
│   └── stock_list.csv           # 銘柄リスト
├── config.yaml                  # 設定ファイル
├── pyproject.toml               # プロジェクト定義・依存パッケージ
└── uv.lock                      # 依存パッケージのロックファイル
```