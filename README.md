# OneilSuggester

O'Neil 流のスクリーニング基準を使い、日本株の買いシグナルをスコアリングするツールです。

---

## 機能

- 日本株のOHLCVデータを取得し、O'Neil 流パターン（ベース・ブレイクアウト等）を検出
- 各銘柄をスコアリングし、上位銘柄を `docs/data/latest.json` に出力
- GitHub Actions で毎朝自動実行

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
# 依存パッケージを追加する場合（uv.lock も自動更新される）
uv add <package>

# 開発用パッケージを追加する場合（uv.lock も自動更新される）
uv add --dev <package>

# 依存パッケージを削除する場合（uv.lock も自動更新される）
uv remove <package>
```

### uv.lock の更新タイミング

`uv.lock` は **再現性を保証するファイル**です。以下の判断基準で更新してください。

| 状況 | 操作 |
|---|---|
| `uv add` / `uv remove` を実行した | ✅ 自動更新される（手動操作不要） |
| `pyproject.toml` を手動編集した | `uv lock` を実行して更新 |
| 依存パッケージの最新版に上げたい | `uv lock --upgrade` を実行 |
| 特定パッケージだけ最新版に上げたい | `uv lock --upgrade-package <package>` を実行 |
| 既存の環境を再現したいだけ（CI含む） | ❌ 更新不要（`uv sync` で lock の内容がそのまま使われる） |

> **ポイント**: `uv.lock` はコミットして共有することで、チーム全員・CI が同じバージョンを使うことを保証します。意図しないアップグレードを避けるため、更新は慎重に行い、変更内容を PR でレビューしてください。

---

## 設定ファイル

| ファイル | 説明 |
|---|---|
| `config.yaml` | スクリーニング設定（`top_n`, `lookback_days` など） |
| `data/stock_list.csv` | スクリーニング対象の銘柄リスト |
| `docs/data/latest.json` | バッチ処理の出力（スクリーニング結果） |

---

## プロジェクト構成

```
OneilSuggester/
├── .github/workflows/daily.yml  # GitHub Actions（毎朝自動実行）
├── batch/
│   └── run_daily.py             # バッチ処理エントリーポイント
├── src/
│   ├── fetcher.py               # OHLCVデータ取得
│   ├── indicators.py            # テクニカル指標
│   ├── scoring.py               # スコアリング
│   └── patterns/                # パターン検出ロジック
├── docs/
│   ├── index.html               # フロントエンド
│   └── data/latest.json         # スクリーニング結果（自動更新）
├── data/
│   └── stock_list.csv           # 銘柄リスト
├── config.yaml                  # 設定ファイル
├── pyproject.toml               # プロジェクト定義・依存パッケージ
└── uv.lock                      # 依存パッケージのロックファイル
```