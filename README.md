# 法令検索AIアシスタント (Legal Search AI Assistant)

日本の法令を検索し、法的質問に回答するAIアシスタントアプリケーションです。e-Gov法令APIを活用して法令データを取得し、Gemini APIを使った自然言語処理によってユーザーの質問に対して関連法令を特定・回答します。

## プロジェクトの背景

- GeminiAPIと[法令API](https://digital-gov.note.jp/n/n7a1b35e58969)を組み合わせた実験プロジェクト
- 法律学科知的財産権ゼミ所属の卒業論文研究の一環として開発
- 該当条文を効率的に見つけることを目的
- 抽象的な質問を入力すると、該当する法分野や条文を自動的に特定して出力

## 機能

- 自然言語での法律質問に対する回答生成
- 関連法令の自動特定
- 複数の法令からの統合回答
- 法令条文のキャッシュ機能
- 条文構造の解析と関連条文の抽出

## 必要条件

- Python 3.8以上
- インターネット接続 (e-Gov法令APIおよびGoogle Gemini APIへのアクセス用)
- Google API Key (Gemini API用)

## インストール方法

1. リポジトリをクローン:
```
git clone https://github.com/YourUsername/legalAI
cd legalAI
```

2. 必要なライブラリをインストール:
```
pip install requests google-generativeai python-dotenv streamlit lxml
```

3. `.env`ファイルで、Google APIキーを設定:
```
GOOGLE_API_KEY=your_google_api_key_here
```

## 使用方法

1. アプリケーションを起動:
```
streamlit run app.py
```

2. Webブラウザが開き、アプリケーションUIが表示されます
3. 質問を入力し「検索」ボタンをクリック
4. AIが関連法令を特定し、回答を生成します

## 技術的詳細

### 使用API

- **e-Gov法令API**: 日本の法令データ取得用 (https://laws.e-gov.go.jp/api/1)
- **Google Gemini API**: 自然言語処理と回答生成用

### 主要コンポーネント

- **法令特定**: ユーザーの質問から関連する法令を自動的に特定
- **法令取得**: e-Gov法令APIからの法令データ取得とパース
- **キャッシュシステム**: 頻繁に参照される法令データをローカルにキャッシュ
- **条文抽出**: 質問に関連する条文の自動抽出
- **回答生成**: Gemini APIによる自然な日本語での回答生成

### データフローパイプライン

1. ユーザー入力の受け取り
2. 関連法令の特定 (Gemini API)
3. 法令データの取得 (e-Gov法令API)
4. 関連条文の抽出 (Gemini API)
5. 回答の生成 (Gemini API)
6. 結果の表示

## e-Gov法令APIについて

このアプリケーションは、e-Gov法令API (Version 1)を使用しています。API仕様は[公式ドキュメント](https://laws.e-gov.go.jp/)を参照してください。

主要エンドポイント:
- 法令名一覧取得API
- 法令取得API
- 条文内容取得API
- 更新法令一覧取得API

## 将来の改善予定

- 検索オプションの拡充
- 複数法令間の関連性分析
- 判例情報との連携
- レスポンスパフォーマンスの最適化
- 大規模法令の効率的な分割処理