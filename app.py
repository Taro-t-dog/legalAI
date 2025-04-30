import os
import requests
import re
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import xml.etree.ElementTree as ET
import sys
from datetime import datetime, timedelta
import json
import time
import hashlib
import zlib
import pickle
from functools import lru_cache
from datetime import datetime, timedelta
import logging
import traceback

# 既存のキャッシュディレクトリ定数を使用
CACHE_DIR = "cache"

# キャッシュ設定の追加（既存の定数定義セクションに）
CACHE_CONFIG = {
    # キャッシュタイプごとの有効期限（日数）
    'expire_days': {
        'lawlists': 30,      # 法令一覧は比較的安定しているので長めに
        'laws': 14,          # 法令本文は改正があるので短めに
        'articles': 14,      # 条文も同様
        'updated_laws': 7,   # 更新法令は最も短く
        'search_results': 1  # 検索結果は1日のみ
    },
    # メモリキャッシュの最大サイズ
    'memory_cache_size': 128,
    # ディスクキャッシュの最大サイズ（MB）
    'disk_cache_max_size': 500,
    # キャッシュ自動クリーンアップの間隔（日数）
    'cleanup_interval': 7
}
class APIError(Exception):
    """API関連のエラーを表すカスタム例外クラス"""
    def __init__(self, message, status_code=None, detail=None):
        self.message = message
        self.status_code = status_code
        self.detail = detail
        super().__init__(self.message)
def handle_legal_search_errors(func):
    """法令検索関連の関数のエラーハンドリングを行うデコレータ"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.ConnectionError:
            error_msg = "e-Gov法令APIサーバーに接続できません。インターネット接続を確認してください。"
            logger.error(error_msg)
            st.error(error_msg)
            return None
        except requests.exceptions.Timeout:
            error_msg = "APIリクエストがタイムアウトしました。サーバーが混雑しているか、接続が不安定です。"
            logger.error(error_msg)
            st.error(error_msg)
            return None
        except requests.exceptions.RequestException as e:
            error_msg = f"APIリクエスト中にエラーが発生しました: {e}"
            logger.error(error_msg)
            st.error(error_msg)
            return None
        except APIError as e:
            error_msg = f"API処理エラー: {e.message}"
            if e.detail:
                error_msg += f" (詳細: {e.detail})"
            logger.error(error_msg)
            st.error(error_msg)
            return None
        except ET.ParseError as e:
            error_msg = f"XML解析エラー: {e}。データ形式に問題があります。"
            logger.error(error_msg)
            st.error(error_msg)
            return None
        except Exception as e:
            error_msg = f"予期しないエラーが発生しました: {e}"
            logger.exception(error_msg)  # スタックトレースを含むログ
            st.error(error_msg + "\n詳細はログを確認してください。")
            return None
    return wrapper

# APIサーバーとの接続確認
def check_api_connection():
    """APIサーバーとの接続を確認する関数 (リトライ機能付き)"""
    max_retries = 3
    retry_delay = 2  # 秒
    
    for attempt in range(max_retries):
        try:
            test_url = f"{EGOV_API_BASE_URL}/lawlists/1"
            response = requests.get(test_url, timeout=15)
            if response.status_code == 200:
                logger.info("e-Gov法令APIへの接続に成功しました")
                return True
            else:
                logger.warning(f"APIサーバーとの接続に問題があります (試行 {attempt+1}/{max_retries}): ステータスコード {response.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数バックオフ
                else:
                    logger.error(f"APIサーバーとの接続に失敗しました: ステータスコード {response.status_code}")
                    return False
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"APIサーバーとの接続中にエラーが発生しました (試行 {attempt+1}/{max_retries}): {e}")
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数バックオフ
            else:
                logger.error(f"APIサーバーとの接続中に最終的にエラーが発生しました: {e}")
                return False
    
    return False
def get_law_content_chunked(law_id=None, law_num=None, max_chunk_size=50):
    """
    大きな法令データを分割して取得する関数
    条文を一度に最大max_chunk_size個ずつ取得する
    
    Returns:
        dict: 構造化された法令データ
    """
    # パラメータチェック
    if not law_id and not law_num:
        raise APIError("法令IDまたは法令番号が指定されていません")
    
    # 法令の基本情報を取得
    basic_info = None
    try:
        # 法令IDまたは法令番号から法令情報を取得
        if law_id:
            url = f"{EGOV_API_BASE_URL}/lawdata/{law_id}"
        else:
            url = f"{EGOV_API_BASE_URL}/lawdata/{law_num}"
        
        # プログレス表示
        with st.status(f"大規模法令データを分割取得中...") as status:
            # 基本情報のリクエスト
            response = requests.get(url, timeout=15)
            if response.status_code != 200:
                status.update(label=f"法令データの取得に失敗しました: {response.status_code}", state="error")
                return None
            
            # 基本情報の解析
            success, xml_data = handle_api_response(response, "法令基本情報の取得に失敗しました")
            if not success:
                status.update(label="法令基本情報の取得に失敗しました", state="error")
                return None
            
            # XMLからLawIdとLawNumを抽出
            try:
                root = ET.fromstring(xml_data)
                law_id_elem = root.find("./ApplData/LawId")
                law_num_elem = root.find("./ApplData/LawNum")
                
                if law_id_elem is not None and law_id_elem.text:
                    law_id = law_id_elem.text
                
                if law_num_elem is not None and law_num_elem.text:
                    law_num = law_num_elem.text
                
                # TOC要素から目次情報を取得（条文一覧）
                law_full_text = root.find("./ApplData/LawFullText")
                if law_full_text is not None and law_full_text.text:
                    try:
                        law_body = ET.fromstring(law_full_text.text)
                        toc = law_body.find(".//TOC")
                        
                        if toc is not None:
                            # 条文一覧を抽出
                            article_list = []
                            for toc_item in toc.findall(".//TOCArticle"):
                                article_num = toc_item.find("./ArticleTitle")
                                if article_num is not None and article_num.text:
                                    article_list.append(f"第{article_num.text}条")
                            
                            status.update(label=f"目次から{len(article_list)}個の条文を抽出しました", state="running")
                            
                            # 法令基本情報を作成
                            basic_info = {
                                'law_id': law_id,
                                'law_num': law_num,
                                'article_list': article_list
                            }
                    except Exception as e:
                        logger.error(f"目次解析中にエラーが発生しました: {e}")
            except Exception as e:
                logger.error(f"法令基本情報の解析中にエラーが発生しました: {e}")
    
    except Exception as e:
        logger.error(f"法令基本情報の取得中にエラーが発生しました: {e}")
        return None
    
    # 条文リストがある場合は分割取得
    if basic_info and 'article_list' in basic_info and basic_info['article_list']:
        article_list = basic_info['article_list']
        
        # 結果格納用
        all_articles = []
        total_articles = len(article_list)
        
        with st.progress(0) as progress_bar:
            # チャンク処理
            for i in range(0, total_articles, max_chunk_size):
                chunk = article_list[i:i+max_chunk_size]
                progress_percent = i / total_articles
                progress_bar.progress(progress_percent)
                
                # 各条文を個別に取得
                for j, article_num in enumerate(chunk):
                    try:
                        article_content = get_article_content(
                            law_id=law_id, 
                            law_num=law_num, 
                            article=article_num
                        )
                        
                        if article_content:
                            parsed_article = parse_article_content_xml(article_content)
                            if parsed_article:
                                all_articles.append({
                                    'article_number': article_num,
                                    'content': parsed_article
                                })
                        
                        # 進捗更新
                        current_progress = (i + j + 1) / total_articles
                        progress_bar.progress(min(current_progress, 1.0))
                        
                    except Exception as e:
                        logger.error(f"条文 {article_num} の取得中にエラーが発生しました: {e}")
                
                # APIリクエストの間隔を空ける（サーバー負荷対策）
                time.sleep(0.5)
            
            # 完了
            progress_bar.progress(1.0)
        
        # 最終結果の構築
        result = {
            'law_id': law_id,
            'law_num': law_num,
            'title': '分割取得による法令データ',
            'structured_articles': all_articles
        }
        
        # キャッシュに保存
        cache_key = f"chunked_{law_id or law_num}"
        save_to_cache("laws", cache_key, result)
        
        return result
    
    # 条文リストが取得できなかった場合は通常の取得を試みる
    return get_law_content(law_id=law_id, law_num=law_num)

# メモリキャッシュの活用（頻繁に使用される法令用）
@lru_cache(maxsize=CACHE_CONFIG['memory_cache_size'])
def get_cached_law_metadata(law_id):
    """
    法令の基本メタデータをメモリキャッシュから取得する
    ※軽量なデータのみを対象とする
    """
    # キャッシュキー
    cache_key = f"meta_{law_id}"
    
    # ディスクキャッシュをチェック
    cached_data = load_from_cache("laws", cache_key)
    if cached_data:
        return cached_data
    
    # APIから取得
    try:
        url = f"{EGOV_API_BASE_URL}/lawdata/{law_id}"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None
        
        # 応答を解析
        root = ET.fromstring(response.text)
        
        # 基本メタデータを抽出
        metadata = {}
        
        law_id_elem = root.find("./ApplData/LawId")
        if law_id_elem is not None and law_id_elem.text:
            metadata['law_id'] = law_id_elem.text
        
        law_num_elem = root.find("./ApplData/LawNum")
        if law_num_elem is not None and law_num_elem.text:
            metadata['law_num'] = law_num_elem.text
        
        # 法令本文から法令名を抽出
        law_full_text = root.find("./ApplData/LawFullText")
        if law_full_text is not None and law_full_text.text:
            try:
                law_body = ET.fromstring(law_full_text.text)
                law_title = law_body.find(".//LawTitle")
                if law_title is not None and law_title.text:
                    metadata['law_title'] = law_title.text
            except:
                pass
        
        # キャッシュに保存（より長い有効期限）
        save_to_cache("laws", cache_key, metadata)
        
        return metadata
    
    except Exception as e:
        logger.error(f"法令メタデータ取得中にエラーが発生しました: {e}")
        return None
# ロギング設定
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("legal_ai.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('legal_ai_assistant')

# 環境変数のロード
load_dotenv()

# APIキーの設定
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# e-Gov 法令APIのベースURL（2024年7月29日の仕様変更に対応）
EGOV_API_BASE_URL = "https://laws.e-gov.go.jp/api/1"
# APIサーバーとの接続確認を行う
# APIサーバーとの接続確認を行うボタンを追加
st.sidebar.subheader("API接続ステータス")
if st.sidebar.button("API接続を確認"):
    with st.sidebar:
        with st.spinner("接続確認中..."):
            if check_api_connection():
                st.success("✅ e-Gov法令APIに接続できています")
            else:
                st.error("❌ e-Gov法令APIに接続できません")
# キャッシュディレクトリ
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# キャッシュ関連の関数
def setup_cache_directories():
    """キャッシュディレクトリの構造を整備する"""
    # メインキャッシュディレクトリ
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # キャッシュタイプごとのサブディレクトリ
    for cache_type in CACHE_CONFIG['expire_days'].keys():
        os.makedirs(os.path.join(CACHE_DIR, cache_type), exist_ok=True)
    
    # キャッシュ情報ファイル（最終クリーンアップ日など）
    cache_info_file = os.path.join(CACHE_DIR, 'cache_info.json')
    if not os.path.exists(cache_info_file):
        with open(cache_info_file, 'w', encoding='utf-8') as f:
            json.dump({
                'last_cleanup': datetime.now().isoformat(),
                'version': '1.0',
                'stats': {t: {'count': 0, 'size_bytes': 0} for t in CACHE_CONFIG['expire_days'].keys()}
            }, f)
    
    logger.info("キャッシュディレクトリを初期化しました")
    
    return True
def get_cache_path(cache_type, key):
    """キャッシュファイルのパスを生成する（最適化版）"""
    # キーをハッシュ化して短くする（特に長い法令名や条文指定に対応）
    hash_key = hashlib.md5(key.encode('utf-8')).hexdigest()
    return os.path.join(CACHE_DIR, cache_type, f"{hash_key}.json")
def save_to_cache(cache_type, key, data):
    """データをキャッシュに保存する（最適化版）"""
    # キャッシュタイプの有効性チェック
    if cache_type not in CACHE_CONFIG['expire_days']:
        logger.warning(f"不明なキャッシュタイプ: {cache_type}")
        cache_type = 'search_results'  # デフォルトの短い有効期限を使用
    
    expire_days = CACHE_CONFIG['expire_days'][cache_type]
    cache_path = get_cache_path(cache_type, key)
    
    try:
        cache_data = {
            "key": key,  # 元のキーも保存（デバッグ用）
            "data": data,
            "created": datetime.now().isoformat(),
            "expires": (datetime.now() + timedelta(days=expire_days)).isoformat(),
            "cache_type": cache_type
        }
        
        # データの圧縮（大きな法令データ用）
        if sys.getsizeof(str(data)) > 100000:  # 100KB以上の場合
            try:
                import zlib
                import pickle
                
                # データをシリアライズして圧縮
                serialized_data = pickle.dumps(data)
                compressed_data = zlib.compress(serialized_data)
                
                # 圧縮データを保存
                with open(cache_path.replace('.json', '.zlib'), 'wb') as f:
                    f.write(compressed_data)
                
                # JSONファイルには圧縮されたことを記録
                cache_data["data"] = None
                cache_data["compressed"] = True
                cache_data["original_size"] = sys.getsizeof(str(data))
                cache_data["compressed_size"] = len(compressed_data)
                
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, ensure_ascii=False)
                
                logger.info(f"大きなデータを圧縮してキャッシュに保存しました: {cache_path} (圧縮率: {len(compressed_data)/sys.getsizeof(str(data)):.2f})")
                return True
                
            except (ImportError, Exception) as e:
                logger.warning(f"データ圧縮に失敗しました: {e}、通常の方法で保存します")
        
        # 通常のJSONとして保存（小さなデータまたは圧縮失敗時）
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False)
        
        logger.info(f"キャッシュを保存しました: {cache_path}")
        
        # キャッシュ統計の更新
        update_cache_stats(cache_type, 1, os.path.getsize(cache_path))
        
        return True
    except Exception as e:
        logger.error(f"キャッシュの保存中にエラーが発生しました: {e}")
        return False
def load_from_cache(cache_type, key):
    """キャッシュからデータを読み込む（最適化版）"""
    cache_path = get_cache_path(cache_type, key)
    compressed_path = cache_path.replace('.json', '.zlib')
    
    # 圧縮ファイルの存在確認
    if os.path.exists(compressed_path):
        try:
            import zlib
            import pickle
            
            # 圧縮ファイルからメタデータを読み込む
            if os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_meta = json.load(f)
                
                # 有効期限のチェック
                if "expires" in cache_meta:
                    expire_date = datetime.fromisoformat(cache_meta["expires"])
                    if datetime.now() > expire_date:
                        logger.info(f"キャッシュの有効期限が切れています: {cache_path}")
                        return None
                
            # 圧縮ファイルを読み込んで解凍
            with open(compressed_path, 'rb') as f:
                compressed_data = f.read()
                
            serialized_data = zlib.decompress(compressed_data)
            data = pickle.loads(serialized_data)
            
            logger.info(f"圧縮されたキャッシュからデータを読み込みました: {compressed_path}")
            return data
            
        except Exception as e:
            logger.error(f"圧縮キャッシュの読み込み中にエラーが発生しました: {e}")
            # 通常のJSONファイルを試す
    
    # 通常のJSONファイルから読み込み
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        # 有効期限のチェック
        if "expires" in cache_data:
            expire_date = datetime.fromisoformat(cache_data["expires"])
            if datetime.now() > expire_date:
                logger.info(f"キャッシュの有効期限が切れています: {cache_path}")
                return None
        
        logger.info(f"キャッシュからデータを読み込みました: {cache_path}")
        return cache_data["data"]
    except Exception as e:
        logger.error(f"キャッシュの読み込み中にエラーが発生しました: {e}")
        return None

def update_cache_stats(cache_type, count_delta, size_delta):
    """キャッシュ統計情報を更新する"""
    cache_info_file = os.path.join(CACHE_DIR, 'cache_info.json')
    try:
        if os.path.exists(cache_info_file):
            with open(cache_info_file, 'r', encoding='utf-8') as f:
                cache_info = json.load(f)
            
            if 'stats' not in cache_info:
                cache_info['stats'] = {}
            
            if cache_type not in cache_info['stats']:
                cache_info['stats'][cache_type] = {'count': 0, 'size_bytes': 0}
            
            cache_info['stats'][cache_type]['count'] += count_delta
            cache_info['stats'][cache_type]['size_bytes'] += size_delta
            
            with open(cache_info_file, 'w', encoding='utf-8') as f:
                json.dump(cache_info, f, ensure_ascii=False)
    except Exception as e:
        logger.error(f"キャッシュ統計の更新中にエラーが発生しました: {e}")

def clean_expired_cache():
    """期限切れのキャッシュを削除する"""
    cache_info_file = os.path.join(CACHE_DIR, 'cache_info.json')
    cleaned_count = 0
    cleaned_size = 0
    
    try:
        # キャッシュ情報ファイルを読み込む
        if os.path.exists(cache_info_file):
            with open(cache_info_file, 'r', encoding='utf-8') as f:
                cache_info = json.load(f)
            
            # 最終クリーンアップ日をチェック
            last_cleanup = datetime.fromisoformat(cache_info.get('last_cleanup', '2000-01-01T00:00:00'))
            cleanup_interval = CACHE_CONFIG['cleanup_interval']
            
            # クリーンアップの必要がなければ終了
            if (datetime.now() - last_cleanup).days < cleanup_interval:
                logger.info(f"キャッシュクリーンアップは不要です。最終クリーンアップ: {last_cleanup.isoformat()}")
                return 0, 0
        
        # 各キャッシュタイプについて処理
        for cache_type, expire_days in CACHE_CONFIG['expire_days'].items():
            cache_dir = os.path.join(CACHE_DIR, cache_type)
            if not os.path.exists(cache_dir):
                continue
            
            # ディレクトリ内のファイルを処理
            for filename in os.listdir(cache_dir):
                file_path = os.path.join(cache_dir, filename)
                
                # JSONファイルの場合
                if filename.endswith('.json'):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            cache_data = json.load(f)
                        
                        # 有効期限のチェック
                        if "expires" in cache_data:
                            expire_date = datetime.fromisoformat(cache_data["expires"])
                            if datetime.now() > expire_date:
                                # 通常のファイルを削除
                                file_size = os.path.getsize(file_path)
                                os.remove(file_path)
                                cleaned_count += 1
                                cleaned_size += file_size
                                
                                # 対応する圧縮ファイルも削除
                                zlib_path = file_path.replace('.json', '.zlib')
                                if os.path.exists(zlib_path):
                                    zlib_size = os.path.getsize(zlib_path)
                                    os.remove(zlib_path)
                                    cleaned_size += zlib_size
                    except Exception as e:
                        logger.error(f"キャッシュファイル {filename} の処理中にエラーが発生しました: {e}")
                
                # 圧縮ファイルの場合（対応するJSONがない場合）
                elif filename.endswith('.zlib'):
                    json_path = file_path.replace('.zlib', '.json')
                    if not os.path.exists(json_path):
                        # 孤立した圧縮ファイルを削除
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)
                        cleaned_count += 1
                        cleaned_size += file_size
        
        # キャッシュ情報の更新
        cache_info['last_cleanup'] = datetime.now().isoformat()
        with open(cache_info_file, 'w', encoding='utf-8') as f:
            json.dump(cache_info, f, ensure_ascii=False)
        
        logger.info(f"キャッシュクリーンアップ完了: {cleaned_count}ファイル、{cleaned_size/1024/1024:.2f}MB削除")
        return cleaned_count, cleaned_size
        
    except Exception as e:
        logger.error(f"キャッシュクリーンアップ中にエラーが発生しました: {e}")
        return 0, 0

# タイトルとアプリの説明
st.title("法令検索AIアシスタント")
st.write("法的な質問を入力すると、関連する法令を検索して回答します。")

def handle_api_response(response, error_message="APIエラーが発生しました"):
    """APIレスポンスを処理する共通関数 (XMLエラー対策強化版)"""
    if response.status_code == 200:
        try:
            # レスポンスの詳細ログ記録
            response_text = response.text
            logger.info(f"API応答 (最初の500文字): {response_text[:500]}...")
            
            # デバッグログ用に一時ファイルに保存
            debug_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_file = f"debug_xml_{debug_timestamp}.xml"
            with open(os.path.join(CACHE_DIR, debug_file), 'w', encoding='utf-8') as f:
                f.write(response_text)
            logger.info(f"デバッグ用にXMLを保存しました: {debug_file}")
            
            # 前処理：XML宣言が複数ある場合に最初の1つだけ残す
            if response_text.count('<?xml') > 1:
                logger.warning("複数のXML宣言が検出されました。最初の宣言のみを保持します。")
                xml_parts = response_text.split('<?xml', 1)
                response_text = '<?xml' + xml_parts[1]
            
            # 空白文字や制御文字の削除
            response_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', response_text)
            
            # XML解析
            try:
                root = ET.fromstring(response_text)
            except ET.ParseError as xml_err:
                # lxmlを試す（より寛容なXMLパーサー）
                logger.warning(f"標準XMLパーサーでエラー: {xml_err}。lxmlを試みます。")
                from lxml import etree
                root = etree.fromstring(response_text.encode('utf-8'))
                # lxmlから標準ElementTreeに変換
                root = ET.fromstring(etree.tostring(root, encoding='unicode'))
            
            result_code = root.find("./Result/Code").text
            result_message = root.find("./Result/Message").text

            if result_code == "0":  # 正常
                return True, response_text
            elif result_code == "1":  # エラー
                error_msg = f"API実行エラー: {result_message}"
                logger.error(error_msg)
                st.error(error_msg)
                return False, None
            elif result_code == "2":  # 複数候補あり（別表取得時）
                return True, response_text  # 複数候補を後で処理
        except ET.ParseError as e:
            # XMLではない可能性がある場合は、HTMLレスポンスをチェック
            if "Request Rejected" in response.text:
                error_msg = "「Request Rejected」エラーが発生しました。リクエストパラメータが長すぎる可能性があります。"
                logger.error(f"{error_msg} レスポンス: {response.text[:200]}...")
                st.error(error_msg)
            else:
                error_msg = f"XMLの解析中にエラーが発生しました: {e}"
                logger.error(f"{error_msg} レスポンス: {response.text[:200]}...")
                
                # XML解析エラーの詳細情報を記録
                debug_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                error_file = f"error_xml_{debug_timestamp}.xml"
                with open(os.path.join(CACHE_DIR, error_file), 'w', encoding='utf-8') as f:
                    f.write(response.text)
                logger.error(f"エラーのあるXMLを保存しました: {error_file}")
                
                # XMLではなくHTMLが返された可能性をチェック
                if '<html' in response.text.lower() or '<!doctype html' in response.text.lower():
                    logger.error("XMLではなく、HTMLが返されました。APIの仕様変更の可能性があります。")
                    st.error("APIがHTMLを返しました。e-Gov法令APIの仕様が変更された可能性があります。")
                else:
                    st.error(error_msg)
            return False, None
        except Exception as e:
            error_msg = f"XMLの解析中にエラーが発生しました: {e}"
            logger.error(f"{error_msg} レスポンス: {response.text[:200]}...")
            st.error(error_msg)
            return False, None
    elif response.status_code == 300:  # Multiple Choices
        logger.info("複数の候補があります。より具体的な条件で検索してください。")
        st.info("複数の候補があります。より具体的な条件で検索してください。")
        return True, response.text  # 複数候補として処理
    elif response.status_code == 400:
        error_msg = "不正なリクエストです。パラメータを確認してください。"
        logger.error(f"{error_msg} レスポンス: {response.text[:200]}...")
        st.error(error_msg)
    elif response.status_code == 404:
        # 該当データがない場合はエラーではなく、情報として返す
        logger.info("該当するデータが見つかりませんでした。")
        return False, "NOT_FOUND"
    elif response.status_code == 406:
        error_msg = "法令APIで返却可能な容量を超えているか、複数の法令データが存在します。"
        logger.error(f"{error_msg} レスポンス: {response.text[:200]}...")
        st.error(error_msg)
    elif response.status_code == 500:
        error_msg = "サーバー内部でエラーが発生しました。しばらく経ってからお試しください。"
        logger.error(f"{error_msg} レスポンス: {response.text[:200]}...")
        st.error(error_msg)
    else:
        error_msg = f"{error_message}: ステータスコード {response.status_code}"
        logger.error(f"{error_msg} レスポンス: {response.text[:200]}...")
        st.error(error_msg)
    
    return False, None

try:
    from lxml import etree as lxml_etree
    LXML_AVAILABLE = True
    logger.info("lxmlライブラリが見つかりました。XML解析に使用します。")
except ImportError:
    LXML_AVAILABLE = False
    logger.warning("lxmlライブラリが見つかりません。標準のxml.etreeを使用します。大きな法令の解析に失敗する可能性があります。")
    logger.warning("より安定した動作のために 'pip install lxml' の実行を推奨します。")


# 閉じられていないタグを修正する関数
def fix_unclosed_tags(xml_text):
    """XMLの閉じられていないタグを検出して修正する"""
    # よく使われるタグのリスト
    common_tags = ['Law', 'LawBody', 'LawTitle', 'Article', 'Paragraph', 'Item']
    
    # 各タグについて開始タグと終了タグの数を比較
    for tag in common_tags:
        open_count = xml_text.count(f'<{tag}')
        close_count = xml_text.count(f'</{tag}')
        
        # 開始タグの方が多い場合は不足している終了タグを追加
        if open_count > close_count:
            logger.warning(f"タグ<{tag}>が{open_count - close_count}個閉じられていません")
            # 文末に終了タグを追加
            xml_text += f'</{tag}>' * (open_count - close_count)
    
    return xml_text
     


def get_law_list(law_type=1):
    """
    法令名一覧を取得する関数（エラーハンドリング強化版）
    law_type: 法令種別 (1: 全法令, 2: 憲法・法律, 3: 政令・勅令, 4: 府省令・規則)
    """
    url = f"{EGOV_API_BASE_URL}/lawlists/{law_type}"
    
    # 接続確認
    if not check_api_connection():
        raise APIError("APIサーバーとの接続に失敗しました。しばらくしてからお試しください。")
    
    # キャッシュチェック
    cache_key = f"lawlist_{law_type}"
    cached_data = load_from_cache("lawlists", cache_key)
    if cached_data:
        logger.info(f"法令一覧をキャッシュから取得しました (種別: {law_type})")
        return cached_data
    
    # APIリクエスト
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            status_message = {
                400: "パラメータが不正です",
                404: "法令一覧が見つかりません",
                500: "APIサーバー内部でエラーが発生しています"
            }.get(response.status_code, f"予期しないステータスコード: {response.status_code}")
            
            raise APIError(
                f"法令一覧の取得に失敗しました: {status_message}", 
                status_code=response.status_code,
                detail=response.text[:200] if response.text else None
            )
        
        # レスポンス処理
        success, xml_data = handle_api_response(response, "法令一覧の取得に失敗しました")
        if not success:
            if xml_data == "NOT_FOUND":
                logger.warning(f"法令一覧が見つかりませんでした (種別: {law_type})")
                return None
            raise APIError("法令一覧データの解析に失敗しました")
        
        # キャッシュに保存
        save_to_cache("lawlists", cache_key, xml_data)
        logger.info(f"法令一覧をAPIから取得してキャッシュに保存しました (種別: {law_type})")
        return xml_data
        
    except requests.exceptions.Timeout:
        raise APIError("APIリクエストがタイムアウトしました", detail="サーバーの応答が遅いか、接続が不安定です")
    except requests.exceptions.RequestException as e:
        raise APIError(f"APIリクエスト中にエラーが発生しました: {e}")

def identify_relevant_laws(query):
    """
    ユーザーの質問から関連する可能性のある日本の法令名を特定する関数
    Gemini 2.0 Flash モデルを使用
    """
    prompt = f"""
    あなたは日本の法律に詳しい弁護士です。以下の質問を読んで、最も関連性が高いと思われる法令を特定してください。

    質問: {query}

    指示:
    1. この質問に最も関連する法令名を3つまで特定してください
    2. 「民法」「刑法」「個人情報の保護に関する法律」など、正式な法令名を使用してください
    3. 一般的な法令名（民法、刑法など）から、より具体的な特別法（借地借家法、児童虐待防止法など）を優先してください
    4. 回答は法令名のみをリスト形式で出力してください

    回答例:
    - 民法
    - 不正競争防止法
    - 著作権法
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        
        # 改行で分割して各行をリストとして返す
        laws = [law.strip().replace('- ', '') for law in response.text.strip().split('\n') if law.strip()]
        
        # リスト形式でなかった場合の処理（例：「以下の法令が関連します：」などの文言が含まれる場合）
        if len(laws) == 1 and ("、" in laws[0] or "," in laws[0]):
            if "、" in laws[0]:
                laws = [law.strip() for law in laws[0].split("、")]
            else:
                laws = [law.strip() for law in laws[0].split(",")]
        
        # 法令名の正規化
        normalized_laws = []
        for law in laws:
            # 余分な文字を削除（「1. 」や「・」など）
            law = law.lstrip("1234567890. -・*").strip()
            # 「○○法」のような形式かチェック
            if law and not law.endswith(("法", "条例", "規則", "令")):
                # 名称が不完全な場合、補完を試みる
                if "民法" in law.lower():
                    law = "民法"
                elif "刑法" in law.lower():
                    law = "刑法"
                elif "商法" in law.lower():
                    law = "商法"
                # その他の法令も必要に応じて追加
            
            if law:  # 空でない場合のみ追加
                normalized_laws.append(law)
        
        return normalized_laws[:3]  # 最大3つまで
    except Exception as e:
        st.error(f"Gemini APIの呼び出し中にエラーが発生しました: {e}")
        return []

def parse_law_content_xml(xml_data):
    """
    法令内容のXMLを解析して本文を抽出し、構造化する（最終手段的前処理追加版）
    """
    if xml_data == "NOT_FOUND":
        logger.warning("APIから 'NOT_FOUND' を受け取りました。法令が見つかりません。")
        return None
    if isinstance(xml_data, dict):
        # ★★★ ここで警告を出すように変更 ★★★
        logger.warning("parse_law_content_xml に辞書が渡されました。本来XML文字列が期待されます。")
        logger.debug(f"渡された辞書データ (一部): {str(xml_data)[:200]}...")
        # 辞書が渡された場合は、それが解析済みデータとみなしてそのまま返す
        return xml_data
    if not isinstance(xml_data, str) or not xml_data.strip():
        logger.error("解析対象のXMLデータが無効(空文字列または文字列以外)です。")
        return None

    logger.debug(f"解析対象XMLデータ (最初の500文字): {xml_data[:500]}...")

    law_xml_content_original = None # 法令本文のXML文字列を格納する変数 (加工前)
    appl_data = None               # ApplData要素を格納する変数

    try:
        # --- 外側XMLの解析 ---
        outer_root = None
        if LXML_AVAILABLE:
            try:
                outer_parser = lxml_etree.XMLParser(recover=False, encoding='utf-8')
                outer_root_lxml = lxml_etree.fromstring(xml_data.encode('utf-8'), parser=outer_parser)
                appl_data_lxml = outer_root_lxml.xpath("./ApplData")
                if appl_data_lxml:
                    appl_data = appl_data_lxml[0]
                    law_text_elem_lxml = appl_data.xpath("./LawFullText")
                    if law_text_elem_lxml and law_text_elem_lxml[0].text:
                        law_xml_content_original = law_text_elem_lxml[0].text # 元のテキストを保持
                        logger.info("lxmlで外側XMLを解析し、LawFullTextを取得しました。")
                    else:
                        logger.warning("lxmlでLawFullTextが見つからないか、内容が空です。")
                else:
                    logger.error("lxmlで外側XMLからApplDataタグが見つかりませんでした。")
            except lxml_etree.XMLSyntaxError as outer_lxml_err:
                logger.warning(f"lxmlでの外側XML解析に失敗: {outer_lxml_err}。標準のETを試します。")
                outer_root = None
            except Exception as e:
                logger.error(f"lxmlでの外側XML解析中に予期せぬエラー: {e}")
                outer_root = None

        if law_xml_content_original is None: # まだLawFullTextが取れていない場合
             try:
                 outer_root = ET.fromstring(xml_data)
                 appl_data_et = outer_root.find("./ApplData")
                 if appl_data_et is not None:
                      appl_data = appl_data_et
                      law_text_elem_et = appl_data.find("./LawFullText")
                      if law_text_elem_et is not None and law_text_elem_et.text:
                           law_xml_content_original = law_text_elem_et.text # 元のテキストを保持
                           logger.info("標準ETで外側XMLを解析し、LawFullTextを取得しました。")
                      else:
                           logger.warning("標準ETでLawFullTextが見つからないか、内容が空です。")
                 else:
                      logger.error("標準ETで外側XMLからApplDataタグが見つかりませんでした。")
             except ET.ParseError as outer_xml_err:
                  logger.error(f"標準ETでの外側XMLデータの解析にも失敗しました: {outer_xml_err}")
                  return None
             except Exception as e:
                  logger.error(f"標準ETでの外側XML解析中に予期せぬエラー: {e}")
                  return None

        # --- 内側XML (LawFullTextの中身) の解析 ---
        if law_xml_content_original: # 元のテキストが取得できているか確認
            # 前処理 (strip, BOM除去)
            logger.debug(f"取得直後の LawFullText (最初の50文字、repr): {repr(law_xml_content_original[:50])}")
            law_xml_content_processed = law_xml_content_original.strip()
            if law_xml_content_processed.startswith('\ufeff'):
                law_xml_content_processed = law_xml_content_processed[1:]
                logger.info("LawFullText の先頭からBOMを除去しました。")

            # ★★★ 更なる前処理: 最初の '<' までを削除する試み ★★★
            first_lt_index = law_xml_content_processed.find('<')
            if first_lt_index > 0:
                logger.warning(f"XMLの開始タグ '<' の前に {first_lt_index} 文字が存在します。除去を試みます。")
                logger.debug(f"除去対象文字列 (repr): {repr(law_xml_content_processed[:first_lt_index])}")
                law_xml_content_processed = law_xml_content_processed[first_lt_index:]
            elif first_lt_index < 0:
                logger.error("LawFullText内に開始タグ '<' が見つかりません。解析不能です。")
                # 元のテキストでテキスト抽出を試みる
                if isinstance(law_xml_content_original, str) and law_xml_content_original.strip():
                     # ...(テキスト抽出処理)...
                     logger.warning("XML解析不能のため、テキスト抽出にフォールバックします。")
                     try:
                         text_based_articles = extract_articles_from_text(law_xml_content_original) # 元データから抽出
                         return {'full_text': law_xml_content_original,'structured_articles': text_based_articles}
                     except Exception as text_err:
                         logger.error(f"テキスト抽出も失敗: {text_err}")
                         return {'full_text': law_xml_content_original,'structured_articles': [] }
                else:
                     return {'full_text': "法令本文データの内容が無効でした。",'structured_articles': [] }

            logger.debug(f"最終前処理後の LawFullText (最初の100文字、repr): {repr(law_xml_content_processed[:100])}")
            logger.debug(f"最終前処理後の LawFullText (最後の50文字、repr): {repr(law_xml_content_processed[-50:])}")

            if not law_xml_content_processed:
                logger.error("最終前処理後、LawFullTextの内容が空になりました。")
                return { 'full_text': "法令本文データの内容が無効でした。", 'structured_articles': [] }

            # ここから解析処理
            inner_xml_to_parse = law_xml_content_processed # 解析対象は最終前処理後のもの
            logger.info("LawFullTextの内容 (内側XML) を解析します...")
            law_element = None

            # 1. lxml で試す
            if LXML_AVAILABLE:
                try:
                    logger.info("lxmlパーサー (recover=True) で内側XMLの解析を試みます...")
                    inner_parser = lxml_etree.XMLParser(recover=True, encoding='utf-8')
                    law_xml_bytes = inner_xml_to_parse.encode('utf-8')
                    logger.debug(f"lxml解析対象バイト列 (最初の100バイト、repr): {repr(law_xml_bytes[:100])}")
                    law_element_lxml = lxml_etree.fromstring(law_xml_bytes, parser=inner_parser)

                    if law_element_lxml is not None:
                        # recover=Trueで何か構造が復元されたかチェック (簡単なチェック)
                        if len(lxml_etree.tostring(law_element_lxml)) < 10: # あまりに短い場合は失敗とみなす
                            logger.warning("lxml(recover=True)解析結果が非常に短いため、失敗とみなします。")
                            law_element = None
                        else:
                            logger.info("lxml解析成功。ET形式への変換を試みます...")
                            try:
                                law_element_str = lxml_etree.tostring(law_element_lxml, encoding='unicode')
                                logger.debug(f"lxml -> ET変換前文字列 (最初の100文字、repr): {repr(law_element_str[:100])}")
                                law_element = ET.fromstring(law_element_str)
                                logger.info("lxml -> ET変換成功。")
                            except ET.ParseError as conversion_err:
                                logger.warning(f"lxmlで解析後、ETへの変換に失敗: {conversion_err}。標準ETを試します。")
                                law_element = None
                    else:
                         logger.warning("lxml (recover=True) でも内側XMLの解析結果がNoneでした。標準ETを試します。")

                except lxml_etree.XMLSyntaxError as inner_lxml_err:
                    # line 2, column 6 エラーがここで発生している可能性が高い
                    logger.warning(f"lxmlによる内側XMLの解析でXMLSyntaxError: {inner_lxml_err}。標準ETを試します。")
                except Exception as e:
                    logger.error(f"lxmlでの内側XML解析中に予期せぬエラー: {e}")
                    logger.error(traceback.format_exc())

            # 2. lxmlで失敗 or lxmlがない場合は標準ETで試す
            if law_element is None:
                try:
                    logger.info("標準ETパーサーで内側XMLの解析を試みます...")
                    cleaned_law_xml = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', inner_xml_to_parse)
                    logger.debug(f"ET解析対象文字列 (最初の100文字、repr): {repr(cleaned_law_xml[:100])}")
                    law_element = ET.fromstring(cleaned_law_xml)
                    logger.info("標準ETによる内側XMLの解析に成功しました。")
                except ET.ParseError as inner_xml_err:
                    logger.warning(f"標準ETでも内側XMLの解析に失敗 (ParseError): {inner_xml_err}。")
                    # (エラー詳細ログ部分は省略 - 前回のコードと同じ)
                    law_element = None # テキスト抽出フォールバックへ
                except Exception as general_err:
                    logger.error(f"標準ETでの内側XML解析中に予期せぬエラー: {general_err}")
                    logger.error(traceback.format_exc())
                    law_element = None # テキスト抽出フォールバックへ

            # --- 構造抽出またはテキスト抽出フォールバック ---
            if law_element is not None:
                logger.info("XML解析結果から条文構造を抽出します。")
                try:
                    structured_data = extract_structured_law(law_element)
                    if structured_data and isinstance(structured_data.get('structured_articles'), list):
                         if structured_data['structured_articles']:
                              logger.info(f"条文構造の抽出に成功。{len(structured_data['structured_articles'])} 個の条文が見つかりました。")
                              structured_data['full_text'] = law_xml_content_original # 元のテキストを返す
                              return structured_data
                         else:
                              logger.warning("extract_structured_law で条文が見つかりませんでした。テキスト抽出にフォールバックします。")
                    else:
                         logger.warning("extract_structured_law が無効なデータを返しました。テキスト抽出にフォールバックします。")
                except Exception as extract_err:
                    logger.error(f"extract_structured_law実行中にエラー: {extract_err}")
                    logger.error(traceback.format_exc())
                law_element = None # 構造抽出失敗時はテキスト抽出へ

            # XML解析 or 構造抽出が失敗した場合のテキスト抽出フォールバック
            if law_element is None:
                 # テキスト抽出の対象は、元の未加工テキスト law_xml_content_original を使う
                if isinstance(law_xml_content_original, str) and law_xml_content_original.strip():
                    logger.warning("XML解析または構造抽出が失敗したため、テキストからの条文抽出を試みます。")
                    try:
                        text_based_articles = extract_articles_from_text(law_xml_content_original)
                        if text_based_articles:
                             logger.info(f"テキスト抽出により {len(text_based_articles)} 個の条文らしきものを抽出しました。")
                        else:
                             logger.warning("テキスト抽出でも条文を見つけられませんでした。")
                        return {
                            'full_text': law_xml_content_original,
                            'structured_articles': text_based_articles
                        }
                    except Exception as text_extract_err:
                        logger.error(f"テキストからの条文抽出中にエラー: {text_extract_err}")
                        logger.error(traceback.format_exc())
                        return {
                            'full_text': law_xml_content_original,
                            'structured_articles': []
                        }
                else:
                    logger.error("LawFullTextの内容が空または無効だったため、テキスト抽出もスキップします。")
                    return {
                        'full_text': "法令本文データの内容が無効でした。",
                        'structured_articles': []
                    }

        else:
            # --- LawFullText自体が空だった場合の処理 ---
            image_data = None
            if appl_data is not None:
                 if LXML_AVAILABLE and hasattr(appl_data, 'xpath'):
                      image_data_list = appl_data.xpath("./ImageData")
                      image_data = image_data_list[0] if image_data_list else None
                 elif hasattr(appl_data, 'find'):
                      image_data = appl_data.find("./ImageData")

            if image_data is not None:
                logger.info("LawFullTextは空ですが、ImageDataが存在します。")
                return {
                    'full_text': "法令の本文には画像が含まれています。詳細は元のウェブサイトでご確認ください。",
                    'structured_articles': []
                }
            else:
                logger.error("LawFullTextタグが見つからないか、内容が空で、ImageDataも見つかりませんでした。")
                return {
                    'full_text': "法令の本文を抽出できませんでした。",
                    'structured_articles': []
                }

    except Exception as e:
        # --- 予期せぬエラー処理 ---
        logger.error(f"法令内容の解析中に予期せぬエラーが発生しました: {e}")
        logger.error(traceback.format_exc())
        if isinstance(law_xml_content_original, str) and law_xml_content_original.strip():
             try:
                 logger.warning("予期せぬエラー発生のため、最後の手段としてテキスト抽出を試みます。")
                 return {
                     'full_text': law_xml_content_original,
                     'structured_articles': extract_articles_from_text(law_xml_content_original)
                 }
             except: pass
        return None
       
    
def extract_structured_law(law_element):
    """
    法令XMLの階層構造から条文情報を抽出する
    """
    result = {
        'full_text': ET.tostring(law_element, encoding='unicode'),
        'structured_articles': []
    }
    
    try:
        # 法令番号を取得
        law_num = law_element.find(".//LawNum")
        if law_num is not None and law_num.text:
            result['law_number'] = law_num.text
        
        # 法令名を取得
        law_title = law_element.find(".//LawTitle")
        if law_title is not None and law_title.text:
            result['law_title'] = law_title.text
        
        # 条文を抽出
        articles = law_element.findall(".//Article")
        
        for article in articles:
            article_data = {'paragraphs': []}
            
            # 条番号を取得
            article_title = article.find("./ArticleTitle")
            if article_title is not None and article_title.text:
                article_data['article_number'] = f"第{article_title.text}条"
            else:
                article_data['article_number'] = "条文"
            
            # 条見出しを取得
            article_caption = article.find("./ArticleCaption")
            if article_caption is not None and article_caption.text:
                article_data['caption'] = article_caption.text
            
            # 条文内容を取得
            article_content = []
            if article_caption is not None and article_caption.text:
                article_content.append(article_caption.text)
            
            # 項を抽出
            paragraphs = article.findall("./Paragraph")
            for paragraph in paragraphs:
                paragraph_data = {}
                
                # 項番号を取得
                para_num = paragraph.find("./ParagraphNum")
                if para_num is not None and para_num.text:
                    paragraph_data['paragraph_number'] = para_num.text
                else:
                    paragraph_data['paragraph_number'] = ""
                
                # 項内容を取得
                para_content = []
                para_caption = paragraph.find("./ParagraphCaption")
                if para_caption is not None and para_caption.text:
                    para_content.append(para_caption.text)
                
                # 号を抽出
                items = paragraph.findall("./Item")
                for item in items:
                    item_title = item.find("./ItemTitle")
                    item_caption = item.find("./ItemCaption")
                    
                    if item_title is not None and item_title.text:
                        item_text = f"{item_title.text} "
                    else:
                        item_text = ""
                    
                    if item_caption is not None and item_caption.text:
                        item_text += item_caption.text
                    
                    if item_text:
                        para_content.append(item_text)
                
                paragraph_data['content'] = "\n".join(para_content)
                article_data['paragraphs'].append(paragraph_data)
                article_content.append(paragraph_data['content'])
            
            article_data['content'] = "\n".join(article_content)
            result['structured_articles'].append(article_data)
        
        return result
    except Exception as e:
        logger.error(f"法令構造の抽出中にエラーが発生しました: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # エラー時はテキストベースのフォールバック処理
        return {
            'full_text': ET.tostring(law_element, encoding='unicode'),
            'structured_articles': extract_articles_from_text(ET.tostring(law_element, encoding='unicode'))
        }
def extract_articles_from_text(text):
    """
    テキストベースで条文を抽出するフォールバック関数（改良版）
    XMLパースに失敗した場合に使用
    """
    structured_articles = []
    
    # より柔軟な条文パターンを定義
    article_patterns = [
        r'(第\s*[一二三四五六七八九十百千]+(?:[一二三四五六七八九十百千]+)*\s*条(?:\s*の\s*[一二三四五六七八九十百千]+(?:[一二三四五六七八九十百千]+)*)?)',  # 漢数字
        r'(第\s*\d+\s*条(?:\s*の\s*\d+)?)'  # アラビア数字
    ]
    
    # 各パターンで条文を探す
    for pattern in article_patterns:
        # 条文で分割を試みる
        matches = re.finditer(pattern, text)
        last_pos = 0
        article_sections = []
        
        for match in matches:
            article_num = match.group(1)
            start_pos = match.start()
            
            # 前の条文があれば追加
            if last_pos > 0:
                article_sections.append((article_sections[-1][0], text[last_pos:start_pos]))
            
            last_pos = start_pos
            article_sections.append((article_num, ""))
        
        # 最後の条文内容を追加
        if article_sections and last_pos < len(text):
            article_sections[-1] = (article_sections[-1][0], text[last_pos:])
        
        # 結果があれば処理
        if article_sections:
            for article_num, content in article_sections:
                # 項のパターン（日本語の項は「１　」のような形式が多い）
                paragraph_pattern = r'^\s*([0-9１２３４５６７８９０]+)\s+(.*?)$'
                paragraphs = []
                current_paragraph = None
                current_content = []
                
                # 項を抽出
                for line in content.split('\n'):
                    paragraph_match = re.match(paragraph_pattern, line)
                    if paragraph_match:
                        # 前の項があれば保存
                        if current_paragraph is not None:
                            paragraphs.append({
                                'paragraph_number': current_paragraph,
                                'content': '\n'.join(current_content)
                            })
                        
                        # 新しい項を開始
                        current_paragraph = paragraph_match.group(1)
                        current_content = [paragraph_match.group(2)]
                    else:
                        # 現在の項の続き
                        if current_paragraph is not None:
                            current_content.append(line)
                        elif line:  # 項に属さない内容は条文の直接の内容とする
                            current_content.append(line)
                
                # 最後の項を保存
                if current_paragraph is not None:
                    paragraphs.append({
                        'paragraph_number': current_paragraph,
                        'content': '\n'.join(current_content)
                    })
                
                structured_articles.append({
                    'article_number': article_num,
                    'content': content,
                    'paragraphs': paragraphs
                })
            
            # パターンが見つかったらループを抜ける
            if structured_articles:
                break
    
    # 何も見つからなかった場合のフォールバック
    if not structured_articles and text:
        structured_articles.append({
            'article_number': '全文',
            'content': text,
            'paragraphs': []
        })
    
    return structured_articles
        
     
        
    
def structure_law_text(law_text):
    """
    法令テキストを解析して条文番号と内容を構造化する関数（改良版）
    """
    if law_text is None or law_text.strip() == "":
        return {'full_text': law_text if law_text else "", 'structured_articles': []}
    
    # 返却用の辞書を初期化
    result = {
        'full_text': law_text,
        'structured_articles': []
    }
    
    # 条文のパターンを定義（「第X条」や「第X条の2」など）- より柔軟に
    article_pattern = r'(第\s*[一二三四五六七八九十百千]+(?:[一二三四五六七八九十百千]+)*\s*条(?:\s*の\s*[一二三四五六七八九十百千]+(?:[一二三四五六七八九十百千]+)*)?)'
    
    # まず前処理として、行ごとに分割して余分な空白を除去
    lines = [line.strip() for line in law_text.split('\n') if line.strip()]
    processed_text = '\n'.join(lines)
    
    # 条文で分割を試みる
    article_parts = re.split(f'({article_pattern})', processed_text)
    
    # デバッグログを追加
    print(f"Detected article parts: {len(article_parts)}")
    
    # 分割結果を処理
    i = 0
    current_article = None
    current_content = []
    
    while i < len(article_parts):
        part = article_parts[i].strip()
        if re.match(article_pattern, part):
            # 前の条文があれば保存
            if current_article is not None:
                content_text = '\n'.join(current_content)
                # 項の抽出
                paragraphs = extract_paragraphs(content_text)
                result['structured_articles'].append({
                    'article_number': current_article,
                    'content': content_text,
                    'paragraphs': paragraphs
                })
            
            # 新しい条文を開始
            current_article = part
            current_content = []
            i += 1
            
            # 次の部分が条文番号でなければ、その部分を内容として追加
            if i < len(article_parts) and not re.match(article_pattern, article_parts[i]):
                current_content.append(article_parts[i].strip())
        else:
            # 条文番号でなく、かつ現在の条文がある場合は、その条文の内容として追加
            if current_article is not None:
                current_content.append(part)
        i += 1
    
    # 最後の条文を保存
    if current_article is not None:
        content_text = '\n'.join(current_content)
        paragraphs = extract_paragraphs(content_text)
        result['structured_articles'].append({
            'article_number': current_article,
            'content': content_text,
            'paragraphs': paragraphs
        })
    
    # 条文が見つからなかった場合のフォールバック
    if not result['structured_articles'] and law_text.strip():
        # とりあえず全体を1つの条文として扱う
        result['structured_articles'].append({
            'article_number': '全文',
            'content': law_text,
            'paragraphs': []
        })
    
    return result
  
 

def extract_paragraphs(article_content):
    """
    条文内容から項を抽出する関数
    
    Returns:
        list: [
            {'paragraph_number': '1', 'content': '内容...'},
            {'paragraph_number': '2', 'content': '内容...'},
            ...
        ]
    """
    # 項のパターン（「1 」「２ 」などの数字で始まる行）
    paragraph_pattern = r'^\s*([0-9１２３４５６７８９０]+)\s+(.*?)$'
    
    paragraphs = []
    lines = article_content.split('\n')
    current_paragraph = None
    current_content = []
    
    for line in lines:
        paragraph_match = re.match(paragraph_pattern, line)
        if paragraph_match:
            # 前の項があれば保存
            if current_paragraph is not None:
                paragraphs.append({
                    'paragraph_number': current_paragraph,
                    'content': '\n'.join(current_content)
                })
            
            # 新しい項を開始
            current_paragraph = paragraph_match.group(1)
            current_content = [paragraph_match.group(2)]
        else:
            # 現在の項の続き
            if current_paragraph is not None:
                current_content.append(line)
    
    # 最後の項を保存
    if current_paragraph is not None:
        paragraphs.append({
            'paragraph_number': current_paragraph,
            'content': '\n'.join(current_content)
        })
    
    return paragraphs
def parse_law_list_xml(xml_data):
    """
    法令一覧のXMLを解析して法令名と法令IDの辞書を返す
    仕様書の「2.1.3. 応答結果XML」に基づいて実装
    """
    try:
        root = ET.fromstring(xml_data)
        law_dict = {}
        
        # ApplDataタグの存在確認
        appl_data = root.find("./ApplData")
        if appl_data is None:
            st.error("法令一覧データが見つかりませんでした")
            return {}
            
        # LawNameListInfoの各要素を処理
        for law_info in appl_data.findall("./LawNameListInfo"):
            law_id = law_info.find("LawId").text if law_info.find("LawId") is not None else ""
            law_name = law_info.find("LawName").text if law_info.find("LawName") is not None else ""
            law_no = law_info.find("LawNo").text if law_info.find("LawNo") is not None else ""
            
            if law_id and law_name:
                # 法令名と法令番号の両方を保存
                law_dict[law_name] = {"id": law_id, "no": law_no}
                
        return law_dict
    except Exception as e:
        st.error(f"XML解析中にエラーが発生しました: {e}")
        return {}
def search_law_by_name(law_name, law_dict):
    """
    法令名から法令IDを検索する関数
    検索精度を向上させた改良版
    """
    # 完全一致
    if law_name in law_dict:
        return law_dict[law_name]["id"], law_dict[law_name]["no"], law_name
    
    # 部分一致の検索（完全に含まれる場合）
    exact_matches = []
    for name, info in law_dict.items():
        if law_name in name:
            exact_matches.append((name, info["id"], info["no"]))
    
    if exact_matches:
        # 最も短い名前（最も具体的なもの）を優先
        exact_matches.sort(key=lambda x: len(x[0]))
        return exact_matches[0][1], exact_matches[0][2], exact_matches[0][0]
    
    # あいまい検索（似ている名前を探す）
    # 簡易的な実装として、法令名の一部が含まれるものを探す
    for word in law_name.split():
        if len(word) >= 2:  # 2文字以上の単語のみ対象
            for name, info in law_dict.items():
                if word in name:
                    return info["id"], info["no"], name
    
    return None, None, None

def get_law_content(law_id=None, law_num=None):
    """
    法令IDまたは法令番号から法令の全文を取得する関数（エラーハンドリング強化版）
    """
    # パラメータチェック
    if not law_id and not law_num:
        raise APIError("法令IDまたは法令番号が指定されていません", detail="少なくとも一方のパラメータが必要です")
    
    # URLの構築
    if law_id:
        url = f"{EGOV_API_BASE_URL}/lawdata/{law_id}"
        cache_key = f"law_{law_id}"
    else:
        url = f"{EGOV_API_BASE_URL}/lawdata/{law_num}"
        cache_key = f"law_num_{law_num}"
    
    # キャッシュチェック
    cached_data = load_from_cache("laws", cache_key)
    if cached_data:
        logger.info(f"法令データをキャッシュから取得しました: {law_id or law_num}")
        return cached_data
    
    # 接続確認
    if not check_api_connection():
        raise APIError("APIサーバーとの接続に失敗しました。しばらくしてからお試しください。")
    
    # APIリクエスト
    try:
        # プログレスメッセージ
        with st.status("法令データを取得中...", expanded=False) as status:
            status.update(label=f"法令 {law_id or law_num} を取得中...")
            response = requests.get(url, timeout=15)
            
            if response.status_code != 200:
                status_message = {
                    400: "パラメータが不正です",
                    404: "指定された法令が見つかりません",
                    406: "データが大きすぎるか、複数の法令データが存在します",
                    500: "APIサーバー内部でエラーが発生しています"
                }.get(response.status_code, f"予期しないステータスコード: {response.status_code}")
                
                status.update(label=f"法令データの取得に失敗しました: {status_message}", state="error")
                raise APIError(
                    f"法令データの取得に失敗しました: {status_message}", 
                    status_code=response.status_code,
                    detail=response.text[:200] if response.text else None
                )
            
            # レスポンス処理
            status.update(label="法令データを解析中...", state="running")
            success, xml_data = handle_api_response(response, "法令データの取得に失敗しました")
            
            if not success:
                if xml_data == "NOT_FOUND":
                    status.update(label="指定された法令が見つかりませんでした", state="complete")
                    logger.warning(f"法令が見つかりませんでした: {law_id or law_num}")
                    return "NOT_FOUND"
                
                status.update(label="法令データの解析に失敗しました", state="error")
                raise APIError("法令データの解析に失敗しました")
            
            # キャッシュに保存
            parsed_data = parse_law_content_xml(xml_data)
            if parsed_data:
                save_to_cache("laws", cache_key, parsed_data)
                status.update(label=f"法令データを取得しました: {law_id or law_num}", state="complete")
                logger.info(f"法令データをAPIから取得してキャッシュに保存しました: {law_id or law_num}")
                return parsed_data
            else:
                status.update(label="法令データの解析に失敗しました", state="error")
                raise APIError("法令データのパースに失敗しました")
    
    except requests.exceptions.Timeout:
        raise APIError("APIリクエストがタイムアウトしました", detail="法令データが大きいか、サーバーの応答が遅い可能性があります")
    except requests.exceptions.RequestException as e:
        raise APIError(f"APIリクエスト中にエラーが発生しました: {e}")

def get_article_content(law_id=None, law_num=None, article=None, paragraph=None, appdx_table=None):
    """
    条文内容を取得する関数
    仕様書の「2.3. 条文内容取得API」に基づいて実装
    """
    # URLパラメータの構築
    params = []
    
    if law_id:
        params.append(f"lawId={law_id}")
    elif law_num:
        params.append(f"lawNum={law_num}")
    else:
        st.error("法令IDまたは法令番号が指定されていません")
        return None
        
    if article:
        params.append(f"article={article}")
    
    if paragraph:
        params.append(f"paragraph={paragraph}")
    
    if appdx_table:
        params.append(f"appdxTable={appdx_table}")
        
    # URLの構築
    url = f"{EGOV_API_BASE_URL}/articles;{''.join(params)}"
    
    try:
        response = requests.get(url, timeout=15)
        success, xml_data = handle_api_response(response, "条文内容の取得に失敗しました")
        return xml_data if success else None
    except requests.exceptions.RequestException as e:
        st.error(f"条文内容の取得中にエラーが発生しました: {e}")
        return None



def parse_article_content_xml(xml_data):
    """
    条文内容のXMLを解析して本文を抽出する
    仕様書の「2.3.3. 応答結果XML」に基づいて実装
    """
    try:
        root = ET.fromstring(xml_data)
        
        # ApplDataタグの存在確認
        appl_data = root.find("./ApplData")
        if appl_data is None:
            st.error("条文内容データが見つかりませんでした")
            return None
            
        # 条文内容の抽出
        law_contents = appl_data.find("./LawContents")
        
        if law_contents is not None and law_contents.text:
            return law_contents.text
        else:
            # 複数候補の確認（別表の場合）
            appdx_table_lists = appl_data.find("./AppdxTableTitleLists")
            if appdx_table_lists is not None:
                titles = [title.text for title in appdx_table_lists.findall("./AppdxTableTitle")]
                return f"複数の別表候補があります: {', '.join(titles)}"
                
            # 画像データの確認
            image_data = appl_data.find("./ImageData")
            if image_data is not None:
                return "条文内容には画像が含まれています。詳細は元のウェブサイトでご確認ください。"
                
            return "条文内容を抽出できませんでした。"
    except Exception as e:
        st.error(f"条文内容の解析中にエラーが発生しました: {e}")
        return None

def answer_with_gemini(query, structured_law_content, law_name):
    """
    構造化された法令の内容を参照してGeminiで回答を生成する関数
    Gemini 2.0 Flash モデルを使用し、自然な文章での回答を生成
    """
    # 構造化された法令内容から条文リストを作成
    article_texts = []
    if isinstance(structured_law_content, dict):
        if 'structured_articles' in structured_law_content:
            # 全文から抽出した構造化データの場合
            for article in structured_law_content['structured_articles']:
                article_text = f"{article['article_number']}\n{article['content']}"
                article_texts.append(article_text)
        elif 'article_number' in structured_law_content:
            # 単一条文の場合
            article_number = structured_law_content.get('article_number', '')
            paragraph_number = structured_law_content.get('paragraph_number', '')
            content = structured_law_content.get('content', '')
            
            if article_number and paragraph_number:
                article_text = f"{article_number}第{paragraph_number}項\n{content}"
            elif article_number:
                article_text = f"{article_number}\n{content}"
            else:
                article_text = content
                
            article_texts.append(article_text)
    else:
        # 構造化されていない場合は全文をそのまま使用
        article_texts.append(structured_law_content)
    
    # 条文テキストを制限（長すぎると処理できない）
    combined_articles = "\n\n".join(article_texts)
    if len(combined_articles) > 30000:
        combined_articles = combined_articles[:30000] + "..."
    
    prompt = f"""
    あなたは日本の法律を専門とする弁護士です。一般の方からの質問に対して、専門用語をできるだけ使わず、わかりやすく自然な日本語で回答してください。

    法令名: {law_name}
    
    ユーザーの質問:
    {query}
    
    以下の法令内容を参照して回答してください:
    ---
    {combined_articles}
    ---
    
    回答の作成方法:
    1. 質問に関連する法的概念や罪名などを最初に簡潔に定義してください。例:「殺人罪とは、人を殺すこと、つまり故意に他人の生命を奪う犯罪です。」
    2. 参照する条文番号を必ず「第○条」「第○条第○項」などの形式で明示し、その内容を平易な言葉で説明してください。例:「刑法第199条に規定され、死刑または無期懲役、もしくは5年以上の懲役が科せられます。」
    3. 必要に応じて、条件や例外規定、実務上の適用例なども簡潔に説明してください。
    4. 法律の専門家ではない人にも理解できるよう、2-3段落程度の自然な文章で回答してください。
    5. 長い引用や条文の直接的な引用は避け、内容を咀嚼して伝えてください。
    
    重要:
    - 回答は簡潔で自然な日常会話のような文体を使ってください
    - 「～と考えられます」「～と思われます」などの曖昧な表現は避けてください
    - 質問に直接関係のない法的説明は省略してください
    - 条文番号は必ず明示してください（例：「第1条」「第2条第1項」など）
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"回答生成中にエラーが発生しました: {e}")
        return "回答を生成できませんでした。"

def generate_fallback_answer(query, relevant_laws):
    """
    法令が見つからない場合に一般的な知識で回答を生成する関数
    """
    law_list = ", ".join(relevant_laws) if relevant_laws else "関連する法令"
    
    prompt = f"""
    あなたは日本の法律を専門とする弁護士です。一般の方からの質問に対して、専門用語をできるだけ使わず、わかりやすく自然な日本語で回答してください。

    ユーザーの質問:
    {query}
    
    指示:
    この質問に対して、一般的な法律知識に基づいて回答してください。法令APIからの正確な情報が取得できなかったため、あなたの法律知識で回答する必要があります。
    
    回答の作成方法:
    1. 質問に関連する法的概念や罪名などを最初に簡潔に定義してください。例:「殺人罪とは、人を殺すこと、つまり故意に他人の生命を奪う犯罪です。」
    2. 関連する条文番号を明示し、その内容を平易な言葉で説明してください。例:「刑法第199条に規定され、死刑または無期懲役、もしくは5年以上の懲役が科せられます。」
    3. 必要に応じて、条件や例外規定、実務上の適用例なども簡潔に説明してください。
    4. 法律の専門家ではない人にも理解できるよう、2-3段落程度の自然な文章で回答してください。
    5. 回答の最後に「より正確な情報については、最新の{law_list}を参照してください。」と追加してください。
    
    重要:
    - 回答は簡潔で自然な日常会話のような文体を使ってください
    - 「～と考えられます」「～と思われます」などの曖昧な表現は避けてください
    - 質問に直接関係のない法的説明は省略してください
    - これは法令APIからの正確な情報ではなく、あなたの一般的な知識に基づく回答であることを明示してください
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"回答生成中にエラーが発生しました: {e}")
        return "回答を生成できませんでした。"

# 関連条文抽出関数を追加
def extract_relevant_articles(query, law_content, law_name):
    """
    法令の内容から質問に関連する条文だけを抽出する関数（改良版）
    """
    # 法令の内容をチェック
    if not law_content:
        return f"{law_name}が提供されていないため、条文を抽出することができません。"
    
    # フルテキストと構造化データを準備
    full_text = ""
    if isinstance(law_content, dict):
        if 'full_text' in law_content:
            full_text = law_content['full_text']
        
        if 'structured_articles' in law_content and law_content['structured_articles']:
            # 構造化されたデータがある場合は、それを使用
            articles_text = []
            for article in law_content['structured_articles']:
                article_text = f"{article['article_number']}\n{article['content']}"
                articles_text.append(article_text)
            
            full_text = "\n\n".join(articles_text) if not full_text else full_text
    else:
        full_text = str(law_content)
    
    # テキストが空でないか確認
    if not full_text or full_text.strip() == "":
        return f"{law_name}の条文データが空です。"
    
    # テキストサイズの制限（APIの制限を考慮）
    max_content_length = 30000
    if len(full_text) > max_content_length:
        full_text = full_text[:max_content_length] + "..."
    
    # Geminiを使用して関連条文を抽出
    prompt = f"""
    あなたは法令から関連条文を抽出する専門AIです。

    ユーザーの質問:
    {query}

    以下の法令（{law_name}）から、この質問に直接関連する条文のみを抽出してください:
    ---
    {full_text}
    ---

    抽出指示:
    1. 質問に最も関連する条文（条、項、号）を選択してください
    2. 各条文の前に条文番号を明記してください（例: 「第199条」）
    3. 関連する条文間の参照関係がある場合は、参照先の条文も含めてください
    4. 抽出した条文は元の文言を変えずにそのまま出力してください
    5. 最大で5つの条文に絞ってください - 最も重要なものだけを選んでください
    6. 条文が見つからない場合は、「{law_name}には、ご質問の内容に関する規定はありません。」と回答してください

    出力形式:
    【条文番号】
    条文内容

    【条文番号】
    条文内容
    ...
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"関連条文抽出中にエラーが発生しました: {e}")
        return f"{law_name}から関連条文を抽出中にエラーが発生しました。"

# 統合処理関数を追加
def process_legal_query(query, law_dict):
    """
    法的質問を処理して回答する統合関数（get_law_content の戻り値処理修正）
    """
    relevant_laws = identify_relevant_laws(query)
    if not relevant_laws:
        st.error("ご質問に関連する法令を特定できませんでした。質問の内容を具体的にしていただくか、別の表現で試してみてください。")
        return None, "関連法令特定失敗", [] # エラーメッセージを返すように変更

    answers = []
    found_laws = []
    # st.progress は withの外では使えないため st.empty() を使う
    progress_placeholder = st.empty()
    progress_bar = progress_placeholder.progress(0)
    progress_text = st.empty()
    total_steps = len(relevant_laws) * 3 + 1 # 特定、取得/解析、抽出、回答生成
    current_step = 0

    def update_progress(message):
        nonlocal current_step
        current_step += 1
        progress = min(float(current_step) / total_steps, 1.0)
        try:
            progress_bar.progress(progress)
            progress_text.text(message)
        except Exception as e:
            # Streamlit要素が古くなった場合のエラーを無視
            logger.warning(f"Streamlit progress update error: {e}")

    update_progress("関連法令を特定しました...")

    for i, law in enumerate(relevant_laws):
        structured_law = None # 初期化
        actual_law_name = law # デフォルト名
        try:
            update_progress(f"法令「{law}」を検索中...")
            law_id, law_num, actual_law_name_found = search_law_by_name(law, law_dict)
            if actual_law_name_found:
                actual_law_name = actual_law_name_found # 見つかった正式名称を使う

            if not law_id:
                logger.warning(f"法令「{law}」のIDが見つかりませんでした。")
                continue # 次の法令へ

            update_progress(f"法令「{actual_law_name}」のデータを取得/解析中...")
            # get_law_content は成功すれば辞書、失敗すれば"NOT_FOUND" or 例外を返す
            structured_law = get_law_content(law_id=law_id)

            if structured_law == "NOT_FOUND":
                logger.warning(f"APIから {actual_law_name} ({law_id}) が見つかりませんでした。")
                continue # 次の法令へ
            elif not isinstance(structured_law, dict):
                # get_law_content が予期せず辞書以外を返した場合（APIErrorなどはget_law_content内で処理されるはず）
                logger.error(f"get_law_content から予期しないデータ型 ({type(structured_law)}) が返されました: {actual_law_name}")
                continue

            # --- structured_law が辞書として得られた場合の処理 ---
            logger.info(f"法令「{actual_law_name}」の構造化データを取得しました。")

            # 3. 関連条文を抽出
            update_progress(f"法令「{actual_law_name}」の関連条文を抽出中...")
            try:
                # extract_relevant_articles には常に辞書データを渡す
                relevant_articles_text = extract_relevant_articles(query, structured_law, actual_law_name)
            except Exception as e:
                logger.error(f"条文抽出エラー ({actual_law_name}): {e}")
                logger.error(traceback.format_exc())
                relevant_articles_text = f"{actual_law_name} から関連条文を抽出中にエラーが発生しました。" # エラーメッセージ変更済み

            # 4. 回答を生成
            update_progress(f"法令「{actual_law_name}」に基づく回答を生成中...")
            try:
                answer = answer_with_gemini(query, structured_law, actual_law_name)
            except Exception as e:
                logger.error(f"回答生成エラー ({actual_law_name}): {e}")
                logger.error(traceback.format_exc())
                answer = f"{actual_law_name} に基づく回答を生成中にエラーが発生しました。"

            answers.append((actual_law_name, answer, relevant_articles_text))
            found_laws.append(actual_law_name)

        except APIError as e: # get_law_content 内で発生したAPIErrorもここでキャッチ
            logger.error(f"法令処理中 ({actual_law_name}) にAPIエラー: {e.message} (Detail: {e.detail})")
            # st.error(f"法令「{actual_law_name}」の処理中にAPIエラーが発生しました: {e.message}") # UIへの表示は適宜
        except Exception as e:
            logger.error(f"法令処理中 ({actual_law_name}) に予期せぬエラー: {e}")
            logger.error(traceback.format_exc())
            # st.error(f"法令「{actual_law_name}」の処理中に予期しないエラーが発生しました。") # UIへの表示は適宜
        # finallyブロックは削除（ループ内でステップ更新するため）

    # ループ終了後にプログレスバーを完了させる
    try:
        progress_bar.progress(1.0)
        progress_text.text("処理が完了しました。")
        # 少し待ってから要素を消す（オプション）
        time.sleep(1)
        progress_placeholder.empty()
        progress_text.empty()
    except Exception as e:
        logger.warning(f"Streamlit progress clear error: {e}")


    if answers:
        # 主回答と全回答リストを返す
        return True, answers[0][1], answers
    else:
        # 有効な法令が見つからなかった場合
        logger.warning("関連法令が見つかりましたが、有効なデータを取得・解析できませんでした。")
        # フォールバック回答を試みるか、専用メッセージを返す
        fallback_answer = generate_fallback_answer(query, relevant_laws)
        if fallback_answer == "回答を生成できませんでした。": # フォールバックも失敗した場合
             return False, f"関連する可能性のある法令 ({', '.join(relevant_laws)}) が見つかりましたが、内容を取得・解析できませんでした。", []
        else:
             return False, fallback_answer, [] # フォールバック回答を返す
     

# メイン検索処理の改善版

     
setup_cache_directories()
# メイン処理の改善
# ユーザー入力
# メイン処理部分（修正版）
# ユーザー入力
# メインUIセクション
# ユーザー入力部分（以前の方法で入力を受け付ける）
# ユーザー入力
user_query = st.text_area("質問を入力してください", height=100)

# 検索ボタン
if st.button("検索"):
    if user_query:
        with st.spinner("検索中..."):
            # 法令一覧を取得
            law_list_xml = get_law_list()
            if law_list_xml:
                law_dict = parse_law_list_xml(law_list_xml)
                
                if law_dict:
                    # 処理を統合関数に委譲
                    with st.spinner("回答を生成中..."):
                        found_law, main_answer, all_answers = process_legal_query(user_query, law_dict)
                        
                        # 回答の表示
                        st.subheader("回答")
                        st.write(main_answer)
                        
                        # 関連法令が見つかった場合のみ表示
                        if found_law and len(all_answers) > 0:
                            # 関連法令のリストを表示
                            with st.expander("関連する法令"):
                                for law_name, _, _ in all_answers:
                                    st.write(f"- {law_name}")
                            
                            # 他の法令からの情報がある場合は折りたたみセクションで表示
                            if len(all_answers) > 1:
                                with st.expander("その他の関連法令からの情報"):
                                    for i in range(1, len(all_answers)):
                                        st.subheader(all_answers[i][0])
                                        st.write(all_answers[i][1])
                            
                            # # 関連条文の表示
                            # for law_name, _, relevant_articles in all_answers:
                            #     with st.expander(f"{law_name}の関連条文"):
                            #         # 関連条文が文字列でない場合（辞書など）の対応
                            #         if isinstance(relevant_articles, str):
                            #             st.write(relevant_articles)
                            #         else:
                            #             st.write("関連条文を抽出できませんでした。")
                else:
                    st.error("法令一覧の解析に失敗しました。")
    else:
        st.warning("質問を入力してください")
