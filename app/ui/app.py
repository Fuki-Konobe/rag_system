import streamlit as st
import time
import requests
import json

# APIのベースURL（Docker内でのサービス名またはlocalhost）
API_URL = "http://rag_api:8000"

st.set_page_config(page_title="RAG System", layout="wide")

st.title("RAGシステム")
st.markdown("資料をアップロードして、AIに質問")

# サイドバー：ファイルアップロード機能
with st.sidebar:
    st.header("資料の学習")
    uploaded_file = st.file_uploader("PDFを選択", type="pdf")
    
    if st.button("学習を開始"):
        if uploaded_file:
            # 1. アップロード・保存フェーズ
            with st.spinner("ファイルをサーバーへ送信中..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    response = requests.post(f"{API_URL}/upload", files=files)
                    if response.status_code != 200:
                        st.error("アップロードに失敗しました。")
                        st.stop()
                except Exception as e:
                    st.error(f"通信エラー: {e}")
                    st.stop()

            # 2. インデックス更新監視フェーズ
            status_text = st.empty() # 状態を書き換えるためのプレースホルダ
            start_time = time.time()
            timeout = 60  # 60秒でタイムアウト

            with st.spinner("新しい知識を脳内に定着させています..."):
                while True:
                    try:
                        res = requests.get(f"{API_URL}/status")
                        is_indexing = res.json().get("is_indexing")
                        
                        if not is_indexing:
                            st.success("学習完了！質問を受け付けます。")
                            break
                    except:
                        pass # 一時的な通信エラーは無視して続行
                    
                    # タイムアウトチェック
                    if time.time() - start_time > timeout:
                        st.error("処理がタイムアウトしました。サーバーの状態を確認してください。")
                        break
                    
                    time.sleep(1)  # ここで1秒待機するのが重要！
        else:
            st.warning("ファイルを選択してください。")


# メイン画面：チャット履歴の管理
if "messages" not in st.session_state:
    st.session_state.messages = []

# 履歴の表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def response_generator(prompt):
    # セッション状態をリセット
    st.session_state.last_sources = []
    
    with requests.post(f"{API_URL}/ask_stream", params={"question": prompt}, stream=True) as r:
        for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
            if "SOURCES_JSON:" in chunk:
                # 1. 回答テキストとJSONを分離
                parts = chunk.split("SOURCES_JSON:")
                yield parts[0]
                
                try:
                    # 2. JSONをパース（ユニコードも自動で日本語に戻ります）
                    raw_sources = json.loads(parts[1])
                    
                    # 3. 無効なデータ(null)を除去し、重複を排除
                    cleaned_sources = []
                    seen = set()
                    for s in raw_sources:
                        fname = s.get("file")
                        pnum = s.get("page")
                        if fname: # ファイル名が存在する場合のみ追加
                            label = f"{fname} (p.{pnum})"
                            if label not in seen:
                                cleaned_sources.append(s)
                                seen.add(label)
                    
                    st.session_state.last_sources = cleaned_sources
                except Exception as e:
                    print(f"JSONパースエラー: {e}")
            else:
                yield chunk

# ユーザー入力
if prompt := st.chat_input("質問を入力"):
    # ユーザーメッセージを表示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AIの回答を取得
    with st.chat_message("assistant"):
        # ストリーミング表示の実行
        full_response = st.write_stream(response_generator(prompt))

        # 終了後にメタデータを表示
        if st.session_state.get("last_sources"):
            with st.expander("📚 参照した根拠資料"):
                for s in st.session_state.last_sources:
                    st.caption(f"📄 {s['file']} (p.{s['page']})")

        st.session_state.messages.append({"role": "assistant", "content": full_response})