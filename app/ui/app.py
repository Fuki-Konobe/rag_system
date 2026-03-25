import streamlit as st
import requests

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
            with st.spinner("解析中..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post(f"{API_URL}/upload", files=files)
                if response.status_code == 200:
                    st.success("学習が完了しました！")
                else:
                    st.error("エラーが発生しました。")
        else:
            st.warning("ファイルを選択してください。")

# メイン画面：チャット履歴の管理
if "messages" not in st.session_state:
    st.session_state.messages = []

# 履歴の表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザー入力
if prompt := st.chat_input("質問を入力"):
    # ユーザーメッセージを表示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AIの回答を取得
    with st.chat_message("assistant"):
        # st.write_stream を使うと、ジェネレータから渡される文字をアニメーション表示できます
        def response_generator():
            # stream=True でリクエストを送り、逐次読み取る
            with requests.post(f"{API_URL}/ask_stream", params={"question": prompt}, stream=True) as r:
                for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        yield chunk

        # ストリーミング表示の実行
        full_response = st.write_stream(response_generator())
        st.session_state.messages.append({"role": "assistant", "content": full_response})