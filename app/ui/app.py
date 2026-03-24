import streamlit as st
import requests

# APIのベースURL（Docker内でのサービス名またはlocalhost）
API_URL = "http://rag_api:8000"

st.set_page_config(page_title="Tokushima RAG Assistant", layout="wide")

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
        with st.spinner("思考中..."):
            response = requests.post(f"{API_URL}/ask", params={"question": prompt})
            if response.status_code == 200:
                answer = response.json()["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.error("APIとの通信に失敗しました。")