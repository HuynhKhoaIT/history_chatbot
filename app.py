"""Streamlit chat UI cho RAG Lịch Sử Việt Nam.

Chạy:  streamlit run app.py
Tái dùng pipeline từ main.py (answer_query) + ChatHistory.
"""

import time

import streamlit as st

from config import ensure_index, CATEGORIES
from history import ChatHistory
from main import check_input, answer_query


def stream_text(text: str, delay: float = 0.012):
    """Phát text ra từng từ để tạo hiệu ứng streaming (typewriter)."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(delay)

st.set_page_config(page_title="RAG Lịch Sử Việt Nam", page_icon="🇻🇳")


@st.cache_resource(show_spinner="Đang kết nối Pinecone...")
def get_index():
    """Tạo/lấy index 1 lần, cache suốt session app."""
    return ensure_index()


def init_state():
    if "history" not in st.session_state:
        st.session_state.history = ChatHistory()
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list[{"role", "content"}] để render


init_state()
index = get_index()

# --- Sidebar ---
with st.sidebar:
    st.header("🇻🇳 RAG Lịch Sử Việt Nam")
    st.caption("Pipeline: intent → PDF RAG → Wikipedia → Model fallback")
    st.subheader("Phạm vi (categories)")
    for key, desc in CATEGORIES.items():
        st.markdown(f"- **{key}**: {desc}")
    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.history = ChatHistory()
        st.session_state.messages = []
        st.rerun()

# --- Lịch sử hiển thị ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Input ---
prompt = st.chat_input("Hỏi về lịch sử Việt Nam...")
if prompt:
    ok, query, err = check_input(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not ok:
            answer = f"⚠️ {err}"
            st.markdown(answer)
            intent = None
        else:
            with st.spinner("Đang tra cứu..."):
                answer, intent_result = answer_query(
                    query, index, st.session_state.history, verbose=True
                )
            st.write_stream(stream_text(answer))
            with st.expander("ℹ️ Chi tiết intent"):
                st.write(
                    {
                        "intent": intent_result.intent,
                        "category": intent_result.category,
                        "confidence": round(intent_result.confidence, 2),
                        "reasoning": intent_result.reasoning,
                        "rewritten_query": intent_result.rewritten_query,
                    }
                )
            intent = intent_result.intent

    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Cập nhật ChatHistory (logic giống CLI: bỏ qua clarify)
    if ok and intent != "clarify":
        st.session_state.history.append("user", query)
        st.session_state.history.append("assistant", answer)
