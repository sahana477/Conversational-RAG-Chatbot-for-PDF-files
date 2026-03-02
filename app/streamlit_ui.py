import streamlit as st
import requests

st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")

# ---------- Custom ChatGPT-like CSS ----------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 1rem;
    max-width: 900px;
}

.chat-container {
    max-height: 70vh;
    overflow-y: auto;
}

.stChatMessage {
    border-radius: 12px;
    padding: 0.8rem 1rem;
}

.answer-box {
    background: #f7f7f8;
    padding: 1rem;
    border-radius: 10px;
    margin-top: 0.5rem;
}

.chunk-box {
    background: #ffffff;
    border: 1px solid #e5e5e5;
    padding: 0.7rem;
    border-radius: 8px;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ---------- Sidebar ----------
with st.sidebar:
    st.title("📄 PDF RAG Chatbot")
    uploaded_pdf = st.file_uploader("Upload your PDF", type=["pdf"])
    
    if uploaded_pdf:
        st.session_state["pdf_file"] = uploaded_pdf
    elif "pdf_file" not in st.session_state:
        st.session_state["pdf_file"] = None

    if st.button("🆕 New Chat"):
        st.session_state.messages = []

    st.markdown("---")
    st.markdown("**Backend Stack:**")
    st.markdown("- FastAPI")
    st.markdown("- FAISS")
    st.markdown("- Langfuse")


# ---------- Session State ----------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------- Display Chat History ----------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(f'<div class="answer-box">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(message["content"])

        # Show retrieved chunks if assistant message
        if message["role"] == "assistant" and "retrieved" in message:
            with st.expander("📚 Retrieved Chunks"):
                for i, chunk in enumerate(message["retrieved"]):
                    st.markdown(f"**Chunk {i+1} ({chunk['type']}, page {chunk.get('page','?')})**")

                    if chunk["type"] == "image":
                        st.image(chunk["path"], use_container_width=True)
                    elif chunk["type"] == "table":
                        st.table(chunk["content"])
                    else:
                        st.code(str(chunk["content"]))


# ---------- Chat Input (directly below history) ----------
pdf_file = st.session_state.get("pdf_file")
prompt = st.chat_input("Ask something about your PDF...")
if prompt:

    if not pdf_file:
        st.warning("Please upload a PDF first.")
        st.stop()

    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Call backend
    url = "http://localhost:8000/chat"
    files = {"pdf": (pdf_file.name, pdf_file, "application/pdf")}
    data = {"query": prompt}

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(url, data=data, files=files, timeout=180)

                if response.status_code == 200:
                    res_data = response.json()
                    answer = res_data.get("answer", "No answer returned.")
                    retrieved = res_data.get("retrieved", [])

                    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

                    # Store assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "retrieved": retrieved
                    })

                    # Show retrieved chunks
                    if retrieved:
                        with st.expander("📚 Retrieved Chunks"):
                            for i, chunk in enumerate(retrieved):
                                st.markdown(f"**Chunk {i+1} ({chunk['type']}, page {chunk.get('page','?')})**")

                                if chunk["type"] == "image":
                                    st.image(chunk["path"], use_container_width=True)
                                elif chunk["type"] == "table":
                                    st.table(chunk["content"])
                                else:
                                    st.code(str(chunk["content"]))

                else:
                    st.error(f"Error: {response.status_code} {response.text}")

            except Exception as e:
                st.error(f"Request failed: {e}")