import streamlit as st
import requests
import json

# --- Custom CSS for aesthetics ---
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2a4d69;
        margin-bottom: 0.5em;
        text-align: center;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #4b86b4;
        text-align: center;
        margin-bottom: 2em;
    }
    .stButton > button {
        background-color: #4b86b4;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5em 2em;
        margin-top: 1em;
    }
    .stTextInput > div > input {
        border-radius: 8px;
        border: 1px solid #4b86b4;
        padding: 0.5em;
    }
    .chunk-title {
        color: #2a4d69;
        font-weight: 600;
        margin-top: 1em;
    }
    .answer-box {
        background: #f1f6fa;
        border-radius: 8px;
        padding: 1em;
        margin-bottom: 1em;
        border: 1px solid #dbe9f4;
    }
    .sidebar-logo {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 80%;
        margin-bottom: 1em;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.image("https://raw.githubusercontent.com/streamlit/brand-assets/main/streamlit-logo-primary-colormark-darktext.png", width='stretch', output_format="PNG", caption="PDF RAG Chatbot", channels="sidebar-logo")
st.sidebar.title("PDF RAG Chatbot")
st.sidebar.markdown("""
**Instructions:**
- Enter a question about your PDF (text, tables, images)
- The chatbot will retrieve relevant chunks and generate an answer
- You can view retrieved chunks and images
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Backend:** [FastAPI](https://fastapi.tiangolo.com/) | [Langfuse](https://langfuse.com/) | [FAISS](https://github.com/facebookresearch/faiss)")

# --- Main UI ---
st.markdown('<div class="main-title">PDF RAG Chatbot Demo</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask questions about your PDF and get answers with context, tables, and images.</div>', unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state["history"] = []

with st.form("chat_form"):
    query = st.text_input("Enter your question:", key="query_input")
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"], key="pdf_upload")
    submitted = st.form_submit_button("Ask")

def show_chunks(chunks):
    for i, chunk in enumerate(chunks):
        st.markdown(f'<div class="chunk-title">Chunk {i+1} ({chunk["type"]}, page {chunk.get("page","?")}):</div>', unsafe_allow_html=True)
        if chunk['type'] == 'image':
            st.image(chunk['path'], caption=f"Page {chunk.get('page','?')}", width='stretch')
        elif chunk['type'] == 'table':
            st.table(chunk['content'])
        else:
            st.code(str(chunk['content']))

if submitted and query and uploaded_pdf:
    url = "http://localhost:8000/chat"
    files = {"pdf": (uploaded_pdf.name, uploaded_pdf, "application/pdf")}
    data = {"query": query}
    with st.spinner("Retrieving answer..."):
        try:
            response = requests.post(url, data=data, files=files, timeout=180)
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "No answer returned.")
                retrieved = data.get("retrieved", [])
                st.session_state["history"].append({"query": query, "answer": answer, "retrieved": retrieved})
            else:
                st.error(f"Error: {response.status_code} {response.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
elif submitted and not uploaded_pdf:
    st.warning("Please upload a PDF file to ask questions.")

# --- Conversation History ---
if st.session_state["history"]:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## Conversation History")
    for idx, entry in enumerate(reversed(st.session_state["history"])):
        with st.expander(f"Q{len(st.session_state['history'])-idx}: {entry['query']}", expanded=(idx==0)):
            st.markdown(f'<div class="answer-box"><b>Answer:</b> {entry["answer"]}</div>', unsafe_allow_html=True)
            st.markdown("**Retrieved Chunks:**")
            show_chunks(entry['retrieved'])
