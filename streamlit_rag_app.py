import os
import io
import fitz
import faiss
import ollama
import numpy as np
import pandas as pd
import streamlit as st
from docx import Document
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Local RAG Chatbot", layout="wide")

EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "gemma4:31b-cloud"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 3

@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)

embedder = get_embedder()


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def load_txt_bytes(file_bytes):
    return file_bytes.decode("utf-8", errors="ignore")


def load_pdf_bytes(file_bytes):
    texts = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc):
            page_text = page.get_text(sort=True)
            if page_text and page_text.strip():
                texts.append(f"[Page {page_num + 1}]\n{page_text}")
    return "\n".join(texts)


def load_docx_bytes(file_bytes):
    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def load_csv_bytes(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    return df.fillna("").astype(str).to_csv(index=False)


def load_xlsx_bytes(file_bytes):
    sheets = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
    parts = []
    for sheet_name, df in sheets.items():
        parts.append(f"[Sheet: {sheet_name}]")
        parts.append(df.fillna("").astype(str).to_csv(index=False))
    return "\n".join(parts)


def load_uploaded_document(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    file_bytes = uploaded_file.getvalue()
    if ext in [".txt", ".md"]:
        return load_txt_bytes(file_bytes)
    if ext == ".pdf":
        return load_pdf_bytes(file_bytes)
    if ext == ".docx":
        return load_docx_bytes(file_bytes)
    if ext == ".csv":
        return load_csv_bytes(file_bytes)
    if ext in [".xlsx", ".xls"]:
        return load_xlsx_bytes(file_bytes)
    return None


class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []
        self.sources = []

    def add(self, chunks, sources):
        vecs = embedder.encode(chunks)
        vecs = np.array(vecs).astype("float32")
        self.index.add(vecs)
        self.texts.extend(chunks)
        self.sources.extend(sources)

    def search(self, query, k=TOP_K):
        qvec = embedder.encode([query]).astype("float32")
        distances, indices = self.index.search(qvec, k)
        results = []
        for i in indices[0]:
            if 0 <= i < len(self.texts):
                results.append({"text": self.texts[i], "source": self.sources[i]})
        return results


def build_store(uploaded_files):
    all_chunks = []
    all_sources = []
    loaded_docs = []
    for uploaded_file in uploaded_files:
        text = load_uploaded_document(uploaded_file)
        if text and text.strip():
            chunks = chunk_text(text)
            for chunk in chunks:
                all_chunks.append(chunk)
                all_sources.append(uploaded_file.name)
            loaded_docs.append(uploaded_file.name)
    if not all_chunks:
        return None, loaded_docs, 0
    store = VectorStore(dim=384)
    store.add(all_chunks, all_sources)
    return store, loaded_docs, len(all_chunks)


def llm_call(prompt, model_name=DEFAULT_LLM_MODEL):
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.6
        }
    )
    return response["message"]["content"]


def rag_query(question, store, model_name, history, top_k):
    results = store.search(question, k=top_k)

     # DEBUG LOGGING
    print("\n[DEBUG] Retrieved context for question:", question)
    for r in results:
        print("Source:", r["source"])
        print(r["text"][:500], "\n")

    context = "\n\n---\n\n".join(
        [f"Source: {r['source']}\n{r['text']}" for r in results]
    )
    recent_history = "\n".join(
        [f"User: {q}\nAssistant: {a}" for q, a in history[-3:]]
    )

    prompt = f"""
You are a helpful, friendly assistant.

TASK:
You must answer the user's question using ONLY the rules and facts
found in the provided context. If the context contains a rule that
directly implies the answer (for example, it says something is
"non-refundable" or "cannot be returned"), use that rule to answer
clearly and confidently.

Do NOT say "I couldn't find that" if there is a rule in the context
that clearly applies to the question, even if the question is asked
in a different way (like "How can I return X?" when the rule says
"X is non-refundable").

Only when there is truly NO relevant rule or fact in the context
should you say: "I couldn't find that in the uploaded documents."

STYLE:
- Answer in natural, human-like, conversational language.
- Always use complete sentences.
- Be clear and polite.
- Do not invent any new policies or rules.

EXAMPLE OF CORRECT BEHAVIOR:
Context: "Digital products and gift cards are non-refundable."
User question: "How can I return gift card?"
Correct answer: "According to the uploaded policy, gift cards are non-refundable, so you can't return a gift card."

Recent chat history:
{recent_history}

Context:
{context}

User question:
{question}

Answer:
""".strip()

    answer = llm_call(prompt, model_name)
    return answer, results

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history_pairs" not in st.session_state:
    st.session_state.chat_history_pairs = []
if "store" not in st.session_state:
    st.session_state.store = None
if "loaded_docs" not in st.session_state:
    st.session_state.loaded_docs = []
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

st.title("Local RAG Chatbot")
st.caption("Upload documents, build an index, and chat with them locally using Ollama.")

with st.sidebar:
    st.header("Settings")
    model_name = st.text_input("Ollama model", value=DEFAULT_LLM_MODEL)
    top_k = st.slider("Top K chunks", min_value=1, max_value=8, value=3)
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "xlsx", "xls", "txt", "csv", "md"],
        accept_multiple_files=True,
    )

    if st.button("Build / Rebuild Index", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one supported file.")
        else:
            with st.spinner("Processing documents and building vector index..."):
                store, loaded_docs, chunk_count = build_store(uploaded_files)
                st.session_state.store = store
                st.session_state.loaded_docs = loaded_docs
                st.session_state.chunk_count = chunk_count
                st.session_state.messages = []
                st.session_state.chat_history_pairs = []
            if store is not None:
                st.success(f"Indexed {len(loaded_docs)} file(s) into {chunk_count} chunk(s).")
            else:
                st.error("No readable text could be extracted from the uploaded files.")

    if st.session_state.loaded_docs:
        st.markdown("### Indexed files")
        for name in st.session_state.loaded_docs:
            st.write(f"- {name}")
        st.write(f"Chunks: {st.session_state.chunk_count}")

st.markdown("---")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                seen = set()
                for src in message["sources"]:
                    key = (src["source"], src["text"][:120])
                    if key in seen:
                        continue
                    seen.add(key)
                    st.markdown(f"**{src['source']}**")
                    st.write(src["text"][:1200])

user_prompt = st.chat_input("Ask a question about your uploaded documents")

if user_prompt:
    if st.session_state.store is None:
        st.warning("Please upload documents and click 'Build / Rebuild Index' first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, sources = rag_query(
                        user_prompt,
                        st.session_state.store,
                        model_name,
                        st.session_state.chat_history_pairs,
                        top_k,
                    )
                    st.markdown(answer)
                    with st.expander("Sources"):
                        seen = set()
                        for src in sources:
                            key = (src["source"], src["text"][:120])
                            if key in seen:
                                continue
                            seen.add(key)
                            st.markdown(f"**{src['source']}**")
                            st.write(src["text"][:1200])
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                    st.session_state.chat_history_pairs.append((user_prompt, answer))
                except Exception as e:
                    err = f"Something went wrong: {e}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
