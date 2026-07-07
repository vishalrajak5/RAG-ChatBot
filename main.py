import os
import sys
import fitz
import faiss
import ollama
import numpy as np
import pandas as pd

from docx import Document
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================
DOCUMENTS_FOLDER = "documents"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"
TOP_K = 3
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
DEBUG = False

# =========================
# TEXT CHUNKING
# =========================
def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

# =========================
# FILE LOADERS
# =========================
def load_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def load_pdf(file_path):
    texts = []
    with fitz.open(file_path) as doc:
        for page_num, page in enumerate(doc):
            page_text = page.get_text(sort=True)
            if page_text and page_text.strip():
                texts.append(f"[Page {page_num + 1}]\n{page_text}")
    return "\n".join(texts)

def load_docx(file_path):
    doc = Document(file_path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)

def load_csv(file_path):
    df = pd.read_csv(file_path)
    return df.fillna("").astype(str).to_csv(index=False)

def load_xlsx(file_path):
    sheets = pd.read_excel(file_path, sheet_name=None)
    parts = []

    for sheet_name, df in sheets.items():
        parts.append(f"[Sheet: {sheet_name}]")
        df = df.fillna("").astype(str)
        parts.append(df.to_csv(index=False))

    return "\n".join(parts)

def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext in [".txt", ".md"]:
            return load_txt(file_path)
        elif ext == ".pdf":
            return load_pdf(file_path)
        elif ext == ".docx":
            return load_docx(file_path)
        elif ext == ".csv":
            return load_csv(file_path)
        elif ext in [".xlsx", ".xls"]:
            return load_xlsx(file_path)
        else:
            return None
    except Exception as e:
        print(f"Skipping {file_path} due to error: {e}")
        return None

def load_all_documents(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created '{folder_path}' folder. Put your files there and run again.")
        sys.exit(0)

    docs = []

    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)

        if os.path.isfile(full_path):
            text = load_document(full_path)
            if text and text.strip():
                docs.append({
                    "filename": filename,
                    "text": text
                })

    return docs

# =========================
# VECTOR STORE
# =========================
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

    def search(self, query, k=3):
        qvec = embedder.encode([query]).astype("float32")
        distances, indices = self.index.search(qvec, k)

        results = []
        for i in indices[0]:
            if 0 <= i < len(self.texts):
                results.append({
                    "text": self.texts[i],
                    "source": self.sources[i]
                })
        return results

# =========================
# LLM CALL
# =========================
def llm_call(prompt):
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.6
        }
    )
    return response["message"]["content"]

# =========================
# RAG QUERY
# =========================
chat_history = []

def rag_query(question, k=TOP_K):
    results = store.search(question, k=k)

    if DEBUG:
        print("\n[DEBUG] Retrieved Chunks:")
        for idx, item in enumerate(results, 1):
            print(f"\n--- Chunk {idx} | Source: {item['source']} ---")
            print(item["text"][:1000])

    context = "\n\n---\n\n".join(
        [f"Source: {item['source']}\n{item['text']}" for item in results]
    )

    recent_history = "\n".join(
        [f"User: {q}\nAssistant: {a}" for q, a in chat_history[-3:]]
    )

    prompt = f"""
You are a helpful, friendly assistant.

Answer in a natural, human-like, conversational way.
Use ONLY the information from the provided context.
Use recent chat history only to understand follow-up questions.
Always answer in complete sentences.
Keep the answer clear and helpful, not robotic.
If the answer is not available in the context, say:
"I couldn't find that in the uploaded documents."
Do not invent facts.

Style examples:
User: Do you offer support on weekends?
Assistant: Support availability depends on the service terms, so weekend support may or may not be included.

User: Is delivery available everywhere?
Assistant: Delivery availability can vary by location, so it depends on the area being served.

Recent chat history:
{recent_history}

Context:
{context}

User question:
{question}

Answer:
""".strip()

    answer = llm_call(prompt)
    chat_history.append((question, answer))
    return answer, results

# =========================
# BUILD INDEX
# =========================
print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

print(f"Reading documents from '{DOCUMENTS_FOLDER}'...")
documents = load_all_documents(DOCUMENTS_FOLDER)

if not documents:
    print(f"No supported files found in '{DOCUMENTS_FOLDER}'.")
    print("Supported formats: PDF, DOCX, XLSX, XLS, TXT, CSV, MD")
    sys.exit(0)

all_chunks = []
all_sources = []

for doc in documents:
    chunks = chunk_text(doc["text"], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    for chunk in chunks:
        all_chunks.append(chunk)
        all_sources.append(doc["filename"])

print(f"Loaded {len(documents)} document(s)")
print(f"Created {len(all_chunks)} chunk(s)")

store = VectorStore(dim=384)
store.add(all_chunks, all_sources)

print("RAG chatbot is ready. Type 'exit' to quit.\n")

# =========================
# CHAT LOOP
# =========================
while True:
    question = input("You: ").strip()

    if question.lower() in ["exit", "quit", "bye"]:
        print("Bot: Goodbye!")
        break

    if not question:
        continue

    try:
        answer, sources = rag_query(question)

        print(f"Bot: {answer}\n")

        unique_sources = list(dict.fromkeys([item["source"] for item in sources]))
        if unique_sources:
            print("Sources:", ", ".join(unique_sources))
            print()

    except Exception as e:
        print(f"Bot: Something went wrong: {e}\n")