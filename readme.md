# NewsPulse: Multi-Article AI Research Agent

A lightweight Retrieval-Augmented Generation (RAG) tool built to ingest, index, and query multiple online news articles simultaneously using LangChain, FAISS, and NVIDIA Nemotron-70B.

---

## 📌 Overview

Reading through 5–10 long news articles or press releases to cross-examine facts takes time. Standard LLM web browsers often miss context, summarize too aggressively, or hallucinate citations. 

**NewsPulse** solves this by building a localized vector index across any set of web URLs you provide. You can ask complex questions across all articles at once, and get back factual answers grounded strictly in the source material, complete with direct chunk citations.

---

## 🛠️ Key Features

* **Multi-URL Ingestion:** Pulls clean textual content directly from web links using `WebBaseLoader`.
* **Deterministic Text Chunking:** Employs `RecursiveCharacterTextSplitter` to maintain paragraph coherence and prevent split sentence loss.
* **In-Memory Vector Search:** Uses local 384-dimensional HuggingFace embeddings (`all-MiniLM-L6-v2`) and a local `FAISS` vector index for sub-10ms retrieval.
* **Grounded Answer Generation:** Powered by **NVIDIA Nemotron-70B NIM** via LangChain Expression Language (LCEL) pipelines with strict context-bounding prompts.
* **Source Transparency:** Shows exact text chunks and source URLs fetched during retrieval for verification.

---

## 📐 System Architecture

```
[ Web URLs ] 
      │
      ▼
[ WebBaseLoader ] ──► Extracted Article Content
      │
      ▼
[ Recursive Text Splitter ] ──► Chunks (size: 1000, overlap: 200)
      │
      ▼
[ HuggingFace Embeddings ] ──► 384-dim Vectors
      │
      ▼
[ FAISS Vector Store ] (In-Memory Index)
      │
  [ User Query ] ──► Vector Search (Top-k=4)
      │
      ▼
[ Context + Prompt ] ──► [ NVIDIA Nemotron-70B NIM ] ──► [ Answer with Citations ]
```

---

## 💻 Tech Stack

* **Frontend:** Streamlit
* **LLM Orchestration:** LangChain (LCEL)
* **LLM Inference:** NVIDIA NIM Endpoint (`nvidia/llama-3.1-nemotron-70b-instruct`)
* **Vector Database:** FAISS (`faiss-cpu`)
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` via `langchain-huggingface`
* **Parser/Loader:** BeautifulSoup4 (`WebBaseLoader`)

---

## 🚀 Local Setup

### Prerequisites
* Python 3.10 or higher
* An NVIDIA API Key (Get one at [build.nvidia.com](https://build.nvidia.com))

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/news-rag-agent.git
   cd news-rag-agent
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/venv/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration:**
   Create a `.env` file in the project root:
   ```env
   NVIDIA_API_KEY=your_nvidia_api_key_here
   ```

5. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

---

## 💡 Key Design Decisions & Trade-offs

1. **Why local HuggingFace embeddings instead of API embeddings?**
   * Using `all-MiniLM-L6-v2` locally avoids extra API roundtrips and billing during vector generation while maintaining high accuracy for English news content.

2. **Why FAISS in-memory instead of a managed vector database (Pinecone/Qdrant)?**
   * For short-lived sessions where users analyze a handful of news articles on demand, an in-memory vector store eliminates unnecessary database setup, latency, and operational cost.

3. **Chunk Size (`1000`) & Overlap (`200`):**
   * News articles typically express complete thoughts within 2-3 sentences. A 1000-character window captures enough context for the embedding model, while a 200-character overlap prevents key semantic entities from getting severed between splits.

---

## 🛡️ License

MIT License. Feel free to clone, modify, and build upon this project.