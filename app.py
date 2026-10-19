import os
# Set User-Agent to satisfy web loaders and remove warning
os.environ["USER_AGENT"] = "NewsPulseBot/1.0"

import streamlit as st
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Page Setup
st.set_page_config(page_title="NewsPulse RAG | AI Research Agent", page_icon="📰", layout="wide")
st.title("📰 NewsPulse: Multi-Article AI Research Agent")
st.caption("Retrieval-Augmented Generation (RAG) powered by LangChain, FAISS & NVIDIA Nemotron")

# Sidebar for Setup & Article Input
st.sidebar.header("⚙️ Configuration & Sources")
api_key = os.getenv("NVIDIA_NIM_API_KEY") or st.sidebar.text_input("NVIDIA API Key", type="password")

st.sidebar.subheader("🔗 Add News Article URLs")
# Allow user to input up to 3 news article URLs dynamically
url1 = st.sidebar.text_input("URL 1", value="https://techcrunch.com/2026/01/15/ai-agents-enterprise-automation/")
url2 = st.sidebar.text_input("URL 2", value="")
url3 = st.sidebar.text_input("URL 3", value="")

urls = [u.strip() for u in [url1, url2, url3] if u.strip()]

process_btn = st.sidebar.button("🚀 Process & Index Articles", type="primary", use_container_width=True)

# -------------------------------------------------------------------
# 1. Embeddings & Vector Store Setup
# -------------------------------------------------------------------
@st.cache_resource
def load_embeddings():
    # HuggingFace MiniLM runs 100% locally on Mac CPU (Free, light, and fast)
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embedding_model = load_embeddings()

# Initialize session state for vectorstore
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# -------------------------------------------------------------------
# 2. Ingestion Pipeline (Scrape -> Chunk -> Embed -> FAISS Index)
# -------------------------------------------------------------------
if process_btn:
    if not urls:
        st.sidebar.error("Please enter at least one URL.")
    else:
        with st.spinner("Scraping articles, chunking text, and building FAISS vector index..."):
            try:
                # Step A: Load Web Articles
                loader = WebBaseLoader(urls)
                docs = loader.load()

                # Step B: Recursive Character Splitting
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", " ", ""]
                )
                split_docs = text_splitter.split_documents(docs)

                # Step C: Index into FAISS Vector Store
                vectorstore = FAISS.from_documents(split_docs, embedding_model)
                st.session_state.vectorstore = vectorstore

                st.sidebar.success(f"✅ Indexed {len(urls)} articles into {len(split_docs)} vector chunks!")

            except Exception as e:
                st.sidebar.error(f"Failed to process URLs: {str(e)}")

# -------------------------------------------------------------------
# 3. RAG Q&A Execution Engine
# -------------------------------------------------------------------
st.subheader("💬 Ask Questions Across Your Articles")

query = st.text_input("Enter your question (e.g., 'What are the main takeaways from these articles?'):")

if query:
    if not api_key:
        st.warning("⚠️ Please provide an NVIDIA API Key in the sidebar or `.env` file.")
        st.stop()
        
    if st.session_state.vectorstore is None:
        st.warning("⚠️ Please process at least one news article in the sidebar before asking questions.")
        st.stop()

    with st.spinner("Retrieving relevant context & generating response..."):
        try:
            # Step A: Create Retriever
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})

            # Step B: Retrieve Top Matching Document Chunks
            retrieved_docs = retriever.invoke(query)

            # Step C: Format Context with Metadata for Citation
            context_text = "\n\n".join([
                f"\n{doc.page_content}"
                for doc.index, doc in enumerate(retrieved_docs)
            ])

            # Step D: Setup Prompt and LLM
            prompt = ChatPromptTemplate.from_template(
                "You are an expert investigative journalist and analyst. "
                "Answer the user's question accurately using ONLY the provided news article context below. "
                "If the answer cannot be determined from the context, state that clearly.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer (include source URL references where relevant):"
            )

            llm = ChatOpenAI(
                model="nvidia/llama-3.1-nemotron-70b-instruct",
                api_key=api_key,
                base_url="https://integrate.api.nvidia.com/v1",
                temperature=0.1
            )

            # Step E: Execute RAG Chain
            chain = prompt | llm | StrOutputParser()
            response = chain.invoke({"context": context_text, "question": query})

            # Display Results
            st.markdown("### 💡 AI Answer")
            st.write(response)

            st.markdown("---")
            with st.expander("🔍 View Retrieved Context Chunks & Metadata"):
                for idx, doc in enumerate(retrieved_docs):
                    st.markdown(f"**Chunk {idx+1} Source:** `{doc.metadata.get('source')}`")
                    st.caption(doc.page_content)
                    st.divider()

        except Exception as e:
            st.error(f"Error executing RAG chain: {str(e)}")