from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Dummy data
DUMMY_DOCS = [
    "LangChain is a framework for building context-aware reasoning applications.",
    "It integrates LLMs with tools, memory, and external data sources.",
    "RAG combines retrieval from vector stores with LLM generation.",
    "Vector stores use embeddings for semantic similarity search.",
    "Google Gemini is a multimodal LLM from Google.",
    "Qdrant is an open-source vector database for similarity search.",
    "Chat history maintains conversation context across multiple turns.",
    "Cosine similarity powers semantic search in vector databases.",
]

docs = [Document(page_content=text, metadata={"source": "dummy-data"}) for text in DUMMY_DOCS]

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vector_store = QdrantVectorStore.from_documents(
    docs,
    embeddings,
    url="http://localhost:6333",
    collection_name="demo_rag",
)
print("Data added to vector store successfully.")