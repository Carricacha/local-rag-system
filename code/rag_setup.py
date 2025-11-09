@'
"""
RAG Setup - Initial Indexing
Creates vector database from existing Obsidian notes
"""

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from obsidian_loader import ObsidianDocumentLoader
from dotenv import load_dotenv
import os

load_dotenv()

def setup_vector_database():
    """Initialize RAG system with existing documents"""
    
    print("🚀 Starting RAG Setup...\n")
    
    print("📂 Loading documents from Obsidian...")
    loader = ObsidianDocumentLoader()
    documents = loader.load_all_vault()
    
    if not documents:
        print("❌ No documents found!")
        return None
    
    print(f"✅ Loaded {len(documents)} documents\n")
    
    print("✂️ Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks\n")
    
    print("🧠 Initializing Ollama embeddings...")
    embeddings = OllamaEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    print("✅ Embeddings initialized\n")
    
    print("💾 Creating Chroma vector store...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=os.getenv("CHROMA_PATH", "./chroma_data"),
        collection_name=os.getenv("COLLECTION_NAME", "project-memory")
    )
    
    print(f"✅ Vector store created with {len(chunks)} embeddings!")
    print(f"📦 Saved to: {os.getenv('CHROMA_PATH')}\n")
    
    return vector_store

if __name__ == "__main__":
    db = setup_vector_database()
    if db:
        print("🎉 Setup complete! Now run rag_main.py")
'@ | Out-File -FilePath rag_setup.py -Encoding UTF8
