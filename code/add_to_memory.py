@'
"""
Add to Memory - Daily Updates
Adds new documents to existing memory
"""

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from obsidian_loader import ObsidianDocumentLoader
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()

def add_new_documents_to_memory():
    """Add today's new documents to existing memory"""
    
    print("📝 Adding new documents to memory...\n")
    
    loader = ObsidianDocumentLoader()
    
    today_date = datetime.now().strftime("%Y-%m-%d")
    documents = loader.load_from_path(f"Daily/{today_date}")
    
    if not documents:
        print("⚠️ No new documents found for today")
        return
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    
    embeddings = OllamaEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    
    vector_store = Chroma(
        persist_directory=os.getenv("CHROMA_PATH", "./chroma_data"),
        embedding_function=embeddings,
        collection_name=os.getenv("COLLECTION_NAME", "project-memory")
    )
    
    vector_store.add_documents(chunks)
    
    print(f"✅ Added {len(chunks)} new chunks to memory!")
    print("💾 Memory updated successfully!\n")

if __name__ == "__main__":
    add_new_documents_to_memory()
'@ | Out-File -FilePath add_to_memory.py -Encoding UTF8
