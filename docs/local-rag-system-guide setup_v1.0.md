# Building a Local RAG System with Ollama, LangChain, Chroma & Obsidian
## A Complete Guide to Persistent AI Memory for Knowledge Management

---

## 📋 Overview

This guide walks you through building a **Local Retrieval-Augmented Generation (RAG) system** that integrates your knowledge management platform (like Obsidian) with a local AI model (Ollama), enabling persistent memory and semantic search across your personal notes and documentation.

The system architecture combines:
- **Ollama** - Local LLM inference for privacy-first AI
- **LangChain** - Orchestration framework for LLM workflows
- **Chroma** - Vector database for semantic storage and retrieval
- **Obsidian** - Your knowledge base with REST API integration
- **Python** - Automation and integration layer

### What This System Does

✅ **Day 1**: Write your project notes in Obsidian → System indexes and vectorizes everything  
✅ **Day 2**: Ask "What did we build yesterday?" → System retrieves exact context with commands, IPs, errors  
✅ **Ongoing**: Daily updates automatically augment your memory  
✅ **Advantage**: Everything stays local, encrypted, under your control

---

## 🎯 System Architecture

### Data Flow Diagram

```
DAY 1: Content Creation & Storage
┌─────────────────────────────────────┐
│  Write Documentation in Obsidian    │
│  (Technical notes, commands, IPs)   │
└──────────────┬──────────────────────┘
               │
               │ REST API (port 27123)
               ▼
┌─────────────────────────────────────┐
│  LangChain Document Loader          │
│  (Reads markdown files)             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Ollama Embeddings Service          │
│  (Converts text to vectors)         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Chroma Vector Database             │
│  (Persistent local storage)         │
└─────────────────────────────────────┘

════════════════════════════════════════════════════════════════

DAY 2+: Context Retrieval & Generation
┌─────────────────────────────────────┐
│  User Query / Question              │
│  "What commands did we run?"       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Semantic Search                    │
│  (Query vectorization)              │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Chroma Similarity Retrieval        │
│  (Find top 5 relevant chunks)       │
└──────────────┬──────────────────────┘
               │
               │ Context + Query
               ▼
┌─────────────────────────────────────┐
│  Ollama LLM Response Generation     │
│  (Augmented with context)           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Contextual Answer                  │
│  "You ran: docker run -d nginx..."  │
└─────────────────────────────────────┘
```

---

## 🔧 Prerequisites & Setup

### Prerequisites You Need

- ✅ **Ollama 3.0+** running locally (default: localhost:11434)
- ✅ **Python 3.8+** installed on your system
- ✅ **Obsidian** with Local REST API plugin enabled (port 27123)
- ✅ **8GB+ RAM** for comfortable operation with 8B model
- ✅ **Docker** (optional, only if running Chroma in container)

### Step 1: Create Virtual Environment

**Windows (PowerShell/CMD):**
```bash
# Create virtual environment
python -m venv rag-env

# Activate it
rag-env\Scripts\activate
```

**Linux/macOS/WSL:**
```bash
# Create virtual environment
python3 -m venv rag-env

# Activate it
source rag-env/bin/activate
```

### Step 2: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install core packages
pip install langchain langchain-chroma langchain-ollama chromadb
pip install langchain-core langchain-community
pip install requests python-dotenv

# Optional: for better performance
pip install numpy pandas
```

### Step 3: Project Structure

Create this directory structure:

```
local-rag-system/
│
├── rag-env/                    # Virtual environment
├── chroma_data/                # Vector database storage (auto-created)
├── .env                        # Configuration file
├── requirements.txt            # Dependencies list
│
├── obsidian_loader.py          # Module to read Obsidian notes
├── rag_setup.py                # Initial indexing script
├── rag_main.py                 # Main RAG application
└── add_to_memory.py            # Daily update script
```

---

## ⚙️ Configuration

### Step 1: Create `.env` File

Create a `.env` file in your project root with:

```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.1:8b
EMBEDDING_MODEL=nomic-embed-text

# Obsidian REST API
OBSIDIAN_API_URL=http://localhost:27123
OBSIDIAN_API_KEY=your-api-key-here
OBSIDIAN_VAULT_NAME=your-vault-name

# Chroma Vector Database
CHROMA_PATH=./chroma_data
COLLECTION_NAME=project-memory
```

**How to get OBSIDIAN_API_KEY:**
1. Open Obsidian → Settings → Community Plugins
2. Find "Local REST API" plugin
3. Copy the API key
4. Paste into `.env`

### Step 2: Create `requirements.txt`

```
langchain>=0.1.0
langchain-chroma>=0.1.0
langchain-ollama>=0.1.0
langchain-core>=0.1.0
langchain-community>=0.1.0
chromadb>=0.4.0
requests>=2.31.0
python-dotenv>=1.0.0
```

---

## 📝 Implementation

### Module 1: `obsidian_loader.py` - Reading Your Notes

This module connects to Obsidian via REST API and loads your markdown files:

```python
import requests
from langchain.schema import Document
from dotenv import load_dotenv
import os

load_dotenv()

class ObsidianDocumentLoader:
    """Load documents from Obsidian vault via REST API"""
    
    def __init__(self):
        self.api_url = os.getenv("OBSIDIAN_API_URL")
        self.api_key = os.getenv("OBSIDIAN_API_KEY")
        self.vault_name = os.getenv("OBSIDIAN_VAULT_NAME")
    
    def load_from_path(self, folder_path: str) -> list:
        """Load markdown files from specific folder"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            # Get all files from vault
            response = requests.get(
                f"{self.api_url}/vault",
                headers=headers
            )
            
            if response.status_code != 200:
                print(f"❌ Error connecting to Obsidian: {response.status_code}")
                return []
            
            documents = []
            files = response.json()
            
            # Filter and load markdown files from folder
            for file in files:
                if file['path'].startswith(folder_path) and file['path'].endswith('.md'):
                    try:
                        file_response = requests.get(
                            f"{self.api_url}/vault/read?file={file['path']}",
                            headers=headers
                        )
                        
                        if file_response.status_code == 200:
                            content = file_response.json()['content']
                            
                            doc = Document(
                                page_content=content,
                                metadata={
                                    "source": file['path'],
                                    "vault": self.vault_name,
                                    "type": "documentation"
                                }
                            )
                            documents.append(doc)
                            print(f"✅ Loaded: {file['path']}")
                    
                    except Exception as e:
                        print(f"⚠️ Error loading {file['path']}: {e}")
            
            return documents
        
        except Exception as e:
            print(f"❌ Error: {e}")
            return []
    
    def load_all_vault(self) -> list:
        """Load all markdown files from entire vault"""
        return self.load_from_path("")
```

### Module 2: `rag_setup.py` - Initial Indexing

Run this once to create your vector database from existing notes:

```python
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
    
    # Load documents from Obsidian
    print("📂 Loading documents from Obsidian...")
    loader = ObsidianDocumentLoader()
    documents = loader.load_all_vault()
    
    if not documents:
        print("❌ No documents found!")
        return None
    
    print(f"✅ Loaded {len(documents)} documents\n")
    
    # Split into chunks for better indexing
    print("✂️ Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks\n")
    
    # Create embeddings with local Ollama
    print("🧠 Initializing Ollama embeddings...")
    embeddings = OllamaEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    print("✅ Embeddings initialized\n")
    
    # Store in Chroma vector database
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
```

### Module 3: `rag_main.py` - Query Your Memory

This is the main interface to ask questions about your work:

```python
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

def initialize_rag_system():
    """Initialize RAG components for querying"""
    
    print("🔄 Initializing RAG system...\n")
    
    # Initialize embeddings
    embeddings = OllamaEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    
    # Load vector store
    vector_store = Chroma(
        persist_directory=os.getenv("CHROMA_PATH", "./chroma_data"),
        embedding_function=embeddings,
        collection_name=os.getenv("COLLECTION_NAME", "project-memory")
    )
    
    # Initialize LLM
    llm = ChatOllama(
        model=os.getenv("LLM_MODEL", "llama3.1:8b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.1  # Low temperature for consistent answers
    )
    
    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Return top 5 relevant results
    )
    
    print("✅ RAG system initialized!\n")
    
    return llm, retriever

def create_rag_chain(llm, retriever):
    """Create RAG chain with system prompt"""
    
    template = """You are a technical assistant with access to project memory.
Based on the following context from previous work sessions, answer the question.

Context:
{context}

Question: {question}

Answer with specific details, commands, configurations, and recommendations when relevant:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    
    return chain

def main():
    """Interactive RAG query loop"""
    
    print("=" * 60)
    print("🚀 Local RAG System - Technical Memory Assistant")
    print("=" * 60)
    print("Type 'quit' to exit\n")
    
    # Initialize
    llm, retriever = initialize_rag_system()
    chain = create_rag_chain(llm, retriever)
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        print("\n🤔 Searching memory and thinking...\n")
        
        try:
            response = chain.invoke(user_input)
            print(f"Assistant: {response.content}\n")
            
        except Exception as e:
            print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    main()
```

### Module 4: `add_to_memory.py` - Daily Updates

Run this daily to add new work to your memory:

```python
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
    
    # Load today's notes (customize path as needed)
    today_date = datetime.now().strftime("%Y-%m-%d")
    documents = loader.load_from_path(f"Daily/{today_date}")
    
    if not documents:
        print("⚠️ No new documents found for today")
        return
    
    # Split new documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    
    # Initialize embeddings
    embeddings = OllamaEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    
    # Load existing vector store
    vector_store = Chroma(
        persist_directory=os.getenv("CHROMA_PATH", "./chroma_data"),
        embedding_function=embeddings,
        collection_name=os.getenv("COLLECTION_NAME", "project-memory")
    )
    
    # Add new documents
    vector_store.add_documents(chunks)
    
    print(f"✅ Added {len(chunks)} new chunks to memory!")
    print("💾 Memory updated successfully!\n")

if __name__ == "__main__":
    add_new_documents_to_memory()
```

---

## 🎯 Daily Workflow

### First Time Setup

```bash
# 1. Activate virtual environment
source rag-env/bin/activate  # Linux/macOS
# or
rag-env\Scripts\activate  # Windows

# 2. Make sure Ollama is running
ollama serve  # In a separate terminal

# 3. Initialize your vector database
python rag_setup.py

# 4. Start querying!
python rag_main.py
```

### Day-to-Day Usage

**Morning:**
```bash
# Write new notes in Obsidian
# Save your work documentation, technical decisions, commands

# Activate and run RAG
source rag-env/bin/activate
python rag_main.py

# Example queries:
# "What database schema did we design?"
# "Show me the Docker commands for deployment"
# "What were the performance issues we identified?"
```

**Evening:**
```bash
# Update memory with today's work
python add_to_memory.py

# This ensures tomorrow you have full context
```

---

## ✅ Verification Checklist

Before you start, verify everything is working:

- [ ] Ollama running: `curl http://localhost:11434/api/tags` returns model list
- [ ] Obsidian REST API enabled in settings
- [ ] Python virtual environment created and activated
- [ ] All pip packages installed without errors
- [ ] `.env` file configured with correct paths and API keys
- [ ] Can import required modules: `python -c "from langchain_chroma import Chroma; print('OK')"`

---

## 🚀 Advanced: Scaling Up

### Option 1: Larger Model
```bash
# For better quality responses (requires more RAM)
ollama pull mistral      # 7B model, faster
ollama pull neural-chat  # 7B model, good for chat

# Update .env:
LLM_MODEL=mistral
```

### Option 2: Docker Chroma
```bash
# Run Chroma as separate containerized service
docker run -p 8000:8000 chromadb/chroma:latest

# Update .env:
CHROMA_HOST=localhost
CHROMA_PORT=8000
```

### Option 3: Web UI
```bash
# Add Gradio for web interface
pip install gradio

# Create app.py with Gradio interface
# Users can access via browser instead of CLI
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| **Connection refused to localhost:11434** | Ensure Ollama is running: `ollama serve` in terminal |
| **Obsidian API not responding** | Check Local REST API plugin is enabled in Obsidian settings |
| **Slow embeddings** | Normal for local models; nomic-embed-text ~2-5 sec per chunk |
| **Chroma folder not created** | Verify write permissions to directory |
| **Low quality answers** | Try larger model (mistral) or increase context with more relevant chunks |
| **Out of memory errors** | Reduce chunk_size in splitter or use smaller model |

---

## 📊 Behind the Scenes

### How Semantic Search Works

```
Day 1: Documentation
─────────────────
Your note: "Deployed app to Azure VM (IP: 20.1.2.3), used Docker with Kubernetes"
                    │
                    ▼
Split into chunks: ["Deployed to Azure", "VM IP: 20.1.2.3", "Docker Kubernetes"]
                    │
                    ▼
Convert to vectors: [0.234, -0.567, 0.891, ...], [0.445, 0.123, ...], ...
                    │
                    ▼
Store in Chroma: {chunk, vector, metadata}

Day 2: Memory Retrieval
──────────────────────
Your query: "Where did we deploy the application?"
                    │
                    ▼
Convert to vector: [0.223, -0.551, 0.885, ...]  (similar to Day 1)
                    │
                    ▼
Chroma similarity search: Finds matching vectors
                    │
                    ▼
Retrieved: "Deployed to Azure", "VM IP: 20.1.2.3"
                    │
                    ▼
Ollama reads context + generates: "You deployed to Azure VM (IP: 20.1.2.3) using Docker..."
```

---

## 🔗 Resources & References

- **LangChain Documentation**: https://python.langchain.com
- **Chroma Vector DB**: https://docs.trychroma.com/
- **Ollama API**: https://github.com/ollama/ollama/blob/main/docs/api.md
- **Obsidian REST API**: https://github.com/coddingtonbear/obsidian-local-rest-api

---

## 🎯 Next Steps

1. Complete initial setup following this guide
2. Run `rag_setup.py` to index existing knowledge
3. Test with `rag_main.py` and sample queries
4. Establish daily workflow with `add_to_memory.py`
5. Experiment with different models and chunk sizes
6. Consider scaling options as your knowledge base grows
7. Integrate with your development workflow and tools

---

## 💡 Key Takeaways

✨ **Local-First**: Everything stays on your computer, fully private  
✨ **Always Available**: No API costs, no rate limits, no internet dependency  
✨ **Semantic Search**: Find information by meaning, not keywords  
✨ **Persistent Memory**: Your system learns and improves over time  
✨ **Contextual AI**: Answers grounded in your actual work and decisions  

**Build once, benefit forever. Your knowledge deserves a long-term home.**

---

*Last Updated: November 2025*  
*For questions and updates, refer to official documentation of respective tools*
