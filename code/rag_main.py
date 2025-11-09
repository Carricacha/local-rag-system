@'
"""
RAG Main - Query Interface
Interactive system to query your knowledge base
"""

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
    
    embeddings = OllamaEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    
    vector_store = Chroma(
        persist_directory=os.getenv("CHROMA_PATH", "./chroma_data"),
        embedding_function=embeddings,
        collection_name=os.getenv("COLLECTION_NAME", "project-memory")
    )
    
    llm = ChatOllama(
        model=os.getenv("LLM_MODEL", "llama3.1:8b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.1
    )
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
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
    
    llm, retriever = initialize_rag_system()
    chain = create_rag_chain(llm, retriever)
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == "quit":
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
'@ | Out-File -FilePath rag_main.py -Encoding UTF8
