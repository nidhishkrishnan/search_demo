from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from data import documents

# 1. Initialize Embeddings
# We use the same model for embeddings, or a specific embedding model if available.
# llama3.2:1b can be used for embeddings, though dedicated models like 'nomic-embed-text' are better.
# For this demo, we'll stick to llama3.2:1b for simplicity, or assume it handles embeddings.
embeddings = OllamaEmbeddings(
    model="llama3.2:1b",
    base_url="http://localhost:11434"
)

# 2. Create Vector Store
# We convert our list of strings into Document objects
docs = [Document(page_content=d) for d in documents]

print("Creating vector store (this may take a moment)...")
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="semantic_search_demo"
)

# 3. Initialize LLM
llm = ChatOllama(
    model="llama3.2:1b",
    base_url="http://localhost:11434"
)

def retrieve_documents(query, k=3):
    return vectorstore.similarity_search(query, k=k)

def generate_answer(query, context_docs):
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the following context:

Context:
{context_text}

Question: {query}

Answer:"""
    
    response = llm.invoke(prompt)
    return response.content

def main():
    print("--- Semantic Search Demo (Chroma + Embeddings) ---")
    while True:
        query = input("\nEnter your question (or 'q' to quit): ")
        if query.lower() == 'q':
            break
        
        print(f"Searching for: {query}")
        retrieved_docs = retrieve_documents(query)
        
        print("\nRetrieved Documents:")
        for i, doc in enumerate(retrieved_docs):
            print(f"{i+1}. {doc.page_content}")
            
        print("\nGenerating Answer...")
        answer = generate_answer(query, retrieved_docs)
        print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()
