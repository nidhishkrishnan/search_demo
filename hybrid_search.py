from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from data import documents

# 1. Initialize Embeddings and Vector Store (Semantic)
embeddings = OllamaEmbeddings(
    model="llama3.2:1b",
    base_url="http://localhost:11434"
)

docs = [Document(page_content=d) for d in documents]

print("Creating vector store (this may take a moment)...")
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="hybrid_search_demo"
)
chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 2. Initialize BM25 Retriever (Lexical)
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 3

# 3. Initialize Ensemble Retriever (Hybrid)
# weights: [bm25_weight, chroma_weight]
# Usually 0.5, 0.5 is a good starting point.
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever],
    weights=[0.5, 0.5]
)

# 4. Initialize LLM
llm = ChatOllama(
    model="llama3.2:1b",
    base_url="http://localhost:11434"
)

def retrieve_documents(query):
    return ensemble_retriever.invoke(query)

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
    print("--- Hybrid Search Demo (BM25 + Chroma) ---")
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
