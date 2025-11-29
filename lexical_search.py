from rank_bm25 import BM25Okapi
from langchain_ollama import ChatOllama
from data import documents
import string

# 1. Prepare Data for BM25
# BM25 requires tokenized documents (list of lists of strings)
tokenized_docs = [doc.lower().translate(str.maketrans('', '', string.punctuation)).split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# 2. Initialize LLM
llm = ChatOllama(
    model="llama3.2:1b",
    base_url="http://localhost:11434"
)

def retrieve_documents(query, k=3):
    tokenized_query = query.lower().translate(str.maketrans('', '', string.punctuation)).split()
    return bm25.get_top_n(tokenized_query, documents, n=k)

def generate_answer(query, context_docs):
    context_text = "\n\n".join(context_docs)
    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the following context:

Context:
{context_text}

Question: {query}

Answer:"""
    
    response = llm.invoke(prompt)
    return response.content

def main():
    print("--- Lexical Search Demo (BM25) ---")
    while True:
        query = input("\nEnter your question (or 'q' to quit): ")
        if query.lower() == 'q':
            break
        
        print(f"Searching for: {query}")
        retrieved_docs = retrieve_documents(query)
        
        print("\nRetrieved Documents:")
        for i, doc in enumerate(retrieved_docs):
            print(f"{i+1}. {doc}")
            
        print("\nGenerating Answer...")
        answer = generate_answer(query, retrieved_docs)
        print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()
