# RAG Implementation with LangChain

## Overview

This document provides practical implementation examples of RAG systems using LangChain and Python.

## Prerequisites

```bash
pip install langchain faiss-cpu openai
```

## Basic RAG Implementation

### Complete Multi-Agent System Example

Below is a sample Python code using LangChain to implement the multi-agent flow. It uses OpenAI's GPT-3.5-turbo (requires API key), FAISS for vector storage, and mock data for demonstration.

```python
import os
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"

# Initialize LLM
llm = OpenAI(temperature=0.7)

# Mock product data for Vector DB
product_docs = [
    Document(page_content="Laptop X: High-performance laptop with 16GB RAM, 512GB SSD, Intel i7 processor. Price: $1200."),
    Document(page_content="FAQ: How to return a product? Contact support within 30 days with receipt."),
    Document(page_content="Headphones Y: Wireless noise-canceling headphones with 20-hour battery. Price: $150.")
]

# Create Vector DB
embeddings = OpenAIEmbeddings()
vector_db = FAISS.from_documents(product_docs, embeddings)

# RAG Chain for Product Info
product_rag = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vector_db.as_retriever()
)

# Mock Order Database
orders_db = {
    "12345": "Order status: Shipped on 2025-11-10, expected delivery 2025-11-15.",
    "67890": "Order status: Processing, will ship within 2 days."
}

def check_order_status(order_id: str) -> str:
    return orders_db.get(order_id, "Order not found.")

# Tools for Agents
product_tool = Tool(
    name="ProductInfo",
    description="Retrieve product details or FAQs from knowledge base.",
    func=product_rag.run
)

order_tool = Tool(
    name="OrderStatus",
    description="Check order status by order ID.",
    func=check_order_status
)

# Orchestrator Agent
orchestrator = initialize_agent(
    tools=[product_tool, order_tool],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# Example Usage
if __name__ == "__main__":
    query1 = "Tell me about Laptop X."
    print("Query:", query1)
    print("Response:", orchestrator.run(query1))
    
    query2 = "What's the status of order 12345?"
    print("\nQuery:", query2)
    print("Response:", orchestrator.run(query2))
```

### Code Explanation

- **orchestrator**: Uses LangChain's tool-based system to route queries
- **product_tool**: Wraps the RAG chain for retrieval-augmented responses
- **order_tool**: Handles order lookups from the mock database
- **Agent Decision Making**: The agent analyzes the query and decides which tool to invoke based on the query content

## Handling Multiple Retrieved Chunks

### Problem: Context Window Limits

When Top-K retrieves more chunks than the LLM can process, you need to summarize or condense the information.

### Solution: Map-Reduce Summarization

```python
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA

# ... (previous setup for llm, vector_db) ...

# Summarization chain for condensing multiple documents
summarize_chain = load_summarize_chain(llm, chain_type="map_reduce")

# RAG with summarization for multiple chunks
product_rag_summarized = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="map_reduce",  # Summarizes retrieved docs
    retriever=vector_db.as_retriever(search_kwargs={"k": 10})  # Higher K
)

# Example: Retrieve and summarize top 10 chunks
query = "Summarize product features for laptops."
retrieved_docs = vector_db.similarity_search(query, k=10)
summary = summarize_chain.run(retrieved_docs)
print("Summary:", summary)
```

#### How Map-Reduce Works

1. **Map Phase**: Splits documents and summarizes each independently
2. **Reduce Phase**: Combines individual summaries into a final summary
3. **Benefit**: Allows retrieving more chunks (e.g., K=10) while staying within context limits

This approach improves coverage without exceeding the LLM's context window.

## Advanced RAG Patterns

### 1. Custom Retriever with Filtering

```python
from langchain.vectorstores import FAISS
from langchain.schema import Document

# Add metadata to documents
product_docs_with_metadata = [
    Document(
        page_content="Laptop X: High-performance laptop...",
        metadata={"category": "laptops", "price": 1200, "in_stock": True}
    ),
    Document(
        page_content="Headphones Y: Wireless noise-canceling...",
        metadata={"category": "audio", "price": 150, "in_stock": True}
    )
]

vector_db = FAISS.from_documents(product_docs_with_metadata, embeddings)

# Retrieve with filtering
retriever = vector_db.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"category": "laptops"}  # Only retrieve laptops
    }
)
```

### 2. Hybrid Search (Semantic + Keyword)

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# Semantic retriever (vector-based)
semantic_retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# Keyword retriever (BM25)
bm25_retriever = BM25Retriever.from_documents(product_docs)
bm25_retriever.k = 3

# Combine both retrievers
hybrid_retriever = EnsembleRetriever(
    retrievers=[semantic_retriever, bm25_retriever],
    weights=[0.5, 0.5]  # Equal weight to both methods
)

# Use hybrid retriever in RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=hybrid_retriever
)
```

### 3. Reranking Retrieved Results

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Base retriever
base_retriever = vector_db.as_retriever(search_kwargs={"k": 10})

# Compressor for reranking
compressor = LLMChainExtractor.from_llm(llm)

# Compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Use in RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=compression_retriever
)
```

### 4. Custom Prompt for RAG

```python
from langchain.prompts import PromptTemplate

# Custom prompt template
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always cite the source of your information.

Context: {context}

Question: {question}

Helpful Answer with citations:"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

# Use custom prompt in RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)
```

## Performance Optimization

### 1. Caching Embeddings

```python
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

# Create cache store
cache_store = LocalFileStore("./embedding_cache")

# Cached embeddings
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=OpenAIEmbeddings(),
    document_embedding_cache=cache_store,
    namespace="product_embeddings"
)

# Use cached embeddings
vector_db = FAISS.from_documents(product_docs, cached_embeddings)
```

### 2. Batch Processing

```python
# Process documents in batches
batch_size = 100
all_docs = load_all_documents()  # Your document loading function

for i in range(0, len(all_docs), batch_size):
    batch = all_docs[i:i+batch_size]
    
    if i == 0:
        vector_db = FAISS.from_documents(batch, embeddings)
    else:
        vector_db.add_documents(batch)

# Save to disk
vector_db.save_local("./vector_store")

# Load from disk later
vector_db = FAISS.load_local("./vector_store", embeddings)
```

### 3. Async Operations

```python
import asyncio
from langchain.chains import RetrievalQA

async def query_rag_async(question: str):
    """Async RAG query"""
    result = await rag_chain.arun(question)
    return result

async def main():
    questions = [
        "What is Laptop X?",
        "Tell me about Headphones Y",
        "How do I return a product?"
    ]
    
    # Run queries concurrently
    results = await asyncio.gather(*[
        query_rag_async(q) for q in questions
    ])
    
    for q, r in zip(questions, results):
        print(f"Q: {q}\nA: {r}\n")

asyncio.run(main())
```

## Testing and Evaluation

### 1. Test RAG Retrieval Quality

```python
def evaluate_retrieval(query: str, expected_docs: list, k: int = 3):
    """Evaluate if expected documents are retrieved"""
    retrieved = vector_db.similarity_search(query, k=k)
    retrieved_contents = [doc.page_content for doc in retrieved]
    
    matches = sum(1 for expected in expected_docs 
                  if any(expected in content for content in retrieved_contents))
    
    precision = matches / k
    recall = matches / len(expected_docs)
    
    print(f"Query: {query}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Retrieved: {retrieved_contents}\n")

# Test
evaluate_retrieval(
    query="high performance laptop",
    expected_docs=["Laptop X"]
)
```

### 2. Test End-to-End RAG

```python
def test_rag_chain():
    test_cases = [
        {
            "query": "What are the specs of Laptop X?",
            "expected_keywords": ["16GB RAM", "512GB SSD", "Intel i7"]
        },
        {
            "query": "How do I return a product?",
            "expected_keywords": ["Contact support", "30 days", "receipt"]
        }
    ]
    
    for test in test_cases:
        result = rag_chain.run(test["query"])
        print(f"Query: {test['query']}")
        print(f"Result: {result}")
        
        # Check if expected keywords are in result
        found = [kw for kw in test["expected_keywords"] if kw in result]
        print(f"Found keywords: {found}")
        print(f"Pass: {len(found) == len(test['expected_keywords'])}\n")

test_rag_chain()
```

## Common Issues and Solutions

### Issue 1: Empty Retrieval Results
```python
# Check vector store has documents
print(f"Vector store has {vector_db.index.ntotal} documents")

# Lower similarity threshold
retriever = vector_db.as_retriever(
    search_kwargs={
        "k": 5,
        "score_threshold": 0.3  # Lower threshold
    }
)
```

### Issue 2: Slow Queries
```python
# Use faster search with approximate nearest neighbors
from langchain.vectorstores import FAISS

vector_db = FAISS.from_documents(
    product_docs, 
    embeddings,
    distance_strategy="COSINE"  # Faster than EUCLIDEAN
)
```

### Issue 3: Context Window Exceeded
```python
# Use map_reduce or refine chain type
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="map_reduce",  # or "refine"
    retriever=vector_db.as_retriever()
)
```

## Next Steps

- See **RAG_Best_Practices.md** for optimization strategies
- Check **RAG_Multi_Agent_System.md** for architecture patterns
- Review **RAG_Fundamentals.md** for conceptual understanding
