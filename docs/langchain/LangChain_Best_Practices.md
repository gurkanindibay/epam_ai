# LangChain Best Practices and Use Cases

## Development Best Practices

### 1. Start Simple, Then Optimize
```python
# Start with basic chain
simple_chain = prompt | llm

# Add complexity gradually
enhanced_chain = (
    prompt 
    | llm 
    | output_parser 
    | RunnableLambda(post_process)
)
```

### 2. Use Environment Variables for Secrets
```python
import os
from langchain.llms import OpenAI

# Never hardcode API keys
llm = OpenAI(
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    temperature=0.7
)
```

### 3. Implement Proper Error Handling
```python
from langchain.schema.runnable import RunnablePassthrough

def safe_chain_invoke(chain, input_data):
    try:
        return chain.invoke(input_data)
    except Exception as e:
        return f"Error: {str(e)}"

# Wrap chains for production use
production_chain = RunnableLambda(
    lambda x: safe_chain_invoke(original_chain, x)
)
```

### 4. Use Caching for Repeated Queries
```python
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache

# Set up persistent caching
set_llm_cache(SQLiteCache(database_path=".langchain.db"))
```

### 5. Monitor Token Usage and Costs
```python
from langchain.callbacks import get_openai_callback

def track_usage(chain, input_data):
    with get_openai_callback() as cb:
        result = chain.invoke(input_data)
        print(f"Tokens used: {cb.total_tokens}, Cost: ${cb.total_cost:.4f}")
        return result
```

## Production Deployment Best Practices

### 1. Async for Better Performance
```python
import asyncio
from typing import List, Dict

async def process_batch(chain, inputs: List[Dict]):
    tasks = [chain.ainvoke(input_data) for input_data in inputs]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

### 2. Implement Rate Limiting
```python
import time
from functools import wraps

def rate_limit(calls_per_second=1):
    def decorator(func):
        last_called = [0.0]
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = 1.0 / calls_per_second - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

@rate_limit(calls_per_second=0.5)  # Max 1 call per 2 seconds
def call_llm(chain, input_data):
    return chain.invoke(input_data)
```

### 3. Use Connection Pooling for Vector Stores
```python
from langchain.vectorstores import Pinecone
import pinecone

# Initialize with connection pooling
pinecone.init(
    api_key="your-key",
    environment="your-env",
    pool_threads=30  # Adjust based on needs
)
```

### 4. Implement Health Checks
```python
def health_check(chain):
    try:
        test_result = chain.invoke({"test": "health check"})
        return {"status": "healthy", "response_time": "normal"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

### 5. Use Structured Logging
```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def log_chain_execution(self, chain_name, input_data, output, execution_time):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "chain_name": chain_name,
            "input_hash": hash(str(input_data)),
            "output_length": len(str(output)),
            "execution_time": execution_time,
            "status": "success"
        }
        self.logger.info(json.dumps(log_entry))
```

## Common Use Cases and Patterns

### 1. Question-Answering System
```python
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

def create_qa_system(documents):
    # Process documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    
    # Create vector store
    vectorstore = Chroma.from_documents(texts, OpenAIEmbeddings())
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4})
    )
    
    return qa_chain
```

### 2. Document Summarization Pipeline
```python
from langchain.chains.summarize import load_summarize_chain

def create_summarization_pipeline():
    # For long documents, use map-reduce
    summary_chain = load_summarize_chain(
        llm=ChatOpenAI(temperature=0),
        chain_type="map_reduce",
        return_intermediate_steps=True
    )
    
    return summary_chain

# Usage
def summarize_documents(documents):
    chain = create_summarization_pipeline()
    result = chain({"input_documents": documents})
    return result["output_text"]
```

### 3. Multi-Step Analysis Workflow
```python
def create_analysis_workflow():
    # Step 1: Extract key information
    extraction_prompt = PromptTemplate.from_template(
        "Extract key information from: {text}"
    )
    
    # Step 2: Analyze extracted information
    analysis_prompt = PromptTemplate.from_template(
        "Analyze this information: {extracted_info}"
    )
    
    # Step 3: Generate recommendations
    recommendation_prompt = PromptTemplate.from_template(
        "Based on this analysis, provide recommendations: {analysis}"
    )
    
    # Chain them together
    workflow = (
        {"text": RunnablePassthrough()}
        | extraction_prompt
        | llm
        | {"extracted_info": RunnablePassthrough()}
        | analysis_prompt
        | llm
        | {"analysis": RunnablePassthrough()}
        | recommendation_prompt
        | llm
    )
    
    return workflow
```

### 4. Conversational AI with Memory
```python
def create_conversational_ai():
    memory = ConversationSummaryBufferMemory(
        llm=ChatOpenAI(temperature=0),
        max_token_limit=100,
        return_messages=True
    )
    
    conversation = ConversationChain(
        llm=ChatOpenAI(temperature=0.7),
        memory=memory,
        verbose=True
    )
    
    return conversation
```

### 5. Content Generation Pipeline
```python
def create_content_pipeline():
    # Research phase
    research_chain = (
        PromptTemplate.from_template("Research topic: {topic}")
        | llm
    )
    
    # Outline phase
    outline_chain = (
        PromptTemplate.from_template(
            "Create outline for: {topic}\nResearch: {research}"
        )
        | llm
    )
    
    # Writing phase
    writing_chain = (
        PromptTemplate.from_template(
            "Write content based on:\nTopic: {topic}\nOutline: {outline}"
        )
        | llm
    )
    
    # Complete pipeline
    content_pipeline = {
        "topic": RunnablePassthrough(),
        "research": research_chain
    } | {
        "topic": lambda x: x["topic"],
        "research": lambda x: x["research"],
        "outline": outline_chain
    } | writing_chain
    
    return content_pipeline
```

## Performance Optimization

### 1. Prompt Optimization
```python
# Bad: Verbose, unclear prompt
bad_prompt = """
Please analyze the following text and tell me what you think about it.
Consider all aspects and provide a comprehensive analysis.
Text: {text}
"""

# Good: Specific, structured prompt
good_prompt = """
Analyze the following text for:
1. Main themes (list 3-5)
2. Sentiment (positive/negative/neutral)
3. Key insights (2-3 bullet points)

Text: {text}

Format your response as:
Themes: [theme1, theme2, theme3]
Sentiment: [sentiment]
Insights:
• [insight1]
• [insight2]
"""
```

### 2. Efficient Retrieval Strategies
```python
# Use multiple retrieval strategies
def create_hybrid_retriever(vectorstore):
    # Similarity search retriever
    similarity_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    # MMR (Maximum Marginal Relevance) for diversity
    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20}
    )
    
    # Combine retrievers
    from langchain.retrievers import EnsembleRetriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[similarity_retriever, mmr_retriever],
        weights=[0.5, 0.5]
    )
    
    return ensemble_retriever
```

### 3. Batch Processing for Efficiency
```python
async def process_documents_batch(chain, documents, batch_size=10):
    results = []
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_tasks = [
            chain.ainvoke({"document": doc})
            for doc in batch
        ]
        
        batch_results = await asyncio.gather(*batch_tasks)
        results.extend(batch_results)
        
        # Optional: Add delay between batches
        await asyncio.sleep(1)
    
    return results
```

## Testing Strategies

### 1. Unit Testing Chains
```python
import unittest
from unittest.mock import Mock, patch

class TestMyChain(unittest.TestCase):
    def setUp(self):
        self.mock_llm = Mock()
        self.chain = create_my_chain(self.mock_llm)
    
    def test_chain_with_valid_input(self):
        self.mock_llm.invoke.return_value = "Expected output"
        
        result = self.chain.invoke({"input": "test"})
        
        self.assertEqual(result, "Expected output")
        self.mock_llm.invoke.assert_called_once()
    
    def test_chain_with_invalid_input(self):
        with self.assertRaises(ValueError):
            self.chain.invoke({"invalid": "input"})
```

### 2. Integration Testing
```python
import pytest
from unittest.mock import patch

@pytest.fixture
def test_vectorstore():
    # Create test vector store with known documents
    from langchain.vectorstores import Chroma
    from langchain.embeddings import OpenAIEmbeddings
    
    test_docs = ["Test document 1", "Test document 2"]
    return Chroma.from_texts(test_docs, OpenAIEmbeddings())

def test_qa_system_integration(test_vectorstore):
    qa_chain = create_qa_system_from_vectorstore(test_vectorstore)
    
    result = qa_chain.invoke({"query": "test query"})
    
    assert "answer" in result
    assert len(result["answer"]) > 0
```

### 3. Performance Testing
```python
import time
import statistics

def performance_test(chain, test_inputs, iterations=10):
    execution_times = []
    
    for _ in range(iterations):
        start_time = time.time()
        
        for input_data in test_inputs:
            chain.invoke(input_data)
        
        end_time = time.time()
        execution_times.append(end_time - start_time)
    
    return {
        "average_time": statistics.mean(execution_times),
        "median_time": statistics.median(execution_times),
        "min_time": min(execution_times),
        "max_time": max(execution_times)
    }
```

## Security Considerations

### 1. Input Sanitization
```python
import re

def sanitize_input(user_input: str) -> str:
    # Remove potentially harmful patterns
    sanitized = re.sub(r'[<>"\']', '', user_input)
    
    # Limit length
    sanitized = sanitized[:1000]
    
    # Remove multiple whitespaces
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    
    return sanitized

def secure_chain_invoke(chain, user_input):
    sanitized_input = sanitize_input(user_input)
    return chain.invoke({"input": sanitized_input})
```

### 2. Output Filtering
```python
def filter_sensitive_output(output: str) -> str:
    # Define patterns for sensitive information
    sensitive_patterns = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
        r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b',  # Credit card pattern
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email pattern
    ]
    
    filtered_output = output
    for pattern in sensitive_patterns:
        filtered_output = re.sub(pattern, '[REDACTED]', filtered_output)
    
    return filtered_output
```

### 3. Rate Limiting and Usage Tracking
```python
from collections import defaultdict
from datetime import datetime, timedelta

class UsageTracker:
    def __init__(self, max_requests_per_hour=100):
        self.max_requests = max_requests_per_hour
        self.requests = defaultdict(list)
    
    def can_make_request(self, user_id: str) -> bool:
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        # Clean old requests
        self.requests[user_id] = [
            req_time for req_time in self.requests[user_id]
            if req_time > hour_ago
        ]
        
        return len(self.requests[user_id]) < self.max_requests
    
    def record_request(self, user_id: str):
        self.requests[user_id].append(datetime.now())

# Usage
tracker = UsageTracker()

def rate_limited_invoke(chain, user_id, input_data):
    if not tracker.can_make_request(user_id):
        raise Exception("Rate limit exceeded")
    
    tracker.record_request(user_id)
    return chain.invoke(input_data)
```

## Monitoring and Observability

### 1. Custom Metrics Collection
```python
from dataclasses import dataclass
from typing import Dict, List
import json

@dataclass
class ChainMetrics:
    chain_name: str
    execution_time: float
    token_usage: int
    success: bool
    error_message: str = None

class MetricsCollector:
    def __init__(self):
        self.metrics: List[ChainMetrics] = []
    
    def record_execution(self, metrics: ChainMetrics):
        self.metrics.append(metrics)
    
    def get_success_rate(self, chain_name: str = None) -> float:
        filtered_metrics = self.metrics
        if chain_name:
            filtered_metrics = [m for m in self.metrics if m.chain_name == chain_name]
        
        if not filtered_metrics:
            return 0.0
        
        successful = sum(1 for m in filtered_metrics if m.success)
        return successful / len(filtered_metrics)
    
    def export_metrics(self) -> str:
        return json.dumps([
            {
                "chain_name": m.chain_name,
                "execution_time": m.execution_time,
                "token_usage": m.token_usage,
                "success": m.success,
                "error_message": m.error_message
            }
            for m in self.metrics
        ], indent=2)
```

### 2. Health Check Endpoint
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    version: str
    dependencies: Dict[str, str]

app = FastAPI()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        # Test LLM connectivity
        test_llm = OpenAI()
        test_result = test_llm.invoke("test")
        
        # Test vector store connectivity
        # test_vectorstore.similarity_search("test")
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            dependencies={
                "openai": "✓ Connected",
                "vectorstore": "✓ Connected"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )
```

## Common Pitfalls and Solutions

### 1. Token Limit Exceeding
```python
def smart_text_splitting(text: str, max_tokens: int = 4000):
    from langchain.text_splitter import TokenTextSplitter
    
    splitter = TokenTextSplitter(
        chunk_size=max_tokens - 500,  # Leave room for prompt
        chunk_overlap=200
    )
    
    chunks = splitter.split_text(text)
    return chunks

# Process large documents in chunks
def process_large_document(chain, document):
    chunks = smart_text_splitting(document)
    results = []
    
    for chunk in chunks:
        result = chain.invoke({"text": chunk})
        results.append(result)
    
    return results
```

### 2. Memory Management for Long Conversations
```python
def create_smart_memory(max_tokens: int = 2000):
    return ConversationSummaryBufferMemory(
        llm=ChatOpenAI(temperature=0),
        max_token_limit=max_tokens,
        return_messages=True
    )

# Implement conversation reset logic
class ConversationManager:
    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self.turn_count = 0
        self.memory = create_smart_memory()
    
    def add_interaction(self, human_input: str, ai_output: str):
        self.memory.save_context(
            {"input": human_input},
            {"output": ai_output}
        )
        self.turn_count += 1
        
        # Reset if too many turns
        if self.turn_count >= self.max_turns:
            self.reset_conversation()
    
    def reset_conversation(self):
        self.memory.clear()
        self.turn_count = 0
```

### 3. Handling API Failures
```python
import random
import time

def exponential_backoff_retry(func, max_retries=3, base_delay=1):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            time.sleep(delay)

# Usage
def robust_chain_invoke(chain, input_data):
    return exponential_backoff_retry(
        lambda: chain.invoke(input_data),
        max_retries=3
    )
```

This comprehensive guide covers the essential best practices for building robust, scalable, and secure LangChain applications. Remember to always test thoroughly, monitor performance, and iterate based on real-world usage patterns.