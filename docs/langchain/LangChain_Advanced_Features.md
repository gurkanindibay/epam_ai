# LangChain Advanced Features

## Output Parsers

### PydanticOutputParser
- **Purpose**: Parse LLM output into Pydantic models for type safety
- **Use Case**: When you need structured, validated output

```python
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from pydantic import BaseModel, Field

# Define output structure
class Person(BaseModel):
    name: str = Field(description="person's name")
    age: int = Field(description="person's age")
    occupation: str = Field(description="person's occupation")

# Create parser
parser = PydanticOutputParser(pydantic_object=Person)

# Create prompt with format instructions
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Use with LLM
llm = OpenAI(temperature=0)
_input = prompt.format_prompt(query="Tell me about a fictional character")
output = llm(_input.to_string())

# Parse output
parsed_output = parser.parse(output)
print(f"Name: {parsed_output.name}, Age: {parsed_output.age}")
```

### CommaSeparatedListOutputParser
- **Purpose**: Parse comma-separated lists from LLM output
- **Use Case**: When you need list-formatted responses

```python
from langchain.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()

prompt = PromptTemplate(
    template="List five {subject}.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Use parser
chain = prompt | llm | parser
result = chain.invoke({"subject": "programming languages"})
print(result)  # ['Python', 'Java', 'JavaScript', 'C++', 'Go']
```

### StructuredOutputParser
- **Purpose**: Parse structured output with multiple fields
- **Use Case**: When you need multiple related pieces of information

```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Define response schema
response_schemas = [
    ResponseSchema(name="answer", description="answer to the user's question"),
    ResponseSchema(name="source", description="source used to answer the question")
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)

prompt = PromptTemplate(
    template="answer the users question as best as possible.\n"
             "{format_instructions}\n{question}",
    input_variables=["question"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
```

### DatetimeOutputParser
- **Purpose**: Parse datetime objects from LLM output
- **Use Case**: When working with dates and times

```python
from langchain.output_parsers import DatetimeOutputParser

parser = DatetimeOutputParser()

prompt = PromptTemplate.from_template(
    "Answer the users question:\n\n"
    "{question}\n\n"
    "{format_instructions}"
)

chain = prompt | llm | parser
result = chain.invoke({
    "question": "When was the first man on the moon?",
    "format_instructions": parser.get_format_instructions()
})
```

## Callbacks

### BaseCallbackHandler
- **Purpose**: Monitor and log LangChain operations
- **Use Case**: Debugging, monitoring, custom logging

```python
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any, Dict, List, Optional

class CustomCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        print(f"LLM started with prompts: {prompts}")

    def on_llm_end(self, response: Any, **kwargs: Any) -> Any:
        print(f"LLM ended with response: {response}")

    def on_llm_error(self, error: Exception, **kwargs: Any) -> Any:
        print(f"LLM error: {error}")

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        print(f"Chain started with inputs: {inputs}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        print(f"Chain ended with outputs: {outputs}")

# Use callback
handler = CustomCallbackHandler()
llm = OpenAI(callbacks=[handler])
result = llm("What is the capital of France?")
```

### StdOutCallbackHandler
- **Purpose**: Print LangChain operations to stdout
- **Use Case**: Simple debugging and monitoring

```python
from langchain.callbacks import StdOutCallbackHandler

handler = StdOutCallbackHandler()
llm = OpenAI(callbacks=[handler], verbose=True)
```

### FileCallbackHandler
- **Purpose**: Log operations to a file
- **Use Case**: Production logging and audit trails

```python
from langchain.callbacks import FileCallbackHandler

handler = FileCallbackHandler("langchain.log")
llm = OpenAI(callbacks=[handler])
```

### WandbCallbackHandler
- **Purpose**: Integration with Weights & Biases for experiment tracking
- **Use Case**: ML experiment tracking and monitoring

```python
from langchain.callbacks import WandbCallbackHandler
import wandb

# Initialize wandb
wandb.init()

# Create handler
handler = WandbCallbackHandler()
llm = OpenAI(callbacks=[handler])
```

## LangChain Expression Language (LCEL)

### Basic LCEL Syntax
- **Purpose**: Compose chains using pipe operator
- **Benefits**: More readable, easier to debug, better streaming support

```python
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# Traditional approach
prompt = PromptTemplate.from_template("Tell me a joke about {topic}")
llm = OpenAI()
chain = prompt | llm

# Use the chain
result = chain.invoke({"topic": "programming"})
```

### Complex LCEL Chains
```python
from langchain.schema.runnable import RunnableParallel

# Parallel execution
chain = RunnableParallel({
    "joke": prompt | llm,
    "poem": PromptTemplate.from_template("Write a poem about {topic}") | llm
})

result = chain.invoke({"topic": "nature"})
print(result["joke"])
print(result["poem"])
```

### LCEL with Memory
```python
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnableLambda

memory = ConversationBufferMemory(return_messages=True)

def load_memory(_):
    return memory.load_memory_variables({})["history"]

def save_memory(output):
    memory.save_context({"input": output["input"]}, {"output": output["output"]})
    return output

# Chain with memory
chain = (
    RunnableLambda(load_memory) |
    prompt |
    llm |
    RunnableLambda(save_memory)
)
```

### LCEL with Retrieval
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Create retriever
vectorstore = Chroma.from_texts(texts, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# RAG chain using LCEL
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

result = rag_chain.invoke("What is the main topic?")
```

## Advanced Chain Patterns

### Sequential Chain with Error Handling
```python
from langchain.chains import SequentialChain
from langchain.schema import BaseOutputParser

class ErrorHandlingOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        try:
            # Attempt to parse
            return text.strip()
        except Exception as e:
            return f"Parsing error: {e}"

# Create chain with error handling
chain = (
    prompt
    | llm
    | ErrorHandlingOutputParser()
)
```

### Conditional Chain Execution
```python
from langchain.schema.runnable import RunnableBranch

# Conditional execution based on input
branch = RunnableBranch(
    (lambda x: "math" in x["question"].lower(), math_chain),
    (lambda x: "history" in x["question"].lower(), history_chain),
    general_chain  # default
)

result = branch.invoke({"question": "What is 2+2 in math?"})
```

### Retry Logic
```python
from langchain.schema.runnable import RunnableRetry

# Add retry logic to chains
retry_chain = RunnableRetry(
    bound=llm,
    max_attempts=3,
    wait_exponential_jitter=True
)
```

## Streaming and Async Support

### Streaming Responses
```python
# Stream responses for better user experience
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)
```

### Async Execution
```python
import asyncio

async def async_chain_example():
    # Async chain execution
    result = await chain.ainvoke({"topic": "AI"})
    return result

# Run async
result = asyncio.run(async_chain_example())
```

### Batch Processing
```python
# Process multiple inputs in batch
inputs = [
    {"topic": "AI"},
    {"topic": "ML"},
    {"topic": "DL"}
]

results = chain.batch(inputs)
```

## Custom Runnable Components

### Custom Runnable Class
```python
from langchain.schema.runnable import Runnable
from typing import Any, Dict

class CustomRunnable(Runnable):
    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        # Custom processing logic
        processed_input = input["text"].upper()
        return {"processed": processed_input}

    async def ainvoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        # Async version
        return self.invoke(input)

# Use custom runnable
custom = CustomRunnable()
result = custom.invoke({"text": "hello world"})
```

### RunnableLambda for Quick Functions
```python
from langchain.schema.runnable import RunnableLambda

# Quick function wrapper
uppercase = RunnableLambda(lambda x: x["text"].upper())
chain = prompt | llm | uppercase
```

## Performance Optimization

### Caching
```python
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

# Enable caching
set_llm_cache(InMemoryCache())

# Now LLM calls will be cached
llm = OpenAI()
result1 = llm("What is AI?")  # Makes API call
result2 = llm("What is AI?")  # Uses cache
```

### Parallel Processing
```python
from concurrent.futures import ThreadPoolExecutor
import asyncio

async def process_multiple_chains():
    tasks = [
        chain.ainvoke({"topic": f"topic_{i}"})
        for i in range(10)
    ]
    results = await asyncio.gather(*tasks)
    return results
```

### Token Usage Tracking
```python
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = chain.invoke({"topic": "AI"})
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Total Cost: ${cb.total_cost}")
```

## Testing and Debugging

### Chain Debugging
```python
# Enable verbose mode for debugging
chain = prompt | llm.with_config({"callbacks": [StdOutCallbackHandler()]})

# Or use the debug method
chain.debug({"topic": "AI"})
```

### Unit Testing Chains
```python
import unittest
from unittest.mock import Mock

class TestMyChain(unittest.TestCase):
    def setUp(self):
        self.mock_llm = Mock()
        self.mock_llm.invoke.return_value = "Mocked response"
        self.chain = prompt | self.mock_llm

    def test_chain_invoke(self):
        result = self.chain.invoke({"topic": "test"})
        self.assertEqual(result, "Mocked response")
```

### Error Handling Best Practices
```python
from langchain.schema.runnable import RunnablePassthrough

def safe_invoke(chain, input_data, default_response="Error occurred"):
    try:
        return chain.invoke(input_data)
    except Exception as e:
        print(f"Chain error: {e}")
        return default_response

# Use wrapper function
result = safe_invoke(chain, {"topic": "AI"})
```

## Integration Patterns

### FastAPI Integration
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    result = await chain.ainvoke({"question": request.question})
    return {"answer": result}
```

### Gradio Interface
```python
import gradio as gr

def chat_interface(message, history):
    response = chain.invoke({"question": message})
    return response

iface = gr.ChatInterface(
    fn=chat_interface,
    title="LangChain Chatbot"
)
iface.launch()
```

### Custom Middleware
```python
from langchain.schema.runnable import RunnablePassthrough

class LoggingMiddleware:
    def __init__(self, chain):
        self.chain = chain
    
    def invoke(self, input_data):
        print(f"Input: {input_data}")
        result = self.chain.invoke(input_data)
        print(f"Output: {result}")
        return result

# Wrap chain with middleware
logged_chain = LoggingMiddleware(chain)
```