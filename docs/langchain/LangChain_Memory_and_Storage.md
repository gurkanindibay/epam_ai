# LangChain Memory and Storage Components

## Memory

### ConversationBufferMemory
- **Purpose**: Stores the entire conversation history
- **Use Case**: When you need to maintain context of the entire conversation
- **Memory Limit**: Can grow very large with long conversations

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.llms import OpenAI

# Initialize memory
memory = ConversationBufferMemory()

# Create conversation chain with memory
llm = OpenAI(temperature=0)
conversation = ConversationChain(
    llm=llm, 
    memory=memory, 
    verbose=True
)

# Have a conversation
response1 = conversation.predict(input="Hi, I'm John.")
response2 = conversation.predict(input="What's my name?")
```

### ConversationBufferWindowMemory
- **Purpose**: Maintains a sliding window of the most recent K interactions
- **Use Case**: When you want to limit memory usage while keeping recent context
- **Configuration**: Set `k` parameter to define window size

```python
from langchain.memory import ConversationBufferWindowMemory

# Keep only the last 2 interactions
memory = ConversationBufferWindowMemory(k=2)
conversation = ConversationChain(
    llm=llm, 
    memory=memory, 
    verbose=True
)
```

### ConversationSummaryMemory
- **Purpose**: Summarizes the conversation as it progresses
- **Use Case**: For very long conversations where you need to preserve key information
- **Benefits**: Maintains context while controlling memory size

```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=llm)
conversation = ConversationChain(
    llm=llm, 
    memory=memory, 
    verbose=True
)
```

### ConversationSummaryBufferMemory
- **Purpose**: Combines buffer and summary approaches
- **Behavior**: Keeps recent messages in buffer, summarizes older ones
- **Configuration**: Set `max_token_limit` to control when summarization occurs

```python
from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=100
)
```

## Retrievers

### VectorStoreRetriever
- **Purpose**: Retrieve documents based on vector similarity
- **Use Case**: Semantic search over document collections

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(
    texts=["Document 1 content", "Document 2 content"],
    embeddings=embeddings
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# Retrieve relevant documents
docs = retriever.get_relevant_documents("query")
```

### MultiQueryRetriever
- **Purpose**: Generates multiple queries to improve retrieval
- **Use Case**: When a single query might not capture all relevant documents

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

# Create multi-query retriever
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)
```

### ContextualCompressionRetriever
- **Purpose**: Compresses retrieved documents to relevant portions
- **Use Case**: When retrieved documents are too long or contain irrelevant content

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Create compressor
compressor = LLMChainExtractor.from_llm(llm)

# Create compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever()
)
```

## Vector Stores

### Chroma
- **Purpose**: Local vector database
- **Use Case**: Development, small-scale applications

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Create Chroma vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(
    texts=documents,
    embeddings=embeddings,
    persist_directory="./chroma_db"
)

# Persist the database
vectorstore.persist()
```

### Pinecone
- **Purpose**: Cloud-based vector database
- **Use Case**: Production applications, scalable solutions

```python
import pinecone
from langchain.vectorstores import Pinecone

# Initialize Pinecone
pinecone.init(
    api_key="your-api-key",
    environment="your-environment"
)

# Create index
index_name = "my-index"
vectorstore = Pinecone.from_texts(
    texts=documents,
    embeddings=embeddings,
    index_name=index_name
)
```

### Weaviate
- **Purpose**: Open-source vector database with GraphQL API
- **Use Case**: Complex data relationships, hybrid search

```python
from langchain.vectorstores import Weaviate
import weaviate

# Create Weaviate client
client = weaviate.Client("http://localhost:8080")

# Create vector store
vectorstore = Weaviate.from_texts(
    texts=documents,
    embeddings=embeddings,
    client=client,
    index_name="Document"
)
```

### FAISS
- **Purpose**: Facebook's library for similarity search
- **Use Case**: Fast similarity search, local deployment

```python
from langchain.vectorstores import FAISS

# Create FAISS vector store
vectorstore = FAISS.from_texts(
    texts=documents,
    embeddings=embeddings
)

# Save to disk
vectorstore.save_local("faiss_index")

# Load from disk
new_vectorstore = FAISS.load_local("faiss_index", embeddings)
```

## Document Loaders

### TextLoader
- **Purpose**: Load plain text files
- **Use Case**: Simple text documents

```python
from langchain.document_loaders import TextLoader

loader = TextLoader("./sample.txt")
documents = loader.load()
```

### CSVLoader
- **Purpose**: Load CSV files
- **Use Case**: Structured data in CSV format

```python
from langchain.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path='./data.csv')
documents = loader.load()
```

### PDFLoader
- **Purpose**: Load PDF documents
- **Use Case**: PDF documents, reports

```python
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("./document.pdf")
pages = loader.load_and_split()
```

### DirectoryLoader
- **Purpose**: Load multiple files from a directory
- **Use Case**: Batch processing of documents

```python
from langchain.document_loaders import DirectoryLoader

loader = DirectoryLoader('./documents', glob="*.txt")
docs = loader.load()
```

### WebBaseLoader
- **Purpose**: Load content from web pages
- **Use Case**: Scraping web content for RAG

```python
from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://example.com/article")
docs = loader.load()
```

### GitLoader
- **Purpose**: Load files from Git repositories
- **Use Case**: Code documentation, repository analysis

```python
from langchain.document_loaders import GitLoader

loader = GitLoader(
    clone_url="https://github.com/user/repo",
    repo_path="./temp_repo",
    branch="main"
)
docs = loader.load()
```

## Text Splitters

### CharacterTextSplitter
- **Purpose**: Split text by character count
- **Use Case**: Simple text splitting

```python
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

texts = text_splitter.split_text(raw_text)
```

### RecursiveCharacterTextSplitter
- **Purpose**: Split text hierarchically by different separators
- **Use Case**: Most common choice, works well for most text types

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

texts = text_splitter.split_text(raw_text)
```

### TokenTextSplitter
- **Purpose**: Split text by token count
- **Use Case**: When you need precise token control

```python
from langchain.text_splitter import TokenTextSplitter

text_splitter = TokenTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

texts = text_splitter.split_text(raw_text)
```

### CodeTextSplitter
- **Purpose**: Split code while respecting language syntax
- **Use Case**: Processing code files

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language
)

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000,
    chunk_overlap=200
)

code_texts = python_splitter.split_text(python_code)
```

## Complete RAG Example with Memory

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Load and process documents
loader = TextLoader('./documents/state_of_the_union.txt')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings)

# Create memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create conversational retrieval chain
qa = ConversationalRetrievalChain.from_llm(
    OpenAI(temperature=0),
    vectorstore.as_retriever(),
    memory=memory
)

# Have a conversation
result = qa({"question": "What did the president say about Ketanji Brown Jackson"})
print(result['answer'])

# Ask follow-up question (memory will be used)
result = qa({"question": "Did he mention her predecessor?"})
print(result['answer'])
```

## Best Practices for Memory and Storage

### Memory Selection
1. **Short conversations**: Use `ConversationBufferMemory`
2. **Long conversations**: Use `ConversationSummaryBufferMemory`
3. **Resource constraints**: Use `ConversationBufferWindowMemory`
4. **Key information preservation**: Use `ConversationSummaryMemory`

### Vector Store Selection
1. **Development/Local**: Chroma, FAISS
2. **Production/Cloud**: Pinecone, Weaviate
3. **Hybrid search**: Weaviate
4. **Performance-critical**: FAISS

### Text Splitting Guidelines
1. **Start with**: `RecursiveCharacterTextSplitter`
2. **Code files**: Use language-specific splitters
3. **Token limits**: Use `TokenTextSplitter`
4. **Chunk size**: Balance between context and retrieval precision

### Retrieval Optimization
1. **Similarity search**: Standard vector retrieval
2. **Multiple perspectives**: `MultiQueryRetriever`
3. **Reduce noise**: `ContextualCompressionRetriever`
4. **Hybrid approach**: Combine multiple retrieval methods