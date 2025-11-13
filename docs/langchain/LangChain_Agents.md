# LangChain Agents

## 4. Agents

Agents in LangChain are systems that can use tools to interact with the world and make decisions about which actions to take.

### 4.1 Core Concepts

#### What is an Agent?
An agent uses an LLM to decide which actions to take and in what order. Actions can be:
- Using a tool and observing its output
- Returning a response to the user

#### Agent Components:
1. **Agent**: The class responsible for deciding which action to take next
2. **Tools**: Functions that an agent can call
3. **Toolkits**: Groups of tools designed to be used together
4. **AgentExecutor**: The runtime for an agent

### 4.2 Tools

Tools are functions that agents can use to interact with the world.

#### Creating Custom Tools

```python
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")

class CustomCalculatorTool(BaseTool):
    name = "Calculator"
    description = "useful for when you need to answer questions about math"
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(
        self, a: int, b: int, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return a * b

    async def _arun(
        self, a: int, b: int, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(a, b, run_manager=run_manager.get_sync())
```

#### Using the @tool Decorator

```python
from langchain.tools import tool

@tool
def search(query: str) -> str:
    """Look up things online."""
    return "LangChain"

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

# Define tools list
tools = [search, multiply]
```

### 4.3 Agent Types

#### ReAct Agent
Uses the ReAct (Reasoning and Acting) framework:

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# Get the prompt to use
prompt = hub.pull("hwchase17/react")

# Create the agent
agent = create_react_agent(llm, tools, prompt)

# Create an agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run the agent
result = agent_executor.invoke({"input": "what is LangChain?"})
```

#### OpenAI Functions Agent
Uses OpenAI's function calling capability:

```python
from langchain.agents import create_openai_functions_agent

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

#### Structured Chat Agent
For multi-input tools:

```python
from langchain.agents import create_structured_chat_agent

agent = create_structured_chat_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

### 4.4 Agent Executors

The AgentExecutor is responsible for calling the agent and executing the actions it selects.

```python
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=3,
    handle_parsing_errors=True,
    return_intermediate_steps=True
)

# Execute with streaming
for chunk in agent_executor.stream({"input": "What's the weather like?"}):
    print(chunk)
```

### 4.5 Tool Usage Examples

#### Web Search Tool

```python
from langchain.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return search.run(query)
```

#### File System Tools

```python
from langchain.tools.file_management import (
    ReadFileTool,
    CopyFileTool,
    DeleteFileTool,
    MoveFileTool,
    WriteFileTool,
    ListDirectoryTool,
)

tools = [
    ReadFileTool(),
    CopyFileTool(),
    DeleteFileTool(),
    MoveFileTool(),
    WriteFileTool(),
    ListDirectoryTool(),
]
```

#### Python REPL Tool

```python
from langchain.tools import PythonREPLTool

python_repl = PythonREPLTool()
tools = [python_repl]
```

### 4.6 Advanced Agent Patterns

#### Multi-Agent Systems

```python
from langchain.agents import initialize_agent, AgentType

# Create specialized agents
researcher = initialize_agent(
    research_tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

writer = initialize_agent(
    writing_tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# Coordinate between agents
def multi_agent_workflow(query):
    research_result = researcher.run(f"Research: {query}")
    final_result = writer.run(f"Write based on: {research_result}")
    return final_result
```

#### Agent with Memory

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)
```

### 4.7 Error Handling and Debugging

```python
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_execution_time=60,
    early_stopping_method="generate"
)

# Custom error handling
def safe_agent_run(query):
    try:
        result = agent_executor.run(query)
        return result
    except Exception as e:
        return f"Agent encountered an error: {str(e)}"
```

### 4.8 Best Practices

1. **Tool Design**: Make tools focused and well-documented
2. **Error Handling**: Always implement proper error handling
3. **Timeouts**: Set reasonable execution timeouts
4. **Validation**: Validate tool inputs and outputs
5. **Logging**: Enable verbose logging for debugging
6. **Testing**: Test agents with various scenarios

### 4.9 Common Patterns

#### Research Agent

```python
@tool
def search_tool(query: str) -> str:
    """Search for information online."""
    # Implementation here
    pass

@tool
def summarize_tool(text: str) -> str:
    """Summarize long text."""
    # Implementation here
    pass

research_tools = [search_tool, summarize_tool]

research_agent = create_react_agent(llm, research_tools, prompt)
research_executor = AgentExecutor(agent=research_agent, tools=research_tools)
```

#### Data Analysis Agent

```python
@tool
def load_data(filepath: str) -> str:
    """Load data from a file."""
    # Implementation here
    pass

@tool
def analyze_data(data: str) -> str:
    """Analyze data and return insights."""
    # Implementation here
    pass

@tool
def create_visualization(data: str, chart_type: str) -> str:
    """Create a visualization of the data."""
    # Implementation here
    pass

analysis_tools = [load_data, analyze_data, create_visualization]
```

This covers the comprehensive agent functionality in LangChain, including creation, execution, and best practices for building robust agent systems.