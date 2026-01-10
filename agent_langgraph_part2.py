# ============================================================
# PART 2 — SINGLE AGENT WITH MULTIPLE TOOLS
# ============================================================
# Purpose:
# - Extend Part 1 by adding a SECOND tool
# - Demonstrate how the SAME agent chooses between tools
# - Show agent reasoning clearly for tutorial/blog usage
#
# FLOW (IMPORTANT):
# agent → tool → END
# NO LOOPS, NO PLANNER, NO CRITIC (yet)
# ============================================================


# -------------------------
# Standard typing utilities
# -------------------------
from typing import TypedDict, List


# -------------------------
# LangChain / LangGraph imports
# -------------------------
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


# ============================================================
# 1️⃣ TOOL DEFINITIONS
# ============================================================

@tool
def multiply(a: int, b: int) -> int:
    """
    Multiply two integers.

    Tool intent:
    - Used when the task involves multiplication
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """
    Add two integers.

    Tool intent:
    - Used when the task involves addition
    """
    return a + b


# -------------------------
# Register ALL tools here
# -------------------------
TOOLS = [multiply, add]


# ============================================================
# 2️⃣ STATE DEFINITION
# ============================================================

class AgentState(TypedDict):
    """
    Shared state passed through the graph.

    messages:
    - Conversation history
    - Includes:
      - HumanMessage (user)
      - AIMessage (agent)
      - ToolMessage (tool result)
    """
    messages: List


# ============================================================
# 3️⃣ LLM SETUP
# ============================================================

# -------------------------
# Create the chat model
# -------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0  # deterministic for tutorials
)

# -------------------------
# Bind tools to the model
# This enables tool calling
# -------------------------
llm_with_tools = llm.bind_tools(TOOLS)


# ============================================================
# 4️⃣ AGENT NODE
# ============================================================

def agent_node(state: AgentState):
    """
    Core agent reasoning step.

    Responsibilities:
    1. Read the conversation so far
    2. Decide whether a tool is needed
    3. Choose the correct tool (if any)
    4. Return ONE AIMessage
    """

    messages = state["messages"]

    # Invoke the LLM with tool support
    response = llm_with_tools.invoke(messages)

    # --------------------------------------------------
    # Tutorial-friendly reasoning output
    # --------------------------------------------------
    if isinstance(response, AIMessage) and response.tool_calls:
        tool_call = response.tool_calls[0]

        print("\n[Agent Reasoning]")
        print("→ Tool selected:", tool_call["name"])
        print("→ Tool arguments:", tool_call["args"])

    # Append agent response to state
    return {"messages": messages + [response]}


# ============================================================
# 5️⃣ TOOL NODE
# ============================================================

# ToolNode automatically:
# - Reads tool_calls from the AIMessage
# - Executes the correct Python function
# - Returns a ToolMessage
tool_node = ToolNode(TOOLS)


# ============================================================
# 6️⃣ ROUTING LOGIC
# ============================================================

def route(state: AgentState):
    """
    Routing rules:

    - If agent requested a tool → go to tool node
    - Otherwise → END
    """

    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    return END


# ============================================================
# 7️⃣ GRAPH CONSTRUCTION
# ============================================================

graph = StateGraph(AgentState)

# Register nodes
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

# Entry point
graph.set_entry_point("agent")

# Conditional routing
graph.add_conditional_edges("agent", route)

# IMPORTANT:
# After tool execution → STOP
graph.add_edge("tools", END)

# Compile graph
app = graph.compile()


# ============================================================
# 8️⃣ RUN THE AGENT
# ============================================================

# Try different prompts to see tool choice
initial_input = HumanMessage(
    #content="Mulitply 8 and 9"
    #content="Add 8 and 9"
    content="combine the values of 8 and 9 . also give value of 10 times 9"   
    # Try:
    # "Multiply 8 and 9"
)

result = app.invoke(
    {"messages": [initial_input]}
)


# ============================================================
# 9️⃣ OUTPUT
# ============================================================

print("\n[Tool Output]")
for msg in result["messages"]:
    if isinstance(msg, ToolMessage):
        print("→ tool result:", msg.content)

print("\nFinal Answer:")
print(result["messages"][-1].content)

