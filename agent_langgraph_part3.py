# ============================================================
# PART 3 — MULTI-AGENT SYSTEM (PLANNER + EXECUTOR)
# ============================================================
# What this program demonstrates:
# - Separation of reasoning and execution
# - Planner agent decides WHAT to do
# - Executor agent decides HOW to do it (tool calls)
# - Same tools as Part-2
#
# Architecture:
# User → Planner → Executor → Tool → END
# ============================================================


from typing import TypedDict, List


# -------------------------
# LangChain / LangGraph
# -------------------------
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage
)

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


# ============================================================
# 1️⃣ TOOLS (unchanged from Part-2)
# ============================================================

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


TOOLS = [add, multiply]


# ============================================================
# 2️⃣ SHARED STATE
# ============================================================

class AgentState(TypedDict):
    """
    messages:
    - Shared conversation across agents
    """
    messages: List


# ============================================================
# 3️⃣ MODELS
# ============================================================

# Planner model: focuses on reasoning and intent
planner_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Executor model: allowed to call tools
executor_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
).bind_tools(TOOLS)


# ============================================================
# 4️⃣ PLANNER AGENT
# ============================================================

def planner_node(state: AgentState):
    """
    Planner agent responsibility:
    - Read user request
    - Decide what action should be taken
    - DO NOT call tools
    """

    messages = state["messages"]

    system_instruction = HumanMessage(
        content=(
            "You are a planner agent. "
            "Decide what operation is required (add or multiply) "
            "but do NOT perform calculations or call tools. "
            "Explain the plan briefly."
        )
    )

    response = planner_llm.invoke(messages + [system_instruction])

    print("\n[Planner Reasoning]")
    print(response.content)

    return {"messages": messages + [response]}


# ============================================================
# 5️⃣ EXECUTOR AGENT
# ============================================================

def executor_node(state: AgentState):
    """
    Executor agent responsibility:
    - Read planner output
    - Call the appropriate tool
    """

    messages = state["messages"]

    response = executor_llm.invoke(messages)

    if response.tool_calls:
        tool_call = response.tool_calls[0]
        print("\n[Executor Reasoning]")
        print("→ Tool selected:", tool_call["name"])
        print("→ Tool arguments:", tool_call["args"])

    return {"messages": messages + [response]}


# ============================================================
# 6️⃣ TOOL NODE
# ============================================================

tool_node = ToolNode(TOOLS)


# ============================================================
# 7️⃣ ROUTING LOGIC
# ============================================================

def route_from_executor(state: AgentState):
    """
    If executor requested a tool → go to tool
    Otherwise → END
    """

    last = state["messages"][-1]

    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"

    return END


# ============================================================
# 8️⃣ GRAPH CONSTRUCTION
# ============================================================

graph = StateGraph(AgentState)

graph.add_node("planner", planner_node)
graph.add_node("executor", executor_node)
graph.add_node("tools", tool_node)

graph.set_entry_point("planner")

graph.add_edge("planner", "executor")

graph.add_conditional_edges(
    "executor",
    route_from_executor
)

graph.add_edge("tools", END)

app = graph.compile()


# ============================================================
# 9️⃣ RUN
# ============================================================

initial_input = HumanMessage(
    content="combine the values of 6 and 7"
)

result = app.invoke(
    {"messages": [initial_input]}
)


# ============================================================
# 🔟 OUTPUT
# ============================================================

print("\n[Tool Output]")
for msg in result["messages"]:
    if isinstance(msg, ToolMessage):
        print("→", msg.content)

print("\nFinal Answer:")
print(result["messages"][-1].content)

