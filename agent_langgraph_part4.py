# ============================================================
# PART 4 — PLANNER + EXECUTOR + CRITIC (CORRECT & STABLE)
# ============================================================
# This version FIXES:
# - OpenAI tool-message ordering errors
# - Infinite recursion
# - Critic visibility issues
#
# Incremental change from Part-3:
# - Adds a Critic agent AFTER tool execution
# - Critic receives a CLEAN summary, not raw tool messages
#
# Architecture:
# User → Planner → Executor → Tool → Critic → END
# ============================================================

from typing import TypedDict, List

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


# ============================================================
# 1️⃣ TOOLS (same as Part-3)
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
    Shared memory across all agents.
    """
    messages: List
    tool_result: str  # NEW: clean tool output for Critic


# ============================================================
# 3️⃣ MODELS
# ============================================================

planner_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

executor_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
).bind_tools(TOOLS)

critic_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# 4️⃣ PLANNER AGENT
# ============================================================

def planner_node(state: AgentState):
    """
    Planner:
    - Understands user intent
    - Decides add vs multiply
    - Does NOT call tools
    """

    messages = state["messages"]

    planner_prompt = HumanMessage(
        content=(
            "You are a planner agent. Decide whether to ADD or MULTIPLY "
            "based on the user's request. Do NOT calculate. "
            "Explain your reasoning briefly."
        )
    )

    response = planner_llm.invoke(messages + [planner_prompt])

    print("\n[Planner Reasoning]")
    print(response.content)

    return {
        "messages": messages + [response],
        "tool_result": ""
    }


# ============================================================
# 5️⃣ EXECUTOR AGENT
# ============================================================

def executor_node(state: AgentState):
    """
    Executor:
    - Reads planner reasoning
    - Calls the correct tool
    """

    messages = state["messages"]

    response = executor_llm.invoke(messages)

    if response.tool_calls:
        call = response.tool_calls[0]
        print("\n[Executor Reasoning]")
        print("→ Tool selected:", call["name"])
        print("→ Tool arguments:", call["args"])

    return {
        "messages": messages + [response],
        "tool_result": ""
    }


# ============================================================
# 6️⃣ TOOL NODE
# ============================================================

tool_node = ToolNode(TOOLS)


# ============================================================
# 7️⃣ CAPTURE TOOL OUTPUT (CRITICAL FIX)
# ============================================================

def capture_tool_result(state: AgentState):
    """
    Extracts the tool output and stores it as plain text
    so the Critic can safely review it.
    """

    messages = state["messages"]

    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

    if tool_messages:
        result = tool_messages[-1].content
        print("\n[Tool Output]")
        print("→", result)
    else:
        result = "No tool output found."

    return {
        "messages": messages,
        "tool_result": result
    }


# ============================================================
# 8️⃣ CRITIC AGENT (NEW)
# ============================================================

def critic_node(state: AgentState):
    """
    Critic:
    - Reviews the tool result
    - Confirms correctness
    - Does NOT call tools
    """

    result = state["tool_result"]

    critic_prompt = HumanMessage(
        content=(
          f"The user asked: '{state['messages'][0].content}'. "
          f"The tool produced the result: {result}. "
          "Verify whether this result is correct. Answer clearly."
        )
    )

    response = critic_llm.invoke([critic_prompt])

    print("\n[Critic Review]")
    print(response.content)

    return {
        "messages": state["messages"] + [response],
        "tool_result": result
    }


# ============================================================
# 9️⃣ ROUTING LOGIC
# ============================================================

def route_from_executor(state: AgentState):
    last = state["messages"][-1]

    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"

    return END


# ============================================================
# 🔟 GRAPH CONSTRUCTION
# ============================================================

graph = StateGraph(AgentState)

graph.add_node("planner", planner_node)
graph.add_node("executor", executor_node)
graph.add_node("tools", tool_node)
graph.add_node("capture", capture_tool_result)
graph.add_node("critic", critic_node)

graph.set_entry_point("planner")

graph.add_edge("planner", "executor")

graph.add_conditional_edges(
    "executor",
    route_from_executor
)

graph.add_edge("tools", "capture")
graph.add_edge("capture", "critic")
graph.add_edge("critic", END)

app = graph.compile()


# ============================================================
# 1️⃣1️⃣ RUN
# ============================================================

initial_input = HumanMessage(content="ajoutez 12 et 8")

result = app.invoke(
    {"messages": [initial_input], "tool_result": ""}
)

print("\nFinal Answer:")
print(result["tool_result"])

