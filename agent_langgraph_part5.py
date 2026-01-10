# ============================================================
# PART 5 — PLANNER + EXECUTOR + CRITIC + REFLECTION LOOP
# ============================================================
# Incremental change from Part-4:
# - Adds a Reflection / Retry loop
# - If Critic says result is incorrect → retry once
#
# Architecture:
# User → Planner → Executor → Tool → Critic → (Retry or END)
# ============================================================

from typing import TypedDict, List

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


# ============================================================
# 1️⃣ TOOLS (UNCHANGED)
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
# 2️⃣ SHARED STATE (EXTENDED)
# ============================================================

class AgentState(TypedDict):
    """
    Shared memory across all agents.
    """
    messages: List
    tool_result: str
    retries: int   # NEW: retry counter (safety guard)


# ============================================================
# 3️⃣ MODELS (UNCHANGED)
# ============================================================

planner_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

executor_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
).bind_tools(TOOLS)

critic_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# 4️⃣ PLANNER AGENT (UNCHANGED)
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
        "tool_result": state["tool_result"],
        "retries": state["retries"]
    }


# ============================================================
# 5️⃣ EXECUTOR AGENT (UNCHANGED)
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
        "tool_result": "",
        "retries": state["retries"]
    }


# ============================================================
# 6️⃣ TOOL NODE (UNCHANGED)
# ============================================================

tool_node = ToolNode(TOOLS)


# ============================================================
# 7️⃣ CAPTURE TOOL OUTPUT (UNCHANGED)
# ============================================================

def capture_tool_result(state: AgentState):
    """
    Extracts tool output safely for Critic.
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
        "tool_result": result,
        "retries": state["retries"]
    }


# ============================================================
# 8️⃣ CRITIC AGENT (UNCHANGED)
# ============================================================

def critic_node(state: AgentState):
    """
    Critic:
    - Reviews tool result
    - Declares correctness
    """

    critic_prompt = HumanMessage(
        content=(
            f"The user asked: '{state['messages'][0].content}'. "
            f"The tool produced the result: {state['tool_result']}. "
            "Is this result correct? Answer clearly."
        )
    )

    response = critic_llm.invoke([critic_prompt])

    print("\n[Critic Review]")
    print(response.content)

    return {
        "messages": state["messages"] + [response],
        "tool_result": state["tool_result"],
        "retries": state["retries"]
    }


# ============================================================
# 9️⃣ REFLECTION ROUTER (NEW)
# ============================================================

def reflection_router(state: AgentState):
    """
    Reflection logic:
    - If Critic says incorrect AND retry < 1 → retry
    - Else → END
    """

    last_message = state["messages"][-1].content

    if "not correct" in last_message.lower() and state["retries"] < 1:
        print("\n[Reflection]")
        print("❌ Result incorrect — retrying once...\n")

        return "planner"

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
    lambda state: "tools" if isinstance(state["messages"][-1], AIMessage)
    and state["messages"][-1].tool_calls else END
)

graph.add_edge("tools", "capture")
graph.add_edge("capture", "critic")

graph.add_conditional_edges(
    "critic",
    reflection_router,
    {
        "planner": "planner",
        END: END
    }
)

app = graph.compile()


# ============================================================
# 1️⃣1️⃣ RUN
# ============================================================

initial_input = HumanMessage(
    content="add values 12 and 8 and give me the product"
)

result = app.invoke(
    {
        "messages": [initial_input],
        "tool_result": "",
        "retries": 0
    }
)

print("\nFinal Answer:")
print(result["tool_result"])

