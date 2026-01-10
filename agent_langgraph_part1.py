# ============================================================
# PART 1 — SINGLE TOOL LANGGRAPH AGENT (NO RECURSION)
# ============================================================
# Purpose:
# - Demonstrate a simple reactive agent
# - Show how the agent selects and uses ONE tool
# - Print agent reasoning in a tutorial-friendly way
#
# CRITICAL RULE FOR PART-1:
# - agent → tool → END
# - NO looping back to agent
# ============================================================


from typing import TypedDict, List

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


# ============================================================
# 1️⃣ TOOL DEFINITION
# ============================================================

@tool
def multiply(a: int, b: int) -> int:
    """
    Multiply two integers and return the result.
    """
    return a * b


TOOLS = [multiply]


# ============================================================
# 2️⃣ STATE DEFINITION
# ============================================================

class AgentState(TypedDict):
    """
    Shared graph state.

    messages:
    - Conversation history
    - Includes HumanMessage, AIMessage, ToolMessage
    """
    messages: List


# ============================================================
# 3️⃣ LLM SETUP
# ============================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

llm_with_tools = llm.bind_tools(TOOLS)


# ============================================================
# 4️⃣ AGENT NODE
# ============================================================

def agent_node(state: AgentState):
    """
    Agent reasoning step.

    - Reads the user request
    - Decides whether to call a tool
    - Produces ONE AIMessage
    """

    messages = state["messages"]

    response = llm_with_tools.invoke(messages)

    # Safely inspect tool choice (for tutorial visibility)
    if isinstance(response, AIMessage) and response.tool_calls:
        tool_call = response.tool_calls[0]

        print("\n[Agent Reasoning]")
        print("→ Tool selected:", tool_call["name"])
        print("→ Tool arguments:", tool_call["args"])

    return {"messages": messages + [response]}


# ============================================================
# 5️⃣ TOOL NODE
# ============================================================

tool_node = ToolNode(TOOLS)


# ============================================================
# 6️⃣ ROUTING LOGIC (KEY FIX)
# ============================================================

def route(state: AgentState):
    """
    Routing rules:

    - If agent requested a tool → run tool
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

graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")

graph.add_conditional_edges("agent", route)

# 🚨 IMPORTANT:
# After tool execution, STOP.
graph.add_edge("tools", END)

app = graph.compile()


# ============================================================
# 8️⃣ RUN THE AGENT
# ============================================================

initial_input = HumanMessage(
    content="What is 25 multiplied by 17?"
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
        print("→ multiply result:", msg.content)

print("\nFinal Answer:")
print(result["messages"][-1].content)

