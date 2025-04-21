
# LangGraph Dual-Agent Research Assistant (Optimized)


# Load Tools
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

# Tavily tool
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

tavily = TavilySearchResults()
tools = [arxiv, wiki, tavily]


# Load LLM
from langchain_groq import ChatGroq
llm = ChatGroq(model="qwen-qwq-32b")
llm_with_tools = llm.bind_tools(tools=tools)

#prompt
answer_agent = llm.with_config({"system_prompt": "You are a helpful research assistant. Summarize and explain the findings clearly and concisely."})



# State Schema
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from typing import Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# LangGraph Setup
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# Node 1: Tool Calling LLM
def tool_calling_llm(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Node 2: Final Answer Synthesizer
def answer_drafting_agent(state: State):
    return {"messages": [answer_agent.invoke(state["messages"])]}

# Build Graph
builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))
builder.add_node("answer_drafting_agent", answer_drafting_agent)

# Graph Edges
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", "answer_drafting_agent")
builder.add_edge("answer_drafting_agent", END)

# Compile Graph
graph = builder.compile()

# Example Run
query = "Tell me about spacex "

messages = graph.invoke({"messages": query})
for msg in messages["messages"]:
    msg.pretty_print()
