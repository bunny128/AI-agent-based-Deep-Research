import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
from typing import Annotated

# Streamlit UI
st.set_page_config(page_title="Deep Research AI Agent ", layout="centered")
st.title(" Deep Research AI Agent")

with st.sidebar:
    st.header("🔧 Configuration")
    groq_api_key = st.text_input("GROQ API Key", type="password")
    tavily_api_key = st.text_input("Tavily API Key", type="password")

user_query = st.text_area("💬 Enter your search query:", height=200)
run_button = st.button("Search")

if run_button:
    if not groq_api_key or not tavily_api_key or not user_query:
        st.warning("Please provide all required inputs in the sidebar and main field.")
    else:
        try:
            # Set API keys
            os.environ["GROQ_API_KEY"] = groq_api_key
            os.environ["TAVILY_API_KEY"] = tavily_api_key

            # Load tools
            api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)
            arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

            api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
            wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

            tavily = TavilySearchResults()
            tools = [arxiv, wiki, tavily]

            # Load LLMs
            llm = ChatGroq(model="qwen-qwq-32b")
            llm_with_tools = llm.bind_tools(tools=tools)
            answer_agent = llm.with_config({
    "system_prompt": "You are a clear and concise research summarizer. Return only the factual information and insights found from tools. Do not reference yourself, do not describe your process, and do not include internal thoughts.",
    "max_tokens": 1000
})


            # Define state schema
            class State(TypedDict):
                messages: Annotated[list[AnyMessage], add_messages]

            # Node definitions
            def tool_calling_llm(state: State):
                return {"messages": [llm_with_tools.invoke(state["messages"])]}

            def answer_drafting_agent(state: State):
                return {"messages": [answer_agent.invoke(state["messages"])]}

            # Graph building
            builder = StateGraph(State)
            builder.add_node("tool_calling_llm", tool_calling_llm)
            builder.add_node("tools", ToolNode(tools))
            builder.add_node("answer_drafting_agent", answer_drafting_agent)

            builder.add_edge(START, "tool_calling_llm")
            builder.add_conditional_edges("tool_calling_llm", tools_condition)
            builder.add_edge("tools", "answer_drafting_agent")
            builder.add_edge("answer_drafting_agent", END)

            graph = builder.compile()

            # Run the query
            with st.spinner(" Researching..."):
                result = graph.invoke({"messages": user_query})

            import re

            def remove_think_blocks(text):
                return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            
            import re

            def remove_think_blocks(text):
                return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    
            # Display results
            st.success("Research Complete")

            st.markdown("##Final Answer")
            for msg in result["messages"]:
                clean_msg = remove_think_blocks(msg.content)
                with st.container():
                    st.markdown(f"""
                    <div style="background-color:#1e1e1e;padding:1.2em;border-radius:12px;margin-bottom:15px;">
                        <p style="color:#e0e0e0;font-size:1.05rem;">{clean_msg}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
            # Raw message viewer
            with st.expander("🧪 Show Raw Tool Outputs"):
                for i, m in enumerate(result["messages"]):
                    st.markdown(f"**Message {i+1}:**")
                    st.code(m.pretty_repr(), language="json")


        except Exception as e:
            st.error(f" An error occurred: {str(e)}")
