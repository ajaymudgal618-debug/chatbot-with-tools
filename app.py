import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os
from dotenv import load_dotenv
 
load_dotenv()
 
## Arxiv and Wikipedia Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
 
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
 
search = DuckDuckGoSearchRun()
 
tools = [search, arxiv, wiki]
 
st.title("🔎 LangChain - Chat with search")
"""
In this example, we're using LangGraph with InMemorySaver to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""
 #-----------------------------------------------------------------------------------------------------------------------
## Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")
thread_id = st.sidebar.text_input("Session ID", value="default_session")


if not api_key:
    st.warning("Please enter your Groq API Key in the sidebar.")
    st.stop()
 
## InMemorySaver persists across Streamlit reruns via session_state
if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = InMemorySaver()
config = {"configurable": {"thread_id": thread_id} }
  
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="openai/gpt-oss-20b",
    streaming=True
)
model_with_tools=llm.bind_tools(tools)
 
from langgraph.prebuilt import create_react_agent 
search_agent = create_react_agent(
    model_with_tools,
    tools,             
    prompt=SystemMessage(content="You are a helpful assistant who can search the web, arxiv and wikipedia to answer questions."),
    checkpointer=st.session_state.checkpointer
)

#-------------------------------------------------------------------------------------------------------------------
 
user_input = st.text_input("Your question:")
if user_input:
    try:
        response = search_agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )
        
        if response["messages"][-1].content:
            st.write(response["messages"][-1].content)

    except ValueError:
        ## corrupt history — reset karo aur dobara try karo
        del st.session_state.checkpointer
        st.session_state.checkpointer = InMemorySaver()
        st.warning("Chat history reset ho gayi — dobara question karo.")
        st.rerun()


 #.venv\Scripts\python.exe -m streamlit run c:\Users\ajaym\OneDrive\Desktop\project_langchain\here1\app.py                                                         
