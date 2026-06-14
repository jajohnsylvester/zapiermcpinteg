import streamlit as st
import asyncio
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage

# --- PRODUCTION CONFIGURATION ---
# 1. Your exposed public Ngrok tunnel pointing to your local Ollama endpoint
OLLAMA_ENDPOINT = "https://evident-lens-surpass.ngrok-free.dev"
MODEL_NAME = "llama3.2"  # This automatically runs the tool-capable 3B variant

# 2. Your unique hosted Zapier MCP URL obtained from mcp.zapier.com
ZAPIER_MCP_SSE_URL = "https://mcp.zapier.com/api/v1/connect?token=ZjgyMTM5ZWQtYjI2MC00ZjY4LWI5OTktYWQzMzY2MTM5MTkwOjNFRWFGN0wrVnFOL04vaHhzQzVhYWtjQVNPZDJUaHVPQ1hJWnpGYmlkY2M9"

st.set_page_config(page_title="Zapier MCP Agent (Llama 3.2)", layout="wide", page_icon="🤖")

# --- UI STYLING & HEADERS ---
st.title("🤖 Zapier MCP Agent Dashboard")
st.caption("Orchestrating system-wide workflows via LangGraph, Model Context Protocol, and Llama 3.2.")
st.markdown("---")

# --- AGENT INITIALIZATION LOGIC ---
@st.cache_resource
def initialize_mcp_agent():
    """
    Connects to the hosted Zapier MCP server over SSE, extracts tool declarations,
    and binds them to the Llama 3.2 LLM instance running on Ollama.
    """
    # 1. Initialize the ChatOllama Model
    # Lower temperature ensures the 3B model adheres strictly to JSON schema tool definitions
    llm = ChatOllama(
        base_url=OLLAMA_ENDPOINT,
        model=MODEL_NAME,
        temperature=0.1 
    )
    
    # 2. Initialize the MultiServerMCPClient using the official dictionary syntax
    mcp_client = MultiServerMCPClient({
        "zapier-server": {
            "transport": "sse",
            "url": ZAPIER_MCP_SSE_URL
        }
    })
    
    # 3. Securely poll tools via an isolated event loop execution block
    async def fetch_tools():
        return await mcp_client.get_tools()

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        mcp_tools = loop.run_until_complete(fetch_tools())
        
        # 4. Tailor structural routing instructions for Llama 3.2
        system_instruction = (
            "You are a highly capable automation assistant powered by Llama 3.2. "
            "We have access to an ecosystem of Zapier tools via the Model Context Protocol (MCP).\n\n"
            "CRITICAL PROTOCOLS:\n"
            "1. Only call an action/tool if the user's intent explicitly requires interacting with an external application.\n"
            "2. If the user greets us, asks a clarification question, or says something conversational, respond directly. "
            "Do NOT hallucinate or trigger tools unnecessarily.\n"
            "3. Always report the status or outcomes of a tool execution clearly to us."
        )
        
        # 5. Compile into a stateful ReAct execution graph
        agent_executor = create_react_agent(
            llm, 
            tools=mcp_tools,
            state_modifier=system_instruction
        )
        return agent_executor, mcp_tools
    except Exception as e:
        st.error(f"⚠️ Failed to connect to Zapier MCP Server: {e}")
        return None, []

# Execute initialization
agent, tools = initialize_mcp_agent()

# --- SIDEBAR DIAGNOSTICS & MANAGEMENT ---
with st.sidebar:
    st.header("⚙️ Core Infrastructure")
    st.success(f"Ollama Node: `{OLLAMA_ENDPOINT}`")
    st.info(f"Active Model: `{MODEL_NAME}`")
    
    st.subheader("🛠️ Loaded Zapier Tools")
    if tools:
        st.write(f"Successfully integrated **{len(tools)}** tools:")
        for tool in tools:
            with st.expander(f"🔹 {tool.name}"):
                st.caption(tool.description)
    else:
        st.error("No tools found. Check if your Zapier MCP URL is correct and you have authorized tools in the dashboard.")

# --- CHAT BUFFER INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Re-render
