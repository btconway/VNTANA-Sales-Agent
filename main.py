import streamlit as st
import os
from dotenv import find_dotenv, load_dotenv
import anthropic
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatAnthropic
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler 
from langchain.agents import initialize_agent, AgentType
from langchain import PromptTemplate
from langchain.cache import SQLiteCache
from langchain.agents import Tool
from langchain.utilities import SerpAPIWrapper
from langchain.callbacks import tracing_enabled
import langchain

# Set page config
st.set_page_config(page_title="VNTANA Sales", page_icon=":robot:")
st.cache_resource.clear()

LANGCHAIN_TRACING = tracing_enabled(True)

# Load environment variables
load_dotenv(find_dotenv())
anthropic.api_key = os.getenv("ANTHROPIC_API_KEY")

os.environ["SERPAPI_API_KEY"] = "0e54b7e6be59703acd6e30056984a57b38124707e987016a9998b5b0c166f571"

# Initialize SQLite cache
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

# Initialize the memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the chat model
llm = ChatAnthropic(
    model="claude-2.0",
    temperature=0.3,
    max_tokens_to_sample=75000,
    streaming=True,
    verbose=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

search = SerpAPIWrapper()
tools = [
    Tool(
        name = "Current Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world"
    ),
]

# Initialize the agent
agent_chain = initialize_agent(
    tools=tools, 
    llm=llm, 
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
    verbose=True, 
    memory=memory,
    handle_parsing_errors=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

@st.cache_data()  # Updated cache function
def read_prompt_template():
    with open('Anthropic_Prompt.txt', 'r') as file:
        return file.read()

template = read_prompt_template()
prompt = PromptTemplate(input_variables=["chat_history", "input","agent_scratchpad"], template=template)

class StreamlitUI:
    def __init__(self, agent_chain, prompt):
        self.agent_chain = agent_chain
        self.prompt = prompt

    def run(self):
        st.header("VNTANA Sales")

        # Initialize session state variables
        if 'generated' not in st.session_state:
            st.session_state.generated = []
        if 'past' not in st.session_state:
            st.session_state.past = []
        if 'agent_scratchpad' not in st.session_state:
            st.session_state['agent_scratchpad'] = []

        user_input = self.get_text()
       
        if user_input:
            prompt_text = self.prompt.format(chat_history=st.session_state["past"], input=user_input, agent_scratchpad=st.session_state["agent_scratchpad"])
            output = self.agent_chain.run(input=prompt_text)
            print("Language Model Output:", output)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)

        if st.session_state["generated"]:
            for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                with st.chat_message("assistant"):
                    st.write(f"AI: {st.session_state['generated'][i]}")
                with st.chat_message("user"):
                    st.write(f"Human: {st.session_state['past'][i]}")

    def get_text(self):
        return st.chat_input("You: ")

ui = StreamlitUI(agent_chain, prompt)
ui.run()
