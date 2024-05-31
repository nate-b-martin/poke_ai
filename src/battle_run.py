from langchain_core.messages.base import BaseMessage
from poke_api import PokeAPI
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# GLOBALS
store = {}
config = {"configurable": {"session_id": "abc2"}}

model = ChatOpenAI(model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF", base_url="http://localhost:1234/v1", api_key="lm-studio")
# model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

parser = StrOutputParser()

# Functions
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(model, get_session_history)
response: BaseMessage = with_message_history.invoke(
    [HumanMessage(content="Hello, I'm Nathan?")],
    config=config
    )

config = {"configurable": {"session_id": "abc3"}}
response: BaseMessage = with_message_history.invoke(
    [HumanMessage(content="What is my name?")],
    config=config
    )

print(response.content)

print(f"store: {store}")

