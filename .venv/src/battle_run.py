from langchain_core.messages.base import BaseMessage
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI

import openai as OpenAI
from poke_api import PokeAPI
from battle_model import BattleModel


# Initialize the model
model = BattleModel(model_name="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF", api_key="lm-studio", temperature=0.7)

# response: BaseMessage = model.with_message_history.invoke(
#     {"messages":[HumanMessage(content="Lets Battle.")], "pokemon_one": "charizard", "pokemon_two": "dragonite"},
#     config=model.config
#     )

# Chat history
chat_history = ChatMessageHistory()

# initial response
user_input = " "
for r in model.with_message_history.stream(
    {"messages": chat_history.messages,
        "pokemon_one":"charizard",
    "pokemon_two":"dragonite",
    "user_input": user_input}, config=model.config):
    print(r.content, end="")

while True:
    user_input = str(input("> "))

    chat_history.add_user_message(user_input)

    if user_input == "exit":
        break

    for r in model.with_message_history.stream(
        {"messages": chat_history.messages,
         "pokemon_one":"charizard",
        "pokemon_two":"dragonite",
        "user_input": user_input}, config=model.config):
    # Stream responses
        print(r.content, end="")



