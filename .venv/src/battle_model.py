
from langchain_core.messages.base import BaseMessage
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI

class BattleModel:
    def __init__(self, model_name, api_key, temperature=0.7):
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.model = ChatOpenAI(model=self.model_name, base_url="http://localhost:1234/v1", api_key=self.api_key, temperature=self.temperature)
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You are the number one pokemon battle referee and have the knowledge of all pokemon. You offer detailed play by play commentary of pokemon battles in a clear, impartial and engaging way. You will keep track of each pokemons health and attack and once a pokemon reaches 0 health you will end the battle. Please keep your responses focused on the simulated battle at hand. After each move you will ask the user what you want to do next providing four options for their next move alternating between each pokemon giving them an option to do something.

                Here are the pokemon that will be battling:
                {pokemon_one}, {pokemon_two}
                """
            ),
            ("human", "{user_input}"),
            MessagesPlaceholder(variable_name="messages")
        ])
        self.chain = self.prompt | self.model
        self.parser = StrOutputParser()
        self.store = {}
        self.config = {"configurable": {"session_id": "abc2"}}
        self.with_message_history = RunnableWithMessageHistory(self.chain, self.get_session_history, input_messages_key="messages")

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]