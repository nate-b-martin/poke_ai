import getpass
import os
from typing import Dict, List, Tuple

# os.environ["OPENAI_API_KEY"] = getpass.getpass()

from langchain_core.messages.base import BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables.base import RunnableSerializable
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from typing import List

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langserve import add_routes

model = ChatOpenAI(model="gpt-3.5-turbo")
# model = ChatOpenAI(model="FaradayDotDev/llama-3-8b-Instruct-GGUF")

# messages = [
#     SystemMessage(content="Translate the following from English into Italian"),
#     HumanMessage(content="talk to you later!"),
# ]

# messages_two = [
#     SystemMessage(content="Translate the following from English into Italian"),
#     HumanMessage(content="love you"),
# ]

# parser = StrOutputParser()

# chain = model | parser
# print(chain.invoke(messages))

