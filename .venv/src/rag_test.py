import bs4
from langchain import hub
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader 
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.llms.ollama import Ollama
from langchain.memory.buffer_window import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from poke_api import PokeAPI
import os
import json

class PokeBot:
    def __init__(self):
        self.llm = Ollama(model="llama3", base_url="http://localhost:11434", temperature=0.7)
        self.system_prompt = (
            """
            You are a pokemon battle referee and commentating the battle. Using the context provided, please comment on the battle status. Context: {context}. Keep in mind that we want the user to choose what happens after each move alternating between each pokemon. Also please provide a list of four possible options based off the context provided for the user to pick. We also want to keep track of the pokemon's health after each attack.    
            """
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{input}"),
            ]
        )

        self.contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        self.load_pokemon_data()

        loader = DirectoryLoader('/home/nathan/Documents/Projects/poke_ai/.venv/src/pokemon_data', glob="**/*.json", loader_cls=TextLoader)
        self.document = loader.load()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.texts = self.text_splitter.split_documents(self.document)
        self.embedding = OllamaEmbeddings(model="llama3", base_url="http://localhost:11434",temperature=0.7)
        self.vector_db = Chroma.from_documents(documents=self.texts, embedding=self.embedding)
        self.retriever = self.vector_db.as_retriever()
        self.retriever.search_kwargs = {"k": 2}
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, self.contextualize_q_prompt
        )
        self.chat_history = []
        self.question_answer_chain = create_stuff_documents_chain(llm=self.llm, prompt=self.qa_prompt)
        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, self.question_answer_chain)

    def process_llm_response(self, response):
        print(f'\n\nAnswer: {response["answer"]}')
        # print(f"\n\n{response['context'][0]}")
        # print('\n\nSources:')
        # for metadata in response['context']:
        #     print(metadata.metadata)

    def answer_question(self, question):
        llm_response = self.rag_chain.invoke({"input":question, "chat_history": self.chat_history})
        self.chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=llm_response['answer']),
        ])
        self.process_llm_response(llm_response)

    def load_pokemon_data(self):
        pokemon_one = str(input("Pokemon 1: "))
        pokemon_two = str(input("Pokemon 2: "))
        api = PokeAPI()
        pokemon_data_list = [api.get_pokemon(pokemon_one), api.get_pokemon(pokemon_two)]

        current_dir = os.path.dirname(__file__)
        test_data_dir = os.path.join(current_dir, "pokemon_data")
        os.makedirs(test_data_dir, exist_ok=True)
        with open(os.path.join(test_data_dir, "pokemon_data_list.json"), "w") as f:
            json.dump(pokemon_data_list, f, indent=4)


    def run(self):
        self.answer_question("let the battle begin!")
        while True:
            user_input = str(input("> "))
            if user_input == "exit":
                break
            self.answer_question(user_input)

if __name__ == "__main__":
    PokeBot().run()

