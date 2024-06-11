
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from poke_api import PokeAPI
import os
import json

class BattleModel:
    def __init__(self, model_name="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF", api_key="lm-studio",base_url="http://localhost:1234/v1", temperature=0.7):
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.pokemon_one = str(input("First Pokemon: "))
        self.pokemon_two = str(input("Second Pokemon: "))
        self.load_pokemon_data(self.pokemon_one, self.pokemon_two)
        self.vector_db = self.vector_db()
        self.retriever = self.initialize_retriever(self.vector_db)
        # self.model = ChatOpenAI(model=self.model_name, base_url=self.base_url, api_key=self.api_key, temperature=self.temperature)
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You are the number one pokemon battle referee and have the knowledge of all pokemon. You offer detailed play by play commentary of pokemon battles in a clear, impartial and engaging way. You will keep track of each pokemon's health and attack and once a pokemon reaches 0 health you will end the battle. Please keep your responses focused on the simulated battle at hand. After each move you will ask the user what you want to do next providing four options for their next move alternating between each pokemon giving them an option to do something.

                We will begin the battle and the pokemon that are battle are provided in the context provided: {context}
                """
            ),
            ("human", "{input}"),
            # MessagesPlaceholder(variable_name="messages")
        ])
        # self.chain = self.prompt | self.model
        # self.parser = StrOutputParser()
        # self.store = {}
        # self.config = {"configurable": {"session_id": "abc2"}}
        # self.with_message_history = RunnableWithMessageHistory(self.chain, self.get_session_history, input_messages_key="messages")

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def load_pokemon_data(self, pokemon_one, pokemon_two):
        api = PokeAPI()
        pokemon_data_list = [api.get_pokemon(pokemon_one), api.get_pokemon(pokemon_two)]

        current_dir = os.path.dirname(__file__)
        test_data_dir = os.path.join(current_dir, "pokemon_data")
        os.makedirs(test_data_dir, exist_ok=True)
        with open(os.path.join(test_data_dir, "pokemon_data_list.json"), "w") as f:
            json.dump(pokemon_data_list, f, indent=4)
    
    def vector_db(self):
        loader = DirectoryLoader('/home/nathan/Documents/Projects/poke_ai/.venv/src/pokemon_data', glob="**/*.json", loader_cls=TextLoader)
        document = loader.load()
        # splitting the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(document)

        # CREATE The Chroma VectorStore
        # embed and store the texts
        # supplying a persist_directory will store the embeddings on disk
        persist_directory = 'db'

        ## here we are using OpenAI embeddings but in the future we will swap out to local embeddings 
        embedding = OpenAIEmbeddings()
        vector_db = Chroma.from_documents(documents=texts, embedding=embedding,persist_directory=persist_directory)
        
        # Now we can load the persisted database from disk, and use it as normal
        # vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding)

        return vector_db

    def initialize_retriever(self, vector_db):
        retriever = vector_db.as_retriever()
        retriever.search_kwargs = {"k": 2}
        return retriever

    def process_llm_response(self, response):
        print(f'\n\nAnswer: {response["answer"]}')
        print(f"\n\n{response['context'][0]}")
        # print('\n\nSources:')
        # for metadata in response['context']:
        #     print(metadata.metadata)

if __name__ == "__main__":
    model = BattleModel()

    question_answer_chain = create_stuff_documents_chain(llm=OpenAI(), prompt=model.prompt)
    chain = create_retrieval_chain(model.retriever, question_answer_chain)
    llm_response = chain.invoke({"input": "What pokemon are going to battle?"})
    model.process_llm_response(llm_response)