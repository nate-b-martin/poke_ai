from langchain_community.vectorstores import Chroma
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA 
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

system_prompt = (
    """
    You are a pokemon battle referee and commentating the battle. Using the context provided, please comment on the battle status. Context: {context}. Keep in mind that we want the user to choose what happens after each move alternating between each pokemon. Also please provide a list of four possible options based off the context provided for the user to pick. We also want to keep track of the pokemon's health after each attack.
    """
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

store = {}
config = {"configurable": {"session_id": "abc2"}}

client = Ollama(model="llama3", base_url="http://localhost:11434", temperature=0.7)

# Load and process the text files
loader = DirectoryLoader('/home/nathan/Documents/Projects/poke_ai/.venv/src/pokemon_data', glob="**/*.json", loader_cls=TextLoader)

# loader = DirectoryLoader('/home/nathan/Documents/Projects/poke_ai/.venv/src/test data ', loader_cls=TextLoader)
document = loader.load()

# splitting the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(document)

# CREATE The Chroma VectorStore
# embed and store the texts
# supplying a persist_directory will store the embeddings on disk
persist_directory = 'db'

## here we are using OpenAI embeddings but in the future we will swap out to local embeddings 
# embedding = OpenAIEmbeddings()
embedding = OllamaEmbeddings(model="llama3", base_url="http://localhost:11434",temperature=0.7)
vector_db = Chroma.from_documents(documents=texts, embedding=embedding,persist_directory=persist_directory)

# Now we can load the persisted database from disk, and use it as normal

# make a retriever
retriever = vector_db.as_retriever()
retriever.search_kwargs = {"k": 2}

# Make a chain

## cite sources
def process_llm_response(response):
    print(f'\n\nAnswer: {response["answer"]}')
    # print(f"\n\n{response['context'][0]}")
    # print('\n\nSources:')
    # for metadata in response['context']:
    #     print(metadata.metadata)

# create the cahin to answer questions
question_answer_chain = create_stuff_documents_chain(llm=client, prompt=prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)
llm_response = chain.invoke({"input": "let the battle begin"})
process_llm_response(llm_response)

while True:
    user_input = str(input("> "))
    if user_input == "exit":
        break
    # chain = create_retrieval_chain(retriever, question_answer_chain)
    llm_response = chain.invoke({"input": user_input})
    process_llm_response(llm_response)

