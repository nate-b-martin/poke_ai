from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import json

# Initialize LLM
llm = ChatOpenAI(model_name="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",base_url="http://localhost:1234/v1", api_key="lm-studio", temperature=0.7)

with open('.venv/src/test_data.json') as f:
    data = json.load(f)


jq_schema = '.[] | {name, moves, stats, abilities}'

# Load, chunk and index the contents of the pokeapi
pokemon_loader = JSONLoader(file_path='.venv/src/test_data.json', jq_schema=jq_schema, text_content=False)
poke_data = pokemon_loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(poke_data)
print(len(splits))

# Initialize embeddings
embeddings = OpenAIEmbeddings(api_key=os.environ.get('OPENAI_API_KEY'))


# Initialze vectore store
vectorstore = Chroma(embedding_function=embeddings)
vectorstore.add_documents(poke_data)


#  Retrieve and generate using the relevent snippts of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


while True:
    user_input = str(input("> "))
    if user_input == "exit":
        break
    response = rag_chain.invoke(user_input)
    print(response)