import bs4
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize LLM
llm = ChatOpenAI(model_name="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",base_url="http://localhost:1234/v1", api_key="lm-studio", temperature=0.7)

# Load, chunk and index the contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post_title", "post-header")
        )
    ),
)


# Load, chunk and index the contents of the pokeapi
# pokemon_data_loader = JSONLoader(
#     jq_schema="[.pokemon[] | .name + \" \" + .description] | join(\"\n\")",
#     file_path="src/pokemon_data.json",
# )

with open("E:\\Projects\\poke_ai\\.venv\\src\\test_data.txt") as f:
    pokemon_data = f.read()

pokemon_data_loader = TextLoader(
    file_path="E:\\Projects\\poke_ai\\.venv\\src\\test_data.txt",
    encoding="utf-8",
    )

docs = loader.load()
poke_data = pokemon_data_loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
poke_splits = text_splitter.split_text(poke_data)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

vectorstore.add_texts(texts=poke_splits)

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

response = rag_chain.invoke("What is task decomposition?")
print(response)

response = rag_chain.invoke("What is pikachu's moves?")
print(response)

response = rag_chain.invoke("What is pikachu's hp?")
print(response)