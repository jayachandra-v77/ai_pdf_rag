import os 
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


#loading environmental variables

load_dotenv()

OPENI_API_KEY  = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")


# Loading pdfs

loader = PyPDFLoader(".\data\Customer Shopping Behavior Analysis.pdf")

documents = loader.load()

print(f"Total documents:{len(documents)}")


#chunking

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100
)


docs = text_splitter.split_documents(documents=documents)


print(f"Total doucments after chunking: {(len(docs))}")


#embeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


#intializing pinecone

pc = Pinecone(api_key=PINECONE_API_KEY)

#If index name is present

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name = INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )

    )


#Creating vector store

vector_store = PineconeVectorStore.from_documents(
    index_name=INDEX_NAME,
    embedding=embeddings,
    documents=docs
)

print("Succesfully stored data in vector store")