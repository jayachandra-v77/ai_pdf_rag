import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


#load environmental variables

load_dotenv()


OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY= os.getenv("PINECONE_API_KEY")
INDEX_NAME= os.getenv("INDEX_NAME")


# load documents

loader = PyPDFLoader(".\data\Amazon_report.pdf")

documents = loader.load()

print(f"Total documents : {len(documents)}")

# print((documents[21].page_content[:250]))


#chuncking

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100
)

docs = text_splitter.split_documents(documents=documents)

print(f"Total documents after chunking: {len(docs)}")

# print(docs[27].page_content[:270])



#embeddings 

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)


#initialing pinecone

pc= Pinecone(api_key=PINECONE_API_KEY)


#create index 

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name = INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud = "aws",
            region = "us-east-1"
    )
)


print("Index_already_exit")


# Creating vector store

vector_store = PineconeVectorStore.from_documents(
    index_name=INDEX_NAME,
    embedding=embeddings,
    documents=docs
)

print("documents are uploaded into vector store")
