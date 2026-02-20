import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings


#loading environmental variables

load_dotenv()

OPENI_API_KEY  = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")


# embeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENI_API_KEY)

# intializing pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(INDEX_NAME)
#
vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)
## LLM

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENI_API_KEY
)

# retriever

retriever = vector_store.as_retriever()

# Query

while True:
    try:

        query = input("\nPlease ask your question or enter (exit) to quit: ")

        if query.strip().lower() == "exit":
            print("Bye for now..!!")
            break

        # get documents

        docs = retriever.invoke(query)

        #create context 

        context = "\n\n".join([doc.page_content for doc in docs])

        # Ask LLM

        prompt = f"Answer the question using this context:\n\n{context}\n\nQuestion: {query}"

        response = llm.invoke(prompt)

        print("\nAnswer")
        print(response.content)
    except Exception as e:
        print("Something went wrong")