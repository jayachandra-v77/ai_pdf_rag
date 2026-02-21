import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

#load environment variables

#load environmental variables

load_dotenv()


OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY= os.getenv("PINECONE_API_KEY")
INDEX_NAME= os.getenv("INDEX_NAME")


#embeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

#initializing pinecone

pc = Pinecone(api_key=PINECONE_API_KEY)


#vector store 

vector_store = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings
)

#initializing llm

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY
)

#Creating retriever

retriever = vector_store.as_retriever()


# creating prompt and context

while True:

    question = input("\nPlease ask your question or type(exit) to quit: ")
    try: 

        if question.lower().strip() == "exit":
            print("Bye for now....!!")
            break

        docs = retriever.invoke(question)

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt =( f""" You are the best pdf reader and answer the below context
            If you don't know the answer tell that I don't know
            context:
            {context}
            question:
            {question}
            """
        )

        response = llm.invoke(prompt)

        print(response.content)

    except Exception as e:
        print("Something went wrong")
