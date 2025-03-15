import os
from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, OpenAIEmbeddings

def run_llm(query:str):
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-ada-002",
        azure_deployment="subhamoy-text-embeddings"
    )

    docSearch = PineconeVectorStore(index_name="langchain-docs-index", embedding=embeddings)
    chat = AzureChatOpenAI(
        temperature=0,
        openai_api_key=os.environ['OPENAI_API_KEY'],
        openai_api_version="2024-08-01-preview",  # Specify API version
        azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
        azure_deployment="gpt-4o",
        model="gpt-4o"
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    qa = create_retrieval_chain(retriever=docSearch.as_retriever(), 
                                combine_docs_chain=stuff_documents_chain)
    
    result = qa.invoke(input={"input": query})

    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_document": result["context"]
    }

    return new_result


if __name__ == "__main__":
    query = "What is langchain chain?"
    result = run_llm(query)
    print(result["result"])



