import time
from dotenv import load_dotenv
load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Use Azure OpenAI embeddings
embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-ada-002",
        azure_deployment="subhamoy-text-embeddings"
    # dimensions: Optional[int] = None, # Can specify dimensions with new text-embedding-3 models
    # azure_endpoint="https://<your-endpoint>.openai.azure.com/", If not provided, will read env variable AZURE_OPENAI_ENDPOINT
    # api_key=... # Can provide an API key directly. If missing read env variable AZURE_OPENAI_API_KEY
    # openai_api_version=..., # If not provided, will read env variable AZURE_OPENAI_API_VERSION
    )

def ingest_docs():
    # Add your document ingestion logic here
    loader = ReadTheDocsLoader("C:\\Users\\sburman\\Projects\\Py-DocumentationReader\\documentation-helper\\langchain-docs\\api.python.langchain.com\\en\\latest", encoding='utf-8')
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source":new_url})

    print(f"Going to add {len(documents)} documents to the vector store")

     # Process documents in chunks
    chunk_size = 10  # Number of documents to process in each batch
    wait_time = 5  # Wait time in seconds between batches

    for i in range(0, len(documents), chunk_size):
        batch = documents[i:i + chunk_size]
        PineconeVectorStore.from_documents(batch, embeddings, index_name="langchain-docs-index")
        print(f"Added {len(batch)} documents to the vector store")
        time.sleep(wait_time)

    print("All documents added to the vector store")
    # Process the documents as needed
    

if __name__ == "__main__":
    ingest_docs()