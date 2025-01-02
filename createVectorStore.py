import asyncio
import os
from uuid import uuid4
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = "us-east-1"  # Update to your Pinecone environment
index_name = "brainlox-data"

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)

# Create the Pinecone index if it doesn't exist
if index_name not in pc.list_indexes():
    pc.create_index(
        name=index_name,
        dimension=768,  # Match the dimensions of the embedding model
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(index_name)

# Extract data
async def collect_data():
    urls = ["https://brainlox.com/courses/category/technical"]
    loader = PlaywrightURLLoader(urls=urls, remove_selectors=["header", "footer"])
    data = await loader.aload()
    return [Document(page_content=doc.page_content) for doc in data]

# Split documents into chunks and upsert embeddings
async def process_and_upsert():
    documents = await collect_data()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for doc in documents:
        chunks.extend(text_splitter.split_text(doc.page_content))

    # Generate embeddings
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    embeddings = embed.embed_documents(chunks)

    # Upsert embeddings in batches
    BATCH_SIZE = 100

    def create_batches(embeddings, chunks, batch_size=BATCH_SIZE):
        for i in range(0, len(embeddings), batch_size):
            yield [
                (f"chunk_{i+j}", embeddings[i+j], {"text": chunks[i+j]})
                for j in range(min(batch_size, len(embeddings) - i))
            ]

    for batch in create_batches(embeddings, chunks):
        index.upsert(vectors=batch)

    print("Embeddings successfully upserted to Pinecone.")

# Run the process
asyncio.run(process_and_upsert())

