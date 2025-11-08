import asyncio
import os
import ssl
from operator import truediv
from typing import Any, Dict, List
from langchain_huggingface import HuggingFaceEmbeddings

import certifi
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_anthropic import ChatAnthropic
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import Colors, log_error, log_header, log_info, log_success, log_warning

load_dotenv()

# Configure SSL context to use the CA bundle from certifi
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vector_store = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# use pinecone instead of local vector store
# vector_store = PineconeVectorStore(
#     index_name="doc-index", embedding=embeddings
# )

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def index_documents_async(documents: List[Document], batch_size: int = 50) -> None:
    """Indexes documents into the vector store asynchronously."""
    log_header("VECTOR STORE INDEXING PHASE")
    log_info(
        f"VectorStore: Indexing {len(documents)} documents into Pinecone Vector Store",
        Colors.DARKCYAN,
    )
    
    batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]

    log_info(f"VectorStore: Indexing {len(batches)} batches of {batch_size} documents", Colors.YELLOW)
    semaphore = asyncio.Semaphore(5)  # Max 5 concurrent batches

    async def add_batch(batch: List[Document], batch_num: int) -> bool:
        async with semaphore:
            try:
                await vector_store.aadd_documents(batch)
                log_success(
                    f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} to  Vector Store"
                )
            except Exception as e:
                log_error(f"VectorStore indexing failed to add batch {batch_num} - {e}")
                return False
            return True

    # Process batches concurrently
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate (batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore: All batches indexed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing failed! Processed ({successful}/{len(batches)}) batches successfully!"
        )


async def main():
    """Main async function to run the entire process."""
    log_header("Starting documentation-helper ingestion process")

    log_info(
        "TavilyCrawl: Starting to Crawl documentation",
        Colors.PURPLE,
    )

    # Crawl Documentation site
    res = tavily_crawl.invoke(
        {
            "url": "https://docs.langchain.com",
            "max_depth": 5,
            "extract_depth": "advanced",
            "instructions": "Content on AI Agents"
        }
    )

    # Filter out results that don't have valid raw_content
    all_docs = [
        Document(page_content=result["raw_content"], metadata={"source": result['url']})
        for result in res["results"]
        if result.get("raw_content") is not None and result["raw_content"].strip()
    ]

    log_success(f"TavilyCrawl: Crawled {len(all_docs)} documents successfully")

    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info (
        f"Text Splitter: Processing {len(all_docs)} documents with 4000 chunck size and 200 overlap",
        Colors.YELLOW,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200,
    )

    split_docs = text_splitter.split_documents(all_docs)
    log_success(f"Text Splitter: Created {len(split_docs)} chunks from {len(all_docs)} document chunks successfully")

    # Process documents asynchronously
    await index_documents_async(split_docs, batch_size=500)

    log_header("PIPELINE COMPLETE")
    log_success("Document Pipeline Completed Successfully")
    log_header("PIPELINE COMPLETE")
    log_success("Document Pipeline Completed Successfully")

    log_info(
        f"{Colors.BOLD}Summary:{Colors.END} Crawled {len(all_docs)} documents, created {len(split_docs)} chunks, and indexed them successfully.",
        Colors.GREEN)

if __name__ == "__main__":
    asyncio.run(main())
