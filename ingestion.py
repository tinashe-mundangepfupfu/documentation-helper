import asyncio
import os
import ssl
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

embeddings = HuggingFaceEmbeddings()


# chroma = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
vector_store = PineconeVectorStore(
    index_name="langchain-doc-index", embedding=embeddings
)

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


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

    all_docs = [Document(page_content=result["raw_content"], metadata={"source": result['url']}) for result in res["results"]]

    log_success(f"TavilyCrawl: Crawled {len(all_docs)} documents successfully")


if __name__ == "__main__":
    asyncio.run(main())
