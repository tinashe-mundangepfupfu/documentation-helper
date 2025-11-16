from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic

load_dotenv()

INDEX_NAME = "glorious-elm"
EMBEDDING_MODEL = "text-embedding-3-large"



def run_llm(query: str, chat_history:List[Dict[str, Any]]):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, dimensions=1024)
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatAnthropic(model_name="claude-haiku-4-5-20251001", temperature=0)

    # Prompt for Retrieval-QA chat (local prompt, no Hub dependency)
    retrieval_qa_chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer any questions based solely on the context below. If the answer isn't in the context, say you don't know."),
        ("human", "Question: {input}\n\nContext:\n{context}")
    ])

    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(),prompt=rephrase_prompt
    )

    qa = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=stuff_documents_chain
    )

    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"],
    }
    return new_result


if __name__=="__main__":
    res = run_llm(query="Content on LangChain concepts such as Chains, Agents, Tools, and core usage examples")
    print(res["result"])