from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore

load_dotenv()
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic

INDEX_NAME = "doc-index"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


def run_llm(query: str):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatAnthropic(model_name="claude-haiku-4-5-20251001", temperature=0)

    # Prompt for Retrieval-QA chat (local prompt, no Hub dependency)
    retrieval_qa_chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer any questions based solely on the context below. If the answer isn't in the context, say you don't know."),
        ("human", "Question: {input}\n\nContext:\n{context}")
    ])

    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    qa = create_retrieval_chain(
        retriever=docsearch.as_retriever(),
        combine_docs_chain=stuff_documents_chain
    )

    result = qa.invoke(input={"input": query})
    return result


if __name__=="__main__":
    res = run_llm(query="What is a Chain in LangChain?")
    print(res["answer"])