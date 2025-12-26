from typing import List, Dict, Any, TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END

load_dotenv()

INDEX_NAME = "glorious-elm"
EMBEDDING_MODEL = "text-embedding-3-large"

# Define the state for the graph
class GraphState(TypedDict):
    input: str
    chat_history: List[Any]
    context: List[Document]
    answer: str

def run_llm(query: str, chat_history: List[Dict[str, Any]] = None):
    if chat_history is None:
        chat_history = []
        
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, dimensions=1024)
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatAnthropic(model_name="claude-haiku-4-5-20251001", temperature=0)

    def retrieve(state: GraphState):
        """
        Retrieve documents relevant to the question.
        """
        print("---RETRIEVE---")
        question = state["input"]
        history = state.get("chat_history", [])

        # If there is chat history, we want to rephrase the question
        if history:
            rephrase_prompt = ChatPromptTemplate.from_messages([
                ("system", "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ])
            rephrase_chain = rephrase_prompt | chat
            
            # Convert history to BaseMessages if needed (handling the tuples from main.py)
            formatted_history = []
            for item in history:
                if isinstance(item, tuple):
                    role, content = item
                    if role == "human":
                        formatted_history.append(HumanMessage(content=content))
                    elif role == "ai":
                        formatted_history.append(AIMessage(content=content))
                elif isinstance(item, BaseMessage):
                    formatted_history.append(item)
            
            # Rephrase
            if formatted_history:
                response = rephrase_chain.invoke({"question": question, "history": formatted_history})
                question = response.content

        # Retrieve documents
        docs = docsearch.similarity_search(question)
        return {"context": docs}

    def generate(state: GraphState):
        """
        Generate answer using the context.
        """
        print("---GENERATE---")
        question = state["input"]
        context = state["context"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer any questions based solely on the context below. If the answer isn't in the context, say you don't know.\n\nContext:\n{context}"),
            ("human", "Question: {question}")
        ])
        
        chain = prompt | chat
        response = chain.invoke({"question": question, "context": context})
        return {"answer": response.content}

    # Build the graph
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    app = workflow.compile()

    # Invoke the graph
    inputs = {"input": query, "chat_history": chat_history}
    result = app.invoke(inputs)

    return {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"],
    }

if __name__=="__main__":
    res = run_llm(query="Content on LangChain concepts such as Chains, Agents, Tools, and core usage examples")
    print(res["result"])