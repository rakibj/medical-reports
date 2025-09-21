from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from IPython.display import display, Markdown, Image
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import Tool
import json
import uuid
import random
import gradio as gr
from app.src.report_service import ReportService


class ChatAI:
    def __init__(self, report_service):
        self.report_service = report_service

        tool_search_docs = Tool(
            name="search_medical_documents",
            description="Useful for when you need to look for information in the medical documents provided"
                        "The input to this tool should be a fully formed question.",
            func=self.search_medical_documents
        )
        tools = [tool_search_docs]

        llm = ChatOpenAI(model="gpt-4o-mini")
        llm = llm.bind_tools(tools=tools)
        system_prompt = """
            You are a medical health advisor of the user. You have access to the user's medical documents and can use them to provide 
            relevant information to the user. Whenever possible, customize your responses based on the user's medical documents.   
            """
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="messages")
            ])

        self.chain = prompt | llm

        graph_builder = StateGraph(self.State)

        graph_builder.add_node("advisor", self.advisor_node)
        graph_builder.add_node("tools", ToolNode(tools=tools))

        graph_builder.add_edge(START, "advisor")
        graph_builder.add_conditional_edges("advisor", tools_condition, "tools")
        graph_builder.add_edge("tools", "advisor")
        graph_builder.add_edge("advisor", END)

        memory = MemorySaver()
        self.graph = graph_builder.compile(checkpointer=memory)
        self.config = {"configurable": {"thread_id": self.make_thread_id()}}

    def search_medical_documents(self, query: str) -> str:
        """Useful when you need to look for information in the provided medical documents.
        Input should be a fully formed question."""
        return self.report_service.get_context(query)
    
    def chat(self, user_message, history):
        out = self.graph.invoke({"messages": [HumanMessage(content=user_message)]}, config=self.config)
        return out["messages"][-1].content
    
    class State(TypedDict):
        messages: Annotated[List[BaseMessage], add_messages]

    def log_messages(self, messages: List[BaseMessage]) -> None:
        for message in messages:
            if isinstance(message, HumanMessage):
                print(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                print(f"AI: {message.content}")
            elif isinstance(message, SystemMessage):
                print(f"System: {message.content}")


    def advisor_node(self, old_state: State) -> State:
        response = self.chain.invoke(old_state["messages"])
        print("________________________")
        print(self.log_messages(old_state["messages"] + [response]))
        return {"messages": [response]}
    
    def make_thread_id(self) -> str:
        return str(uuid.uuid4())