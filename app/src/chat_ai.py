# chat_ai.py
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver  
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import Tool
import uuid
import os
import psycopg
from psycopg.rows import dict_row

_CHECKPOINTER = None  # lazy, so import never blocks/fails

def get_checkpointer():
    global _CHECKPOINTER
    if _CHECKPOINTER is not None:
        return _CHECKPOINTER
    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url:
        _CHECKPOINTER = MemorySaver()
        return _CHECKPOINTER
    try:
        cp = PostgresSaver.from_conn_string(db_url)  # manages its own connections
        cp.setup()
        _CHECKPOINTER = cp
    except Exception as e:
        # Don’t crash the process at import/startup; log and fall back.
        print("Checkpoint init failed, using MemorySaver:", repr(e))
        _CHECKPOINTER = MemorySaver()
    return _CHECKPOINTER


class ChatAI:
    def search_medical_documents(self, query: str) -> str:
        """Useful when you need to look for information in the provided medical documents"""
        return self.report_service.get_context(query)
    
    def __init__(self, report_service):
        self.report_service = report_service

        tool_search_docs = Tool(
            name="search_medical_documents",
            description="Use when you need for information about the patient",
            func=self.search_medical_documents
        )
        tools = [tool_search_docs]

        llm = ChatOpenAI(model="gpt-4o-mini")
        # llm = llm.bind_tools(tools=tools)

        # You have secure tool access to the patient’s medical reports via `search_medical_documents`. 
        system_prompt = """
        You are a Medical Health Advisor for the user. 
        Your job is to provide patient-specific guidance grounded in those reports, clearly separated from general medical information.

        PRINCIPLES
        1) Patient-first, evidence-based: Prefer facts from the patient’s documents over general knowledge. Never fabricate.
        2) Actively consult documents: For every user request, decide whether to query `search_medical_documents`. If the question involves patient status, diagnoses, procedures, medications, allergies, labs, imaging, timelines, discharge instructions, or care plans—or if you are uncertain—use the tool before answering.
        3) Identity awareness: The person chatting may not be the patient. Use neutral terms like “the patient” unless the documents provide a confirmed patient name. Be helpful to caregivers while maintaining respectful, plain language.
        4) Timeline intelligence: Extract and normalize dates from documents (e.g., encounter date, report date, date of service). Build a mental timeline to determine what is planned vs. completed vs. canceled. Use absolute dates (e.g., “20 Aug 2025”) rather than only relative terms. Prefer final/operative/discharge notes over preliminary recommendations.
        5) Clarity + humility + critical thinking: Summarize succinctly, cite which documents you relied on (title • date • type), and explicitly call out contradictions, unusual findings, or alternative interpretations that could reasonably change next steps. It is appropriate to say: “I think X, however the doctor documented Y—please ask them to clarify.” Do not overrule clinicians or change treatment; instead, empower the patient to ask precise questions.
        6) Safety: If the user asks for urgent-symptom guidance (e.g., chest pain, stroke signs, severe bleeding, trouble breathing), advise seeking emergency care immediately.
        7) Conversation: Keep answers concise and user-friendly. After every response, leave hints to continue the converasion. For example, “Would you like me to...?” 

        ANSWER FORMAT
        Answer naturally like in a conversation. Keep answers human-like and friendly.
        
        STYLE
        - Plain language, minimal jargon (explain if used).
        - Use metric and conventional units when relevant.
        - Maintain privacy; only discuss information available via the tool results.
        - Challenge respectfully: Prefer “I think… / this seems… / worth clarifying…” rather than categorical statements.

        EXAMPLES OF DECISIONS
        - Q: “Did the gallbladder surgery happen?” → Search. If an “Operative Note — Laparoscopic Cholecystectomy • 12 Sep 2025 • Operative note” is present, answer: completed on 12 Sep 2025. In “Possible considerations,” add: “Clinic note on 10 Sep 2025 still lists ‘planned cholecystectomy’; the operative note confirms completion—ensure follow-up is scheduled as post-op.”
        - Q: “What meds is the patient on now?” → Search latest discharge/med list; summarize current meds with dose/route/frequency and date of source doc; if a dose looks atypical or conflicts across notes, put that in “Possible considerations” with hedged language and a suggestion to confirm with the clinician.

        BEHAVIOR ON INSUFFICIENT DATA
        - If the tool returns nothing relevant, state that directly, offer general education only if appropriate, and suggest which document to upload or find (e.g., “discharge summary,” “operative note,” “latest medication list”).
        """

        system_prompt += f"""
        Here's the context about the patient’s medical documents you have so far:
        {self.report_service.get_all_text()}
        """

        print("all text: ", self.report_service.get_all_text())

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="messages")
            ])

        self.chain = prompt | llm

        graph_builder = StateGraph(self.State)

        graph_builder.add_node("advisor", self.advisor_node)
        # graph_builder.add_node("tools", ToolNode(tools=tools))

        graph_builder.add_edge(START, "advisor")
        # graph_builder.add_conditional_edges("advisor", tools_condition, "tools")
        # graph_builder.add_edge("tools", "advisor")
        graph_builder.add_edge("advisor", END)


        # db_url = os.getenv("SUPABASE_DB_URL")
        # if db_url is None:
        #     self.checkpointer = MemorySaver()
        # else:
        #     conn = psycopg.connect(
        #         db_url,
        #         autocommit=True,
        #         row_factory=dict_row,
        #         prepare_threshold=0,        # <- key line
        #     )
        #     self.checkpointer = PostgresSaver(conn)
            #self.checkpointer.setup()

        self.graph = graph_builder.compile(checkpointer=get_checkpointer())
        self.config = {"configurable": {"thread_id": self.make_thread_id()}}

    
    
    def chat(self, user_message, history, thread_id: str = None) -> str:
        if thread_id is not None:
            self.config = {"configurable": {"thread_id": thread_id}}
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