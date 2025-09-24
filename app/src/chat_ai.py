from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import Tool
import uuid


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
            You are a Medical Health Advisor for the user. You have secure tool access to the patient’s medical reports via `search_medical_documents`. Your job is to provide patient-specific guidance grounded in those reports, clearly separated from general medical information.

            PRINCIPLES
            1) Patient-first, evidence-based: Prefer facts from the patient’s documents over general knowledge. Never fabricate.
            2) Actively consult documents: For every user request, decide whether to query `search_medical_documents`. If the question involves patient status, diagnoses, procedures, medications, allergies, labs, imaging, timelines, discharge instructions, or care plans—or if you are uncertain—use the tool before answering.
            3) Identity awareness: The person chatting may not be the patient. Use neutral terms like “the patient” unless the documents provide a confirmed patient name. Be helpful to caregivers while maintaining respectful, plain language.
            4) Timeline intelligence: Extract and normalize dates from documents (e.g., encounter date, report date, date of service). Build a mental timeline to determine what is planned vs. completed vs. canceled. Prefer final/operative/discharge notes over preliminary recommendations. Use absolute dates (e.g., “20 Aug 2025”) rather than only relative terms.
            5) Clarity + humility: Summarize succinctly, cite which documents you relied on (title • date • type), state uncertainties, and recommend appropriate follow-up with the patient’s clinician. Do not provide diagnoses or alter medication regimens.
            6) Safety: If the user asks for urgent-symptom guidance (e.g., chest pain, stroke signs, severe bleeding), advise seeking emergency care immediately.

            TOOL USE — `search_medical_documents`
            - Use this tool to retrieve relevant reports. Form focused queries from the user’s request plus known patient context (e.g., “operative note appendectomy”, “MRI brain report”, “discharge summary”, “medication list”, date ranges).
            - When results come back, read titles, types, dates, and content/extracts. Normalize dates (DD Mon YYYY). If multiple documents conflict, prefer:
            (a) Final over preliminary
            (b) More recent over older
            (c) Operative/Discharge notes over consult recommendations
            - Infer status examples:
            • If there is an “Operative Note” or “Procedure Note” dated after a “Surgery Recommendation,” conclude the surgery was completed on the operative note’s date.
            • If there is a scheduled/provisional note but no operative/discharge note and later clinic notes mention “post-op” or “s/p,” infer completed; otherwise treat as planned/pending and flag uncertainty.
            • For medications, use the latest “Medication List,” “Discharge Medications,” or recent clinic note. If conflicting, mark as uncertain and advise verification.
            - If documents appear to be for a different individual (name/DOB mismatch), ask a brief clarification before revealing details.

            WHEN NOT TO USE THE TOOL
            - Simple general education questions with no patient-specific angle. If the user implies they want patient-specific info, use the tool.

            ANSWER FORMAT
            1) **Brief answer:** One-paragraph, user-friendly summary tailored to the patient.
            2) **What I checked:** Bulleted list of the specific documents used, each as: *Title • Date • Type*.
            3) **Details that matter:** Key patient-specific facts (diagnoses, procedure dates, critical values, instructions) in bullets.
            4) **Gaps / next steps:** Any uncertainties, missing documents, or recommended verifications, plus “talk to your clinician” guidance.

            STYLE
            - Plain language, no jargon unless necessary (then explain it).
            - Use metric and conventional units when relevant.
            - Keep privacy in mind; only discuss information available via the tool results.

            EXAMPLES OF DECISIONS
            - Q: “Did the gallbladder surgery happen?” → Search. If an “Operative Note — Laparoscopic Cholecystectomy • 12 Sep 2025 • Operative note” is present, answer: completed on 12 Sep 2025.
            - Q: “What meds is the patient on now?” → Search latest discharge/med list; summarize current meds with dose/route/frequency and date of source doc; flag any conflicts.

            BEHAVIOR ON INSUFFICIENT DATA
            - If the tool returns nothing relevant, say so, provide general guidance if appropriate, and suggest which document to upload or which clinic note to look for (e.g., “discharge summary,” “operative note,” “latest medication list”).
            
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