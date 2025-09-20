from dotenv import load_dotenv
from app.src.report_service import ReportService
from app.src.chat_ai import ChatAI
import gradio as gr

def main():
    # Load environment variables from .env file in the current directory
    load_dotenv(override=True)

    report_service = ReportService()
    chat_ai = ChatAI(report_service)

    # Upload report
    # report_id = report_service.upload_report("resources/sample_report.jpg")
    
    # Query report
    # query = "Which universities in Finland offer scholarships for Master's programs in AI and Game Development?"
    # context = report_service.get_context(query)
    # print("Context:\n", context)
    gr.ChatInterface(fn=chat_ai.chat, title="Report AI Assistant").launch()


if __name__ == "__main__":
    main()