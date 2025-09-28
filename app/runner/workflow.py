from dotenv import load_dotenv
from app.src.report_service import ReportService
from app.src.chat_ai import ChatAI

def main():
    # Load environment variables from .env file in the current directory
    load_dotenv(override=True)

    # Initialize services
    report_service = ReportService('lubaba')
    chat_ai = ChatAI(report_service)

    # Upload report
    # report_id = report_service.upload_report("resources/sample_report.jpg")
    
    # Query report
    # query = "Gynaecology"
    # context = report_service.get_context(query)
    # print("Context:\n", context)
    # gr.ChatInterface(fn=chat_ai.chat, title="Report AI Assistant").launch()

    # report_service.list_and_log_reports()
    # url = report_service.presigned_url("b5950b98-aa97-4b1f-8c88-b9477504156b")
    # print("Presigned URL:", url)


if __name__ == "__main__":
    main()