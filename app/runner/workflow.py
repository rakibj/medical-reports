from dotenv import load_dotenv
from app.src.report_service import ReportService

def main():
    # Load environment variables from .env file in the current directory
    load_dotenv(override=True)
    report_service = ReportService()

    # Upload report
    report_id = report_service.upload_report("resources/sample_report.jpg")
    
    # Query report
    # report_id = "3d5612c1-7525-48cd-b1d4-37f6de291f2b"
    # query = "Which universities in Finland offer scholarships for Master's programs in AI and Game Development?"
    # context = report_service.get_context(report_id, query)
    # print("Context:\n", context)

if __name__ == "__main__":
    main()