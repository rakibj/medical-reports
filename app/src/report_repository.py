from supabase import create_client, Client
from postgrest import APIError 
import base64

class ReportRepository:
    def __init__(self, db_url: str, service_role_key: str):
        # Initialize your database connection here
        supabase_url = db_url
        supabase_service_role_key = service_role_key
        self.client: Client = create_client(supabase_url, supabase_service_role_key)
        
    def add_database_report(self, account_id: str, report_id: str, filename: str, mime_type: str, size_bytes: int) -> None:
        try:
            payload = {
            "id": report_id,
            "account_id": account_id,
            "filename": filename,
            "mime_type": mime_type,
            "size_bytes": size_bytes,
            "upload_status": "uploaded",
            "ocr_status": "queued",
            }
            self.client.table("reports").insert(payload).execute()
            print(f"🗄️  Report metadata added to database (ID: {report_id})")
        except APIError as e:
            print("❌ Failed to add report metadata to database:", e)