from dotenv import load_dotenv
import os, re, math
from typing import List, Tuple, Dict, Optional
import uuid
from pathlib import Path
import mimetypes
from supabase import create_client, Client
from postgrest import APIError 
import base64
from openai import OpenAI
from pdf2image import convert_from_path
from PIL import Image
import pypdfium2 as pdfium
from app.src.cloud_storage import CloudStorage
from app.src.report_repository import ReportRepository
from app.src.utils.files import infer_content_type, infer_extension
from app.src.ocr_processor import OCRProcessor
from app.src.text_embedder import TextEmbedder
import logging

class ReportService:
    def __init__(self):
        endpoint_url=os.getenv("B2_ENDPOINT_URL")
        aws_access_key_id=os.getenv("B2_KEY_ID")
        aws_secret_access_key=os.getenv("B2_APP_KEY")
        bucket_name = os.getenv("B2_BUCKET_NAME")
        user_id   = os.getenv("B2_USER_ID")

        self.cloud_storage = CloudStorage(
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            user_id=user_id,
            bucket_name=bucket_name,
        )

        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.supabase_default_account_id = os.getenv("SUPABASE_DEFAULT_ACCOUNT_ID")
        self.database = ReportRepository(self.supabase_url, self.supabase_service_role_key)
        self.text_embedder = TextEmbedder()

        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")



    def upload_report(self, path: str):
        path = Path(path)
        report_id = str(uuid.uuid4())
        filename = "file." + infer_extension(path)
        mime_type = infer_content_type(filename)
        size_bytes = path.stat().st_size

        ocr_processor = OCRProcessor()
        full_text = ocr_processor.ocr_file(path)
        self.cloud_storage.upload_report(str(path), report_id, filename, mime_type)
        url = self.cloud_storage.get_presigned_url(report_id, filename)  
        new_filename = self.text_embedder.suggest_filename(full_text, filename)
        self.database.add_database_report(self.supabase_default_account_id, report_id, new_filename, mime_type, size_bytes)
        self.database.index_ocr_text(self.supabase_default_account_id, report_id, full_text)
        return report_id, url

    def get_context(self, query: str):
        context = self.database.get_context_from_embeddings(self.supabase_default_account_id,query)
        return context
    

    def presigned_url(self, report_id: str, expires_in: int = 900) -> str:
        return self.database.get_presigned_url_for_report(
            self.supabase_default_account_id,
            report_id,
            self.cloud_storage,
            expires_in=expires_in,
            )
    
    def list_reports(self) -> List[Dict[str, str]]:
        return self.database.list_reports(self.supabase_default_account_id)
    
    def list_and_log_reports(self) -> List[Dict[str, str]]:
        """
        Fetch all reports (newest → oldest) and log them in a readable format.
        Returns the list for programmatic use as well.
        """
        reports = self.list_reports()  # uses repo under the hood
        logging.info("Found %d report(s).", len(reports))
        for r in reports:
            rid = r.get("id", "")
            created = r.get("created_at", "")
            fname = r.get("filename", "")
            logging.info("• id=%s | created_at=%s | filename=%s", rid, created, fname)
        return reports