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
from app.src.database import Database
from app.src.utils.files import infer_content_type, infer_extension

def main():
    # Load environment variables from .env file in the current directory
    load_dotenv(override=True)
    BASE_DIR = Path.cwd()
    endpoint_url=os.getenv("B2_ENDPOINT_URL")
    aws_access_key_id=os.getenv("B2_KEY_ID")
    aws_secret_access_key=os.getenv("B2_APP_KEY")
    bucket_name = os.getenv("B2_BUCKET_NAME")
    user_id   = os.getenv("B2_USER_ID")

    cloud_storage = CloudStorage(
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        user_id=user_id,
        bucket_name=bucket_name,
    )

    print("cloud storage initialized")

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    supabase_default_account_id = os.getenv("SUPABASE_DEFAULT_ACCOUNT_ID")

    database = Database(supabase_url, supabase_service_role_key)
    print("database initialized")

    img_path = BASE_DIR / "resources" / "sample_report.jpg"
    path = Path(img_path)
    report_id = str(uuid.uuid4())
    filename = "file." + infer_extension(img_path)
    mime_type = infer_content_type(filename)
    size_bytes = path.stat().st_size


    database.add_database_report(supabase_default_account_id, report_id, filename, mime_type, size_bytes)
    cloud_storage.upload_report(str(img_path), report_id, filename, mime_type)
    url = cloud_storage.get_presigned_url(report_id, filename)  
    
    print("Download from:", url)  




if __name__ == "__main__":
    main()