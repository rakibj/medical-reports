import boto3
from botocore.client import Config
import os
import mimetypes
from pathlib import Path

class CloudStorage:
    def __init__(self, endpoint_url: str, aws_access_key_id: str, aws_secret_access_key: str, user_id: str, bucket_name: str):
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
        )
        self.user_id = user_id
        self.bucket_name = bucket_name

    def make_report_key(self, report_id: str, filename: str, subfolder: str = "source") -> str:
        """
        Build an object key like:
        acct/{user_id}/reports/{report_id}/{subfolder}/{filename}
        """
        return f"acct/{self.user_id}/reports/{report_id}/{subfolder}/{filename}"
    
    def get_presigned_url(self, report_id: str, filename: str, expires_in: int = 900) -> str:
        """
        Generate a presigned URL so the client can download/view the file directly.
        """
        key = self.make_report_key(report_id, filename)
        url = self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket_name, "Key": key},
            ExpiresIn=expires_in
        )
        print(f"🔗 Presigned URL (valid {expires_in}s): {url}")
        return url
    
    @staticmethod
    def infer_content_type(filename: str) -> str:
        ctype, _ = mimetypes.guess_type(filename)
        if ctype:
            return ctype
        ext = Path(filename).suffix.lower()
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".pdf": "application/pdf",
        }.get(ext, "application/octet-stream")

    @staticmethod
    def infer_extension(local_path: str) -> str:
        ext = Path(local_path).suffix.lstrip(".") 
        return ext
    
    def upload_report(self, local_path: str, report_id: str, filename: str, content_type: str) -> str:
        key = self.make_report_key(report_id, filename)
        with open(local_path, "rb") as f:
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=f,
                ContentType=content_type,
                Metadata={"report-id": report_id},
            )
        print(f"✅ Uploaded → {key}")
        return key





