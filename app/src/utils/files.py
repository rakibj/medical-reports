import os
import mimetypes
from pathlib import Path

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