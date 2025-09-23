from __future__ import annotations

import os
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv

# reuse your existing services
from app.src.report_service import ReportService
from app.src.chat_ai import ChatAI

# ---- boot singletons (same as Gradio boot)
load_dotenv(override=True)
report_service = ReportService()
chat_ai = ChatAI(report_service)

# ---- auth (simple API key in header)
API_KEY = os.getenv("API_KEY")  # set this in your .env (server-side only)
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

def require_api_key(key: Optional[str] = Depends(api_key_header)):
    if not API_KEY:
        # if no key configured, allow (dev mode)
        return
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

# ---- app
app = FastAPI(title="Report Service API", version="1.0.0")

# CORS: allow only your Next.js origins in prod
origins_env = os.getenv("CORS_ALLOW_ORIGINS", "")
allowed = [o.strip() for o in origins_env.split(",") if o.strip()]
if not allowed:
    allowed = ["http://localhost:3000"]  # dev default

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- models
class UploadResponse(BaseModel):
    report_id: str
    url: Optional[str] = None

class ListItem(BaseModel):
    id: str
    filename: str
    created_at: Optional[str] = None
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None

class PresignedUrlResponse(BaseModel):
    url: str
    expires_in: int

class ContextResponse(BaseModel):
    context: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[List[str]]] = None
    report_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str

# ---- endpoints

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/reports", response_model=UploadResponse, dependencies=[Depends(require_api_key)])
async def create_report(file: UploadFile = File(...)):
    """
    Upload a PDF/image. Uses your existing OCR/embedding pipeline.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # FastAPI gives you a spooled file; write it to a temp path for your service
    import tempfile, shutil, os
    suffix = os.path.splitext(file.filename)[1] or ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        shutil.copyfileobj(file.file, tmp)

    try:
        report_id, presigned = report_service.upload_report(tmp_path)
        return UploadResponse(report_id=report_id, url=presigned)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

@app.get("/reports", response_model=List[ListItem], dependencies=[Depends(require_api_key)])
def list_reports():
    rows = report_service.list_reports() or []
    # normalize keys to ListItem
    out: List[ListItem] = []
    for r in rows:
        out.append(ListItem(
            id=str(r.get("id") or r.get("report_id") or ""),
            filename=str(r.get("filename") or ""),
            created_at=r.get("created_at"),
            mime_type=r.get("mime_type"),
            size_bytes=r.get("size_bytes"),
        ))
    return out

@app.get("/reports/{report_id}/url", response_model=PresignedUrlResponse, dependencies=[Depends(require_api_key)])
def get_presigned_url(report_id: str, expires_in: int = Query(900, ge=60, le=60*60*24)):
    try:
        url = report_service.presigned_url(report_id, expires_in=expires_in)
        return PresignedUrlResponse(url=url, expires_in=expires_in)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/context", response_model=ContextResponse, dependencies=[Depends(require_api_key)])
def get_context(query: str = Query(..., min_length=1)):
    try:
        ctx = report_service.get_context(query)
        return ContextResponse(context=ctx or "")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: if you want chat from the browser via API
@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
def chat(req: ChatRequest):
    try:
        # keep scope synced if report scoping matters
        if hasattr(chat_ai, "set_scope") and req.report_id:
            try:
                chat_ai.set_scope(
                    account_id=os.getenv("SUPABASE_DEFAULT_ACCOUNT_ID", "default"),
                    report_id=req.report_id
                )
            except Exception:
                pass
        reply = chat_ai.chat(req.message, req.history or [])
        return ChatResponse(reply=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
