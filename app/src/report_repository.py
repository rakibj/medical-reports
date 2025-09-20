from supabase import create_client, Client
from postgrest import APIError 
import base64
from typing import List, Tuple, Dict, Optional
from app.src.text_embedder import TextEmbedder

class ReportRepository:
    def __init__(self, db_url: str, service_role_key: str):
        # Initialize your database connection here
        supabase_url = db_url
        supabase_service_role_key = service_role_key
        self.client: Client = create_client(supabase_url, supabase_service_role_key)
        self.text_embedder = TextEmbedder()
        
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

    def upsert_chunks(
        self, account_id: str, report_id: str, page_no: int | None, chunks: List[str]
    ) -> None:
        if not chunks:
            return
        embs = self.text_embedder.embed_texts(chunks)
        rows = []
        for idx, (c, e) in enumerate(zip(chunks, embs)):
            rows.append({
                "account_id": account_id,
                "report_id": report_id,
                "page_no": page_no,
                "chunk_no": idx,
                "content": c,
                "content_tokens": None,   # fill if you count tokens later
                "embedding": e,
            })
        # Supabase can insert lists of JSON rows directly
        self.client.table("report_chunks").upsert(rows, on_conflict="report_id,chunk_no").execute()
    
    # Example: from your OCR text (whole doc) → chunk → embed → store
    def index_ocr_text(self, account_id: str, report_id: str, full_text: str, page_map: List[Tuple[int, str]] | None = None):
        """
        If you have per-page OCR, pass page_map=[(1, text1), (2, text2)...].
        Otherwise pass full_text and leave page_map=None.
        """
        if page_map:
            for page_no, page_text in page_map:
                chunks = self.text_embedder.simple_chunk(page_text)
                self.upsert_chunks(account_id, report_id, page_no, chunks)
        else:
            chunks = self.text_embedder.simple_chunk(full_text)
            self.upsert_chunks(account_id, report_id, None, chunks)
        print("✅ Upserting embeddings to database completed")

    def search_chunks(self, account_id: str, query: str, k: int = 5, report_id: str | None = None):
        q_emb = self.text_embedder.embed_query(query)
        resp = self.client.rpc(
            "search_report_chunks_json_simple",
            {"p_account_id": account_id, "p_query": q_emb, "p_limit": k, "p_report_id": report_id}
        ).execute()
        return resp.data
    
    def get_context_from_embeddings(self,
    account_id: str,
    query: str,
    report_id: Optional[str],
    *,
    k: int = 5,
    sim_threshold: float = 0.10,     # tweak: 0.30 (weak) · 0.50 (medium) · 0.70 (strong)
    max_snippet: int = 1200,
    max_total_chars: int = 4000,     # safety cap on total context
    min_hits: int = 1,               # require at least this many hits above threshold
    ) -> str:
        rows: List[Dict] = self.search_chunks(account_id, query, k=k, report_id=report_id) or []

        # keep only rows with similarity >= threshold
        good: List[Dict] = []
        for r in rows:
            try:
                sim = float(r.get("similarity", 0.0))
            except (TypeError, ValueError):
                sim = 0.0
            if sim >= sim_threshold:
                good.append(r)

        if len(good) < min_hits:
            return ""  # << no usable context

        parts: List[str] = []
        for r in good:
            sim = float(r.get("similarity", 0.0))
            page = r.get("page_no")
            page_str = str(page) if page is not None else "?"
            snippet = (r.get("content") or "").strip()
            if len(snippet) > max_snippet:
                snippet = snippet[:max_snippet] + " …"
            parts.append(f"[page {page_str} | sim {sim:.3f}] {snippet}")

        context = "\n\n".join(parts)
        if len(context) > max_total_chars:
            context = context[:max_total_chars] + " …"
        return context or "No relevant information found in the documents."
    