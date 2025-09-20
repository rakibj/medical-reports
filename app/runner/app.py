# app/runner/app.py
from __future__ import annotations
import os, time
from pathlib import Path
from typing import Optional, List, Tuple, Any, Generator

from dotenv import load_dotenv
import gradio as gr

from app.src.report_service import ReportService
from app.src.chat_ai import ChatAI

# ---- boot singletons
def boot():
    load_dotenv(override=True)
    account_id = os.getenv("SUPABASE_DEFAULT_ACCOUNT_ID", "default")
    svc = ReportService()
    chat = ChatAI(svc)
    if hasattr(chat, "set_scope"):
        try:
            chat.set_scope(account_id=account_id, report_id=None)
        except Exception:
            pass
    return svc, chat, account_id

report_service, chat_ai, DEFAULT_ACCOUNT = boot()

# ---- helpers
def list_reports(account_id: str) -> List[Tuple[str, str]]:
    if not hasattr(report_service, "list_reports"):
        return []
    try:
        rows = report_service.list_reports(account_id=account_id) or []
    except Exception:
        return []
    out = []
    for r in rows:
        rid = str(r.get("id") or r.get("report_id") or "")
        if not rid:
            continue
        label = str(r.get("filename") or rid)
        out.append((label, rid))
    return out

def poll_status(report_id: str, max_wait_s=15, interval=1.0) -> str:
    if not hasattr(report_service, "get_report_status"):
        return "Upload completed."
    end = time.time() + max_wait_s
    last = "Queued…"
    while time.time() < end:
        try:
            s = report_service.get_report_status(report_id) or {}
            up = str(s.get("upload_status", "")).lower() or "n/a"
            ocr = str(s.get("ocr_status", "")).lower() or "n/a"
            last = f"Upload: {up} | OCR: {ocr}"
            if ocr in {"complete", "done", "finished"}:
                return last + " ✅"
            if ocr in {"failed", "error"}:
                return last + " ❌"
        except Exception as e:
            return f"Status error: {e}"
        time.sleep(interval)
    return last + " ⏳"

# ---- callbacks
def on_mount():
    return list_reports(DEFAULT_ACCOUNT), "Ready."

def on_upload(file: gr.File, current_report_id: Optional[str]):
    if file is None or not file.name:
        return current_report_id or "", "Pick a file to upload.", list_reports(DEFAULT_ACCOUNT), None
    p = Path(file.name)
    try:
        rid = report_service.upload_report(str(p))
    except Exception as e:
        return current_report_id or "", f"Upload failed: {e}", list_reports(DEFAULT_ACCOUNT), None
    status = poll_status(rid, max_wait_s=20, interval=1.0)
    choices = list_reports(DEFAULT_ACCOUNT)
    if hasattr(chat_ai, "set_scope"):
        try:
            chat_ai.set_scope(account_id=DEFAULT_ACCOUNT, report_id=rid)
        except Exception:
            pass
    return rid, f"Report {rid} | {status}", choices, None  # clears File input

def on_select_report(rid: str):
    rid = rid or None
    if hasattr(chat_ai, "set_scope"):
        try:
            chat_ai.set_scope(account_id=DEFAULT_ACCOUNT, report_id=rid)
        except Exception:
            pass
    return f"Active report: {rid or 'All reports'}"

def on_user_message(message: str, history: List[List[str]], current_report_id: Optional[str]
) -> Generator[Tuple[List[List[str]], List[List[str]]], None, None]:
    if not message:
        yield history, history
        return
    # 1) echo user immediately
    hist_user = history + [[message, None]]
    yield hist_user, hist_user
    # 2) compute reply, fill last turn
    try:
        reply = chat_ai.chat(message, history)
    except Exception as e:
        reply = f"Error: {e}"
    hist_user[-1][1] = reply
    yield hist_user, hist_user

def _clear_text():
    return ""

# ---- UI (no custom CSS; fixed sensible sizes)
with gr.Blocks(title="Report AI Workbench", theme="soft") as demo:
    gr.Markdown("### 📄 Report AI Workbench")

    state_report_id = gr.State(value="")

    with gr.Row():
        with gr.Column(scale=1, min_width=320):
            gr.Markdown("#### Upload & Manage")
            file_in = gr.File(label="Upload PDF/Image", file_count="single", type="filepath")
            btn_upload = gr.Button("Upload", variant="primary")
            dd_reports = gr.Dropdown(label="Select a report", choices=[], value=None, interactive=True)
            status_lbl = gr.Markdown("Ready.")

        with gr.Column(scale=3):
            gr.Markdown("#### Chat")
            chatbox = gr.Chatbot(label="Chat with your reports", height=560)  # tall enough
            txt = gr.Textbox(placeholder="Ask about this report (or all reports)…", label=None)
            btn_send = gr.Button("Send", variant="primary")

    # initial population
    demo.load(on_mount, None, [dd_reports, status_lbl])

    # upload
    btn_upload.click(
        on_upload,
        inputs=[file_in, state_report_id],
        outputs=[state_report_id, status_lbl, dd_reports, file_in],
    )

    # select report
    dd_reports.change(on_select_report, [dd_reports], [status_lbl])

    # chat (generator so user msg appears first)
    txt.submit(on_user_message, [txt, chatbox, state_report_id], [chatbox, chatbox], queue=True)
    btn_send.click(on_user_message, [txt, chatbox, state_report_id], [chatbox, chatbox], queue=True)

    # clear input after send
    txt.submit(_clear_text, None, [txt])
    btn_send.click(_clear_text, None, [txt])

if __name__ == "__main__":
    # enable queue for generator + better UX
    demo.queue().launch()
