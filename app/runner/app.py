# app/runner/app.py
from __future__ import annotations
import os, time, threading
from pathlib import Path
from typing import Optional, List, Tuple, Generator, Dict

from dotenv import load_dotenv
import gradio as gr

from app.src.report_service import ReportService
from app.src.chat_ai import ChatAI
from app.src.report_repository import ReportRepository  # for username_exists check

# ---------------------------
# Helpers to create dependencies
# ---------------------------

def _make_repo_from_env() -> ReportRepository:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not supabase_service_role_key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in environment.")
    return ReportRepository(supabase_url, supabase_service_role_key)

def _display_label(filename: str, rid: str) -> str:
    # keep labels unique even with duplicate filenames
    return f"{filename} — {rid[:8]}"

def _build_report_mapping(report_service: ReportService) -> Tuple[Dict[str, str], List[str]]:
    """
    Returns (label->report_id mapping, radio_choices) for the CURRENT user/session
    """
    mapping: Dict[str, str] = {}
    for filename, rid in list_reports(report_service):
        label = _display_label(filename, rid)
        mapping[label] = rid
    return mapping, list(mapping.keys())

# ---------------------------
# Thin wrappers over ReportService (now passed via state)
# ---------------------------

def list_reports(report_service: ReportService) -> List[Tuple[str, str]]:
    """
    Returns [(filename_label, report_id)]
    """
    if not hasattr(report_service, "list_reports"):
        return []
    try:
        rows = report_service.list_reports() or []
    except Exception:
        return []
    out: List[Tuple[str, str]] = []
    for r in rows:
        rid = str(r.get("id") or r.get("report_id") or "")
        if not rid:
            continue
        label = str(r.get("filename") or rid)
        out.append((label, rid))
    return out

def poll_status(report_service: ReportService, report_id: str, max_wait_s=15, interval=1.0) -> str:
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

# ---------------------------
# Login / Logout flow
# ---------------------------

def on_login(username_input: Optional[str]):
    """
    1) If username_input is blank -> proceed with None (ReportService default behavior)
    2) If provided -> check existence via ReportRepository.username_exists
    3) On success -> init ReportService/ChatAI and show main app
    Always returns exactly 10 outputs (to match Gradio wiring).
    """
    try:
        username = (username_input or "").strip() or None

        # check if username exists
        if username is not None:
            repo = _make_repo_from_env()
            if not hasattr(repo, "username_exists"):
                return (
                    gr.update(visible=True),    # login panel
                    gr.update(visible=False),   # app panel
                    "Server missing username_exists(). Please deploy the latest backend.",
                    None, None, "", {},         # svc, chat, default_account, mapping
                    gr.update(choices=[], value=None),  # reports_radio
                    "Error",                    # status_lbl
                    gr.update(link=None),       # btn_view
                )
            if not repo.username_exists(username):
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    f"❌ Username not found: {username}",
                    None, None, "", {},
                    gr.update(choices=[], value=None),
                    "Error",
                    gr.update(link=None),
                )

        # ok to proceed
        svc = ReportService(username=username)
        chat = ChatAI(svc)
        default_account = svc.user_id

        mapping, radio_choices = _build_report_mapping(svc)
        status_text = "Ready."
        view_btn_update = gr.update(link=None)

        return (
            gr.update(visible=False),          # hide login
            gr.update(visible=True),           # show app
            "",                                # clear login status
            svc,                               # state_service
            chat,                              # state_chat_ai
            default_account,                   # state_default_account
            mapping,                           # state_reports_mapping
            gr.update(choices=radio_choices, value=None),  # reports_radio
            status_text,                       # status_lbl
            view_btn_update,                   # btn_view
        )

    except Exception as e:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            f"Login error: {e}",
            None, None, "", {},
            gr.update(choices=[], value=None),
            "Error",
            gr.update(link=None),
        )


def on_logout():
    """
    Clear session states and show the login screen again.
    """
    return (
        gr.update(visible=True),   # login panel visible
        gr.update(visible=False),  # app panel hidden
        "",                        # login status cleared
        None, None, "", {},        # state_service, state_chat_ai, DEFAULT_ACCOUNT, mapping
        gr.update(choices=[], value=None),  # reports_radio
        "Logged out.",
        gr.update(link=None)
    )

# ---------------------------
# App callbacks (now receive stateful svc/chat/default_account)
# ---------------------------

def on_refresh(state_service: ReportService):
    mapping, choices = _build_report_mapping(state_service)
    return mapping, gr.update(choices=choices, value=None), gr.update(link=None)

def on_upload(
    state_service: ReportService,
    state_chat_ai: ChatAI,
    state_default_account: str,
    file_path: Optional[str],
    current_report_id: Optional[str],
    mapping: Dict[str, str],
):
    """
    Gradio v4: File with type='filepath' yields a string path (or None).
    """
    if not file_path:
        return current_report_id or "", "Pick a file to upload.", gr.update(), None, mapping, gr.update(link=None)

    p = Path(file_path)
    try:
        rid, _url = state_service.upload_report(str(p))  # (report_id, presigned_url)
    except Exception as e:
        return current_report_id or "", f"Upload failed: {e}", gr.update(), None, mapping, gr.update(link=None)

    status = poll_status(state_service, rid, max_wait_s=20, interval=1.0)

    # refresh mapping and select the just-uploaded report
    new_mapping, choices = _build_report_mapping(state_service)
    display_label = next((lbl for lbl, _rid in new_mapping.items() if _rid == rid), None)

    # set chat scope
    if hasattr(state_chat_ai, "set_scope"):
        try:
            state_chat_ai.set_scope(account_id=state_default_account, report_id=rid)
        except Exception:
            pass

    # prepare the View button link for single-click open
    try:
        presigned = state_service.presigned_url(rid)
        view_btn_update = gr.update(link=presigned)
    except Exception:
        view_btn_update = gr.update(link=None)

    radio_update = gr.update(choices=choices, value=display_label)
    return rid, f"Report {rid} | {status}", radio_update, None, new_mapping, view_btn_update

def on_pick_report(
    state_service: ReportService,
    state_chat_ai: ChatAI,
    state_default_account: str,
    selected_label: Optional[str],
    mapping: Dict[str, str],
):
    """
    When user selects a report in the radio list:
      - set active scope
      - precompute presigned URL and attach it to the View button (single-click open)
    """
    rid = mapping.get(selected_label or "", None)

    # keep chat scope in sync
    if hasattr(state_chat_ai, "set_scope"):
        try:
            state_chat_ai.set_scope(account_id=state_default_account, report_id=rid)
        except Exception:
            pass

    status_msg = f"Active report: {selected_label or 'All reports'}"

    # default: no link
    view_btn_update = gr.update(link=None)

    if rid:
        try:
            url = state_service.presigned_url(rid)
            view_btn_update = gr.update(link=url)  # make View a native anchor
            status_msg += " — link ready."
        except Exception as e:
            status_msg += f" (link error: {e})"

    return status_msg, (rid or ""), view_btn_update

def on_user_message(
    state_chat_ai: ChatAI,
    message: str,
    history: List[List[str]],
    current_report_id: Optional[str],
) -> Generator[Tuple[List[List[str]], List[List[str]]], None, None]:
    """
    Streaming UX with an in-chat typing indicator.
    Steps:
      1) Append the user message and an assistant 'typing' placeholder.
      2) While the model is working (on a background thread), animate dots.
      3) Replace the placeholder with the real reply when done.
    """
    if not message:
        yield history, history
        return

    # append a placeholder assistant bubble immediately
    hist = (history or []) + [[message, "🤖 thinking"]]
    yield hist, hist  # user sees their message + initial typing bubble instantly

    result: Dict[str, Optional[str]] = {"reply": None, "error": None}

    def _worker():
        try:
            result["reply"] = state_chat_ai.chat(message, history)
        except Exception as e:
            print("Chat error:", repr(e))
            result["error"] = "Something went wrong while looking up your data. Please try again."

    th = threading.Thread(target=_worker, daemon=True)
    th.start()

    dots = ["", ".", "..", "..."]
    i = 0
    # animate while background thread runs
    while th.is_alive():
        hist[-1][1] = f"🤖 thinking{dots[i % len(dots)]}"
        i += 1
        yield hist, hist
        time.sleep(0.35)

    th.join()
    reply = result["reply"] if result["reply"] is not None else f"Error: {result['error']}"
    hist[-1][1] = reply
    yield hist, hist

def _clear_text():
    return ""

# ---------------------------
# UI
# ---------------------------

load_dotenv(override=True)

with gr.Blocks(title="Report AI Workbench", theme="soft") as demo:
    gr.Markdown("### 📄 Report AI Workbench")

    # Session States
    state_service = gr.State(value=None)         # ReportService
    state_chat_ai = gr.State(value=None)         # ChatAI
    state_default_account = gr.State(value="")   # str
    state_report_id = gr.State(value="")         # active report id
    state_reports_mapping = gr.State(value={})   # {display_label: report_id}

    # -------- Login Panel --------
    with gr.Group(visible=True) as login_panel:
        gr.Markdown("#### Sign in")
        username_tb = gr.Textbox(label="Username (leave blank to continue as default)", placeholder="e.g., rakibjahan", value="")
        btn_login = gr.Button("Continue", variant="primary")
        login_status = gr.Markdown("")

    # -------- Main App Panel (hidden until login) --------
    with gr.Group(visible=False) as app_panel:
        with gr.Row():
            gr.Markdown("#### Upload & Manage", elem_id="left-col")
            btn_logout = gr.Button("Logout", variant="secondary")

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                file_in = gr.File(label="Upload PDF/Image", file_count="single", type="filepath")
                btn_upload = gr.Button("Upload", variant="primary")

                gr.Markdown("#### Reports")
                reports_radio = gr.Radio(label="Available reports", choices=[], value=None, interactive=True)

                with gr.Row():
                    btn_refresh = gr.Button("Refresh list")
                    # View button becomes a native link when 'link' is set via updates
                    btn_view = gr.Button("View", variant="secondary")

                status_lbl = gr.Markdown("Ready.")

            with gr.Column(scale=3):
                gr.Markdown("#### Chat")
                chatbox = gr.Chatbot(label="Chat with your reports", height=560)
                txt = gr.Textbox(placeholder="Ask about this report (or all reports)…", label=None)
                btn_send = gr.Button("Send", variant="primary")

    # ---- Login wiring
    btn_login.click(
        on_login,
        inputs=[username_tb],
        outputs=[
            login_panel,             # show/hide login
            app_panel,               # show/hide app
            login_status,            # login error/success
            state_service,           # ReportService instance
            state_chat_ai,           # ChatAI instance
            state_default_account,   # DEFAULT_ACCOUNT
            state_reports_mapping,   # mapping
            reports_radio,           # choices populated
            status_lbl,              # status text
            btn_view,                # clear link
        ],
    )

    # ---- Logout wiring
    btn_logout.click(
        on_logout,
        inputs=None,
        outputs=[
            login_panel,
            app_panel,
            login_status,
            state_service,
            state_chat_ai,
            state_default_account,
            state_reports_mapping,
            reports_radio,
            status_lbl,
            btn_view,
        ],
    )

    # ---- Upload: also primes the View button with the new report's presigned URL
    btn_upload.click(
        on_upload,
        inputs=[state_service, state_chat_ai, state_default_account, file_in, state_report_id, state_reports_mapping],
        outputs=[state_report_id, status_lbl, reports_radio, file_in, state_reports_mapping, btn_view],
    )

    # ---- refresh list (clears the View link)
    btn_refresh.click(on_refresh, [state_service], [state_reports_mapping, reports_radio, btn_view])

    # ---- change selection -> set scope + set View button link
    reports_radio.change(
        on_pick_report,
        [state_service, state_chat_ai, state_default_account, reports_radio, state_reports_mapping],
        [status_lbl, state_report_id, btn_view],
    )

    # ---- chat (generator so user msg appears first) with built-in typing animation
    txt.submit(on_user_message, [state_chat_ai, txt, chatbox, state_report_id], [chatbox, chatbox], queue=True)
    btn_send.click(on_user_message, [state_chat_ai, txt, chatbox, state_report_id], [chatbox, chatbox], queue=True)

    # clear input after send
    txt.submit(_clear_text, None, [txt])
    btn_send.click(_clear_text, None, [txt])

if __name__ == "__main__":
    demo.queue().launch()
