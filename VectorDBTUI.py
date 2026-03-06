#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Menu-driven TUI for VectorDBService (cross-platform, no curses).
Depends on VectorDBClient.py (your client module).

Install:
  pip install prompt_toolkit

Run:
  python vectordb_tui.py --url http://127.0.0.1:8001
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea, Frame, Label
from prompt_toolkit.shortcuts import (
    button_dialog,
    input_dialog,
    message_dialog,
    radiolist_dialog,
    yes_no_dialog,
)

# ----------------------------
# Import YOUR client module
# ----------------------------
# Path per your message: VectorDBClient.py
from VectorDBClient import (
    VectorDBClient,
    ServerBusyError,
    ServerInitializingError,
    RetryableError,
    NonRetryableError,
    InvalidRequestError,
    AuthenticationError,
    ServiceNotConfiguredError,
    VectorDBTimeoutError,
)


# ----------------------------
# Utilities
# ----------------------------

STYLE = Style.from_dict({
    "dialog": "bg:#1e1e1e #dcdcdc",
    "dialog frame.label": "bg:#1e1e1e #00d7ff bold",
    "dialog.body": "bg:#1e1e1e #dcdcdc",
    "dialog shadow": "bg:#000000",
})

def jdump(obj: Any, max_len: int = 20000) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        s = str(obj)
    if len(s) > max_len:
        s = s[:max_len] + "\n... (truncated)"
    return s

def err_str(e: Exception) -> str:
    if isinstance(e, ServiceNotConfiguredError):
        return f"[NOT CONFIGURED] {e}"
    if isinstance(e, AuthenticationError):
        return f"[AUTH] {e}"
    if isinstance(e, InvalidRequestError):
        return f"[INVALID] {e}"
    if isinstance(e, ServerInitializingError):
        return f"[INIT] {e}"
    if isinstance(e, ServerBusyError):
        return f"[BUSY] {e}"
    if isinstance(e, VectorDBTimeoutError):
        return f"[TIMEOUT] {e}"
    if isinstance(e, RetryableError):
        return f"[RETRYABLE] {e}"
    if isinstance(e, NonRetryableError):
        return f"[ERROR] {e}"
    return f"[EXCEPTION] {e}"

def safe_json_loads(s: str) -> Any:
    s = (s or "").strip()
    if not s:
        return None
    return json.loads(s)

def maybe_load_at_file(token: str) -> str:
    token = (token or "").strip()
    if token.startswith("@"):
        path = token[1:]
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return token

def maybe_load_json_at_file(token: str) -> Any:
    token = (token or "").strip()
    if not token:
        return None
    if token.startswith("@"):
        path = token[1:]
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # if looks like json
    if token.startswith("{") or token.startswith("["):
        return json.loads(token)
    # otherwise treat as string
    return token


# ----------------------------
# Pager (scrollable viewer)
# ----------------------------

def pager(title: str, content: str) -> None:
    """
    Full-screen scrollable viewer for long text.
    Keys: q / Esc exit
          Up/Down/PgUp/PgDn scroll (TextArea default)
    """
    kb = KeyBindings()

    text = TextArea(
        text=content,
        read_only=True,
        scrollbar=True,
        line_numbers=False,
        focus_on_click=True,
    )

    @kb.add("q")
    @kb.add("escape")
    def _(event):
        event.app.exit()

    root = Frame(text, title=title)
    app = Application(layout=Layout(root), key_bindings=kb, full_screen=True, style=STYLE)
    app.run()


# ----------------------------
# JSON editor (multiline)
# ----------------------------

def json_editor(title: str, default_obj: Any) -> Optional[Any]:
    """
    Full-screen JSON editor.
    F2: save (validate JSON)
    Esc: cancel
    """
    kb = KeyBindings()
    default_text = jdump(default_obj) if default_obj is not None else ""

    status = Label("F2=保存  Esc=取消")

    editor = TextArea(
        text=default_text,
        multiline=True,
        scrollbar=True,
        line_numbers=True,
        focus_on_click=True
    )

    body = HSplit([
        Frame(editor, title=title),
        Window(height=1, content=FormattedTextControl(lambda: status.text)),
    ])

    app = Application(layout=Layout(body), key_bindings=kb, full_screen=True, style=STYLE)

    result = {"value": None, "cancel": False}

    @kb.add("escape")
    def _(event):
        result["cancel"] = True
        event.app.exit()

    @kb.add("f2")
    def _(event):
        try:
            obj = safe_json_loads(editor.text)
            result["value"] = obj
            event.app.exit()
        except Exception as e:
            status.text = f"JSON 解析失败: {e}（请修正后再 F2 保存）"

    app.run()
    if result["cancel"]:
        return None
    return result["value"]


# ----------------------------
# App state + menus
# ----------------------------

class TUI:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = VectorDBClient(self.base_url)
        self.collection: Optional[str] = None
        self.recent_analysis_jobs: List[str] = []
        self.recent_agg_jobs: List[str] = []

    # -------- Core dialogs --------

    def info(self, title: str, obj: Any):
        txt = jdump(obj)
        if len(txt) > 4000:
            pager(title, txt)
        else:
            message_dialog(title=title, text=txt, style=STYLE).run()

    def error(self, title: str, e: Exception):
        message_dialog(title=title, text=err_str(e), style=STYLE).run()

    def ask_collection_required(self) -> str:
        if not self.collection:
            raise ValueError("未选择 collection。请先到 Collections -> Use 选择一个。")
        return self.collection

    # -------- Main loop --------

    def run(self):
        # quick connect status (non-blocking)
        try:
            st = self.client.get_status()
            self.info("Connected", st)
        except Exception as e:
            self.error("Warning", e)

        while True:
            label = f"当前 collection: {self.collection or '(none)'}    URL: {self.base_url}"
            ret = button_dialog(
                title="VectorDB TUI",
                text=label + "\n\n请选择一个模块：",
                buttons=[
                    ("Status/Queue/Health", "status"),
                    ("Collections", "collections"),
                    ("Documents", "docs"),
                    ("Search", "search"),
                    ("Timestamp Stats", "ts"),
                    ("Analysis", "analysis"),
                    ("Aggregation", "agg"),
                    ("Admin (Backup/Restore)", "admin"),
                    ("Settings", "settings"),
                    ("Quit", "quit"),
                ],
                style=STYLE
            ).run()

            if ret == "quit" or ret is None:
                return
            try:
                if ret == "status":
                    self.menu_status()
                elif ret == "collections":
                    self.menu_collections()
                elif ret == "docs":
                    self.menu_docs()
                elif ret == "search":
                    self.menu_search()
                elif ret == "ts":
                    self.menu_timestamp_stats()
                elif ret == "analysis":
                    self.menu_analysis()
                elif ret == "agg":
                    self.menu_aggregation()
                elif ret == "admin":
                    self.menu_admin()
                elif ret == "settings":
                    self.menu_settings()
            except Exception as e:
                self.error("Error", e)

    # -------- Status --------

    def menu_status(self):
        while True:
            ret = button_dialog(
                title="Status / Queue / Health",
                text="选择操作：",
                buttons=[
                    ("Engine Status", "engine"),
                    ("Queue Status", "queue"),
                    ("Health Check", "health"),
                    ("Wait Until Ready", "wait"),
                    ("Back", "back"),
                ],
                style=STYLE
            ).run()
            if ret in (None, "back"):
                return
            try:
                if ret == "engine":
                    self.info("Engine Status", self.client.get_status())
                elif ret == "queue":
                    self.info("Queue Status", self.client.get_queue_status())
                elif ret == "health":
                    r = requests.get(f"{self.base_url}/api/health", timeout=5)
                    r.raise_for_status()
                    self.info("Health", r.json())
                elif ret == "wait":
                    t = input_dialog(title="Wait Until Ready", text="timeout 秒（默认 60）:", style=STYLE).run()
                    if t is None:
                        continue
                    timeout = float(t or "60")
                    self.client.wait_until_ready(timeout=timeout, poll_interval=2.0)
                    self.info("Ready", {"status": "ready"})
            except Exception as e:
                self.error("Error", e)

    # -------- Collections --------

    def menu_collections(self):
        while True:
            ret = button_dialog(
                title="Collections",
                text="选择操作：",
                buttons=[
                    ("List", "list"),
                    ("Create/Update", "create"),
                    ("Use (select)", "use"),
                    ("Back", "back"),
                ],
                style=STYLE
            ).run()
            if ret in (None, "back"):
                return

            try:
                if ret == "list":
                    cols = self.client.list_collections()
                    self.info("Collections", {"collections": cols})
                elif ret == "create":
                    name = input_dialog(title="Create Collection", text="collection name:", style=STYLE).run()
                    if not name:
                        continue
                    cs = input_dialog(title="Create Collection", text="chunk_size (default 512):", style=STYLE).run()
                    co = input_dialog(title="Create Collection", text="chunk_overlap (default 50):", style=STYLE).run()
                    chunk_size = int(cs or "512")
                    chunk_overlap = int(co or "50")
                    self.client.create_collection(name, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    self.collection = name
                    self.info("OK", {"collection": name, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap})
                elif ret == "use":
                    cols = self.client.list_collections()
                    if not cols:
                        self.info("Info", {"message": "No collections."})
                        continue
                    sel = radiolist_dialog(
                        title="Use Collection",
                        text="选择一个 collection：",
                        values=[(c, c) for c in cols],
                        style=STYLE
                    ).run()
                    if sel:
                        self.collection = sel
                        self.info("OK", {"current_collection": self.collection})
            except Exception as e:
                self.error("Error", e)

    # -------- Docs --------

    def menu_docs(self):
        while True:
            ret = button_dialog(
                title="Documents",
                text=f"当前 collection: {self.collection or '(none)'}\n\n选择操作：",
                buttons=[
                    ("Stats", "stats"),
                    ("Upsert (single)", "upsert"),
                    ("Upsert Batch", "batch"),
                    ("Exists (tri-state)", "exists"),
                    ("Exists Batch", "exists_batch"),
                    ("Delete", "delete"),
                    ("Clear Collection", "clear"),
                    ("Back", "back"),
                ],
                style=STYLE
            ).run()
            if ret in (None, "back"):
                return

            try:
                name = self.ask_collection_required()
                col = self.client.get_collection(name)

                if ret == "stats":
                    self.info("Stats", col.stats())

                elif ret == "upsert":
                    doc_id = input_dialog(title="Upsert", text="doc_id:", style=STYLE).run()
                    if not doc_id:
                        continue
                    text_token = input_dialog(
                        title="Upsert",
                        text="text（可直接输入，或用 @file.txt 读取文件）:",
                        style=STYLE
                    ).run()
                    if text_token is None:
                        continue
                    text = maybe_load_at_file(text_token)

                    meta_obj = json_editor("metadata JSON（可留空）", {"timestamp": int(time.time())})
                    if meta_obj is None:
                        meta_obj = {}
                    if not isinstance(meta_obj, dict):
                        raise ValueError("metadata 必须是 JSON object (dict)")
                    resp = col.upsert(doc_id=doc_id, text=text, metadata=meta_obj)
                    self.info("Queued", resp)

                elif ret == "batch":
                    tip = [{"doc_id": "d1", "text": "hello", "metadata": {"timestamp": int(time.time())}}]
                    batch_obj = json_editor("Batch JSON（list）", tip)
                    if batch_obj is None:
                        continue
                    if not isinstance(batch_obj, list):
                        raise ValueError("batch 必须是 JSON list")
                    resp = col.upsert_batch(batch_obj)
                    self.info("Queued", resp)

                elif ret == "exists":
                    doc_id = input_dialog(title="Exists", text="doc_id:", style=STYLE).run()
                    if not doc_id:
                        continue
                    inc = yes_no_dialog(title="include_pending?", text="是否把 pending 当作 exists=True？", style=STYLE).run()
                    exists = col.exists(doc_id, include_pending=bool(inc))
                    state = col.exists_state(doc_id)
                    self.info("Exists", {"doc_id": doc_id, "exists": exists, "state": state, "include_pending": bool(inc)})

                elif ret == "exists_batch":
                    tip = ["d1", "d2"]
                    ids_obj = json_editor("doc_ids JSON（list）", tip)
                    if ids_obj is None:
                        continue
                    if not isinstance(ids_obj, list):
                        raise ValueError("doc_ids 必须是 JSON list")
                    inc = yes_no_dialog(title="include_pending?", text="是否把 pending 当作 exists=True？", style=STYLE).run()
                    res = col.exists_batch(ids_obj, include_pending=bool(inc))
                    self.info("Exists Batch", {"include_pending": bool(inc), "exists_map": res})

                elif ret == "delete":
                    doc_id = input_dialog(title="Delete", text="doc_id:", style=STYLE).run()
                    if not doc_id:
                        continue
                    ok = col.delete(doc_id)
                    self.info("Delete", {"doc_id": doc_id, "deleted": bool(ok)})

                elif ret == "clear":
                    confirm = yes_no_dialog(title="Confirm", text="确认清空整个 collection？（危险操作）", style=STYLE).run()
                    if confirm:
                        ok = col.clear()
                        self.info("Cleared", {"collection": name, "cleared": bool(ok)})

            except Exception as e:
                self.error("Error", e)

    # -------- Search --------

    def menu_search(self):
        while True:
            ret = button_dialog(
                title="Search",
                text=f"当前 collection: {self.collection or '(none)'}\n\n选择操作：",
                buttons=[
                    ("Run Search", "run"),
                    ("Back", "back"),
                ],
                style=STYLE
            ).run()
            if ret in (None, "back"):
                return

            try:
                name = self.ask_collection_required()
                col = self.client.get_collection(name)

                query = input_dialog(title="Search", text="query:", style=STYLE).run()
                if not query:
                    continue
                top_n = input_dialog(title="Search", text="top_n (default 5):", style=STYLE).run()
                thr = input_dialog(title="Search", text="score_threshold (default 0.0):", style=STYLE).run()
                top_n_i = int(top_n or "5")
                thr_f = float(thr or "0.0")

                filt = None
                use_filter = yes_no_dialog(title="Filter", text="是否设置 filter_criteria？", style=STYLE).run()
                if use_filter:
                    filt_obj = json_editor("filter_criteria JSON", {"category": "news"})
                    if filt_obj is None:
                        filt_obj = None
                    if filt_obj is not None and not isinstance(filt_obj, dict):
                        raise ValueError("filter_criteria 必须是 JSON object (dict)")
                    filt = filt_obj

                res = col.search(query=query, top_n=top_n_i, score_threshold=thr_f, filter_criteria=filt)
                self.info("Search Result", res)

            except Exception as e:
                self.error("Error", e)

    # -------- Timestamp stats --------

    def menu_timestamp_stats(self):
        while True:
            ret = button_dialog(
                title="Timestamp Stats",
                text=f"当前 collection: {self.collection or '(none)'}\n\n选择操作：",
                buttons=[
                    ("Run", "run"),
                    ("Back", "back"),
                ],
                style=STYLE
            ).run()
            if ret in (None, "back"):
                return

            try:
                name = self.ask_collection_required()
                col = self.client.get_collection(name)

                tf = input_dialog(title="Timestamp Stats", text="time_field (default timestamp):", style=STYLE).run()
                sl = input_dialog(title="Timestamp Stats", text="scan_limit (default 20000):", style=STYLE).run()
                tf = (tf or "timestamp").strip()
                sl_i = int(sl or "20000")

                res = col.timestamp_stats(time_field=tf, scan_limit=sl_i)
                self.info("Timestamp Stats", res)

            except Exception as e:
                self.error("Error", e)

    # -------- Analysis --------

    def menu_analysis(self):
        while True:
            ret = button_dialog(
                title="Analysis",
                text=f"当前 collection: {self.collection or '(none)'}\n\n选择操作：",
                buttons=[
                    ("Run (submit job)", "run"),
                    ("Poll job", "poll"),
                    ("Recent jobs", "recent"),
                    ("Back", "back"),
                ],
                style=STYLE
            ).run()
            if ret in (None, "back"):
                return

            try:
                if ret == "run":
                    name = self.ask_collection_required()
                    col = self.client.get_collection(name)
                    default_cfg = {
                        "filter_criteria": {},
                        "time_range": None,
                        "limit": 20000,
                        "reduce_method": "pca",
                        "reduce_params": {},
                        "cluster_method": "birch",
                        "cluster_params": {},
                        "time_weight": 0.1
                    }
                    cfg = json_editor("AnalysisConfig JSON", default_cfg)
                    if cfg is None:
                        continue
                    if not isinstance(cfg, dict):
                        raise ValueError("AnalysisConfig 必须是 JSON object (dict)")
                    res = col.trigger_analysis(cfg)
                    jid = res.get("job_id")
                    if jid:
                        self.recent_analysis_jobs.insert(0, jid)
                        self.recent_analysis_jobs = self.recent_analysis_jobs[:50]
                    self.info("Accepted", res)

                elif ret == "poll":
                    jid = input_dialog(title="Poll Analysis Job", text="job_id:", style=STYLE).run()
                    if not jid:
                        continue
                    res = self.client.get_analysis_job(jid)
                    self.info("Job", res)

                elif ret == "recent":
                    if not self.recent_analysis_jobs:
                        self.info("Info", {"message": "no recent analysis jobs"})
                        continue
                    sel = radiolist_dialog(
                        title="Recent Analysis Jobs",
                        text="选择一个 job_id 进行查询：",
                        values=[(j, j) for j in self.recent_analysis_jobs],
                        style=STYLE
                    ).run()
                    if not sel:
                        continue
                    res = self.client.get_analysis_job(sel)
                    self.info("Job", res)

            except Exception as e:
                self.error("Error", e)

    # -------- Aggregation --------

    def menu_aggregation(self):
        while True:
            ret = button_dialog(
                title="Aggregation",
                text="选择操作：",
                buttons=[
                    ("List Plans", "plans"),
                    ("Add/Update Plan", "add"),
                    ("Delete Plan", "del"),
                    ("Run Plan (offline)", "run"),
                    ("Get Job", "job"),
                    ("Recent Jobs", "recent"),
                    ("Offline Latest", "offline"),
                    ("Online State", "online"),
                    ("Cluster Items (offline)", "items"),
                    ("Back", "back"),
                ],
                style=STYLE
            ).run()
            if ret in (None, "back"):
                return

            try:
                if ret == "plans":
                    self.info("Plans", self.client.list_aggregation_plans())

                elif ret == "add":
                    default_plan = {
                        "plan_id": "agg_demo_24h",
                        "collection_name": self.collection or "your_collection",
                        "time_window_sec": 24 * 3600,
                        "run_every_sec": 3600,
                        "filter_criteria": {},
                        "limit": 50000,
                        "max_points": 50000,
                        "method": "hdbscan",
                        "params": {"min_cluster_size": 3, "min_samples": 2},
                        "semantic_only": True,
                        "enable_online": True,
                        "online_params": {"T_event": 0.85, "T_dup": 0.95},
                        "persist": True,
                        "time_field": "timestamp",
                        "overwrite": True
                    }
                    plan = json_editor("AggregationPlan JSON", default_plan)
                    if plan is None:
                        continue
                    if not isinstance(plan, dict):
                        raise ValueError("plan 必须是 JSON object (dict)")
                    overwrite = bool(plan.pop("overwrite", True))
                    res = self.client.register_aggregation_plan(plan, overwrite=overwrite)
                    self.info("OK", res)

                elif ret == "del":
                    plan_id = input_dialog(title="Delete Plan", text="plan_id:", style=STYLE).run()
                    if not plan_id:
                        continue
                    res = self.client.delete_aggregation_plan(plan_id)
                    self.info("Deleted", res)

                elif ret == "run":
                    plan_id = input_dialog(title="Run Plan", text="plan_id:", style=STYLE).run()
                    if not plan_id:
                        continue
                    overrides = None
                    if yes_no_dialog(title="Overrides", text="是否提供 overrides（JSON）？", style=STYLE).run():
                        overrides = json_editor("Overrides JSON", {"max_points": 20000})
                        if overrides is None:
                            overrides = None
                        if overrides is not None and not isinstance(overrides, dict):
                            raise ValueError("overrides 必须是 JSON object (dict)")
                    tr = None
                    if yes_no_dialog(title="Time Range", text="是否提供 time_range？", style=STYLE).run():
                        start = input_dialog(title="Time Range", text="start_ts:", style=STYLE).run()
                        end = input_dialog(title="Time Range", text="end_ts:", style=STYLE).run()
                        if start and end:
                            tr = (float(start), float(end))
                    res = self.client.run_aggregation_plan(plan_id, overrides=overrides, time_range=tr)
                    jid = res.get("job_id")
                    if jid:
                        self.recent_agg_jobs.insert(0, jid)
                        self.recent_agg_jobs = self.recent_agg_jobs[:50]
                    self.info("Accepted", res)

                elif ret == "job":
                    jid = input_dialog(title="Get Agg Job", text="job_id:", style=STYLE).run()
                    if not jid:
                        continue
                    res = self.client.get_aggregation_job(jid)
                    self.info("Job", res)

                elif ret == "recent":
                    if not self.recent_agg_jobs:
                        self.info("Info", {"message": "no recent aggregation jobs"})
                        continue
                    sel = radiolist_dialog(
                        title="Recent Aggregation Jobs",
                        text="选择一个 job_id：",
                        values=[(j, j) for j in self.recent_agg_jobs],
                        style=STYLE
                    ).run()
                    if not sel:
                        continue
                    res = self.client.get_aggregation_job(sel)
                    self.info("Job", res)

                elif ret == "offline":
                    plan_id = input_dialog(title="Offline Latest", text="plan_id:", style=STYLE).run()
                    if not plan_id:
                        continue
                    res = self.client.get_aggregation_offline_latest(plan_id)
                    self.info("Offline Latest", res)

                elif ret == "online":
                    plan_id = input_dialog(title="Online State", text="plan_id:", style=STYLE).run()
                    if not plan_id:
                        continue
                    res = self.client.get_aggregation_online_state(plan_id)
                    self.info("Online State", res)

                elif ret == "items":
                    plan_id = input_dialog(title="Cluster Items", text="plan_id:", style=STYLE).run()
                    if not plan_id:
                        continue
                    cluster_id = input_dialog(title="Cluster Items", text="cluster_id (e.g. cluster_0):", style=STYLE).run()
                    if not cluster_id:
                        continue
                    limit = input_dialog(title="Cluster Items", text="limit (default 100):", style=STYLE).run()
                    lim = int(limit or "100")
                    res = self.client.get_aggregation_offline_cluster_items(plan_id, cluster_id, limit=lim)
                    self.info("Cluster Items", res)

            except Exception as e:
                self.error("Error", e)

    # -------- Admin (backup/restore) --------

    def menu_admin(self):
        while True:
            ret = button_dialog(
                title="Admin",
                text="选择操作：",
                buttons=[
                    ("Download Backup", "backup"),
                    ("Restore Backup", "restore"),
                    ("Back", "back"),
                ],
                style=STYLE
            ).run()
            if ret in (None, "back"):
                return

            try:
                if ret == "backup":
                    path = input_dialog(title="Backup", text="保存路径 (e.g. ./backup.zip):", style=STYLE).run()
                    if not path:
                        continue
                    resp = requests.get(f"{self.base_url}/api/admin/backup", timeout=300, stream=True)
                    resp.raise_for_status()
                    with open(path, "wb") as f:
                        for chunk in resp.iter_content(8192):
                            if chunk:
                                f.write(chunk)
                    self.info("OK", {"saved_to": os.path.abspath(path)})

                elif ret == "restore":
                    path = input_dialog(title="Restore", text="zip 文件路径:", style=STYLE).run()
                    if not path:
                        continue
                    if not os.path.exists(path):
                        raise FileNotFoundError(path)
                    with open(path, "rb") as f:
                        files = {"file": (os.path.basename(path), f, "application/zip")}
                        resp = requests.post(f"{self.base_url}/api/admin/restore", files=files, timeout=300)
                        resp.raise_for_status()
                        self.info("Restore", resp.json())

            except Exception as e:
                self.error("Error", e)

    # -------- Settings --------

    def menu_settings(self):
        while True:
            ret = button_dialog(
                title="Settings",
                text=f"URL: {self.base_url}\nCollection: {self.collection or '(none)'}\n\n选择操作：",
                buttons=[
                    ("Set Base URL", "url"),
                    ("Clear current collection", "clr"),
                    ("Back", "back"),
                ],
                style=STYLE
            ).run()
            if ret in (None, "back"):
                return
            try:
                if ret == "url":
                    new_url = input_dialog(title="Set Base URL", text="base_url:", style=STYLE).run()
                    if not new_url:
                        continue
                    self.base_url = new_url.rstrip("/")
                    self.client = VectorDBClient(self.base_url)
                    self.info("OK", {"base_url": self.base_url})
                elif ret == "clr":
                    self.collection = None
                    self.info("OK", {"collection": None})
            except Exception as e:
                self.error("Error", e)


def main():
    ap = argparse.ArgumentParser(description="VectorDB menu-driven TUI (prompt_toolkit)")
    ap.add_argument("--url", default=os.getenv("VECTORDB_URL", "http://127.0.0.1:8001"))
    args = ap.parse_args()

    TUI(args.url).run()


if __name__ == "__main__":
    main()
