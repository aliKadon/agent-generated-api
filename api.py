"""
api.py — REST API for the Agent Builder project.

Agent management:
  GET    /agents                   — list all agents + total count
  GET    /agents/{id}              — get agent details + full source code
  DELETE /agents/{id}              — delete agent from DB and disk
  POST   /agents/sync              — rescan disk and update the database

Agent generation (3-step wizard):
  POST   /agents/generate/start         — step 1: describe → get model list
  POST   /agents/generate/start/stream  — step 1 with live SSE progress updates
  POST   /agents/generate/continue      — step 2: pick model → get questions + tools
  POST   /agents/generate/continue      — step 3: answer questions → agent created

Chat:
  POST   /chat                     — send a message to the orchestrator

Run locally:
  pip install fastapi uvicorn psycopg2-binary
  python api.py

Production (Render):
  Set DATABASE_URL env var to your Supabase / Neon PostgreSQL connection string.
"""

import asyncio
import glob
import json
import os
import queue
import re
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import db as _db

_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Agent file parser ──────────────────────────────────────────────────────────

_METHOD_INPUT_FORMAT = {
    "image_to_image": "image_path|||edit description",
    "text_to_image":  "text prompt describing the image",
}


def _parse_agent_file(file_path: str) -> dict | None:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()

        name = os.path.basename(file_path).replace("_agent.py", "")

        system_prompt = ""
        for pattern in [
            r'SYSTEM_PROMPT\s*=\s*"""(.*?)"""',
            r"SYSTEM_PROMPT\s*=\s*'''(.*?)'''",
            r'SYSTEM_PROMPT\s*=\s*"(.*?)"',
            r"SYSTEM_PROMPT\s*=\s*'(.*?)'",
        ]:
            m = re.search(pattern, source, re.DOTALL)
            if m:
                system_prompt = m.group(1).strip()
                break

        m      = re.search(r'"method"\s*:\s*"([^"]+)"', source)
        method = m.group(1) if m else "chat_completion"

        if system_prompt:
            first       = system_prompt.split(".")[0].strip()
            description = first[:120] + ("..." if len(first) > 120 else "")
        else:
            description = f"Agent using {method}"

        return {
            "name":         name,
            "description":  description,
            "method":       method,
            "input_format": _METHOD_INPUT_FORMAT.get(method, "text"),
            "file_path":    os.path.relpath(file_path, _ROOT).replace("\\", "/"),
            "source_code":  source,
        }
    except Exception as e:
        print(f"[sync] skipping {file_path}: {e}")
        return None


# ── Restore agent files from DB (runs on startup in production) ────────────────

def restore_agents_from_db() -> int:
    """
    Writes every agent's source_code back to disk from the database.
    Needed on hosted servers where the filesystem resets on redeploy.
    Returns the number of files restored.
    """
    restored = 0
    with _db.get_db() as conn:
        rows = _db.fetchall(conn, "SELECT name, file_path, source_code FROM agents")

    for row in rows:
        if not row["source_code"] or not row["file_path"]:
            continue
        full_path = os.path.join(_ROOT, row["file_path"])
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        if not os.path.exists(full_path):
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(row["source_code"])
            restored += 1

    return restored


# ── Sync disk → DB ─────────────────────────────────────────────────────────────

def sync_agents_to_db() -> int:
    """
    Full sync: upsert found agents, remove deleted ones.
    Safe to call from anywhere — initialises the DB if it doesn't exist yet.
    Returns total agent count after sync.
    """
    _db.init_db()
    now         = datetime.now(timezone.utc).isoformat()
    found_names: list[str]  = []
    found_metas: list[dict] = []

    for file_path in glob.glob(os.path.join(_ROOT, "**", "*_agent.py"), recursive=True):
        if os.path.basename(file_path) == "orchestrator_agent.py":
            continue
        meta = _parse_agent_file(file_path)
        if meta:
            found_names.append(meta["name"])
            found_metas.append(meta)

    with _db.get_db() as conn:
        # Snapshot is_active per file_path BEFORE any deletes/inserts so that a
        # name-change (e.g. "generating_image_agent" → "generating_image") does not
        # silently reset an inactive agent back to active.
        existing_active: dict[str, bool] = {
            r["file_path"]: bool(r["is_active"])
            for r in _db.fetchall(conn, "SELECT file_path, is_active FROM agents")
            if r.get("file_path")
        }

        if found_names:
            placeholders = ",".join(["%s" if os.getenv("DATABASE_URL") else "?"] * len(found_names))
            _db.execute(conn, f"DELETE FROM agents WHERE name NOT IN ({placeholders})", tuple(found_names))
        else:
            _db.execute(conn, "DELETE FROM agents")

        for meta in found_metas:
            _db.upsert_agent(conn, meta, now)
            # Restore is_active if this file was already tracked under a different name
            if meta["file_path"] in existing_active and not existing_active[meta["file_path"]]:
                ph = "%s" if os.getenv("DATABASE_URL") else "?"
                _db.execute(conn, f"UPDATE agents SET is_active = {ph} WHERE name = {ph}", (False, meta["name"]))

        total = _db.count_agents(conn)

    return total


# ── App lifespan ───────────────────────────────────────────────────────────────

def _load_active_state_from_db() -> None:
    """Restore _ACTIVE_AGENTS from the is_active column on server startup."""
    global _ACTIVE_AGENTS
    try:
        with _db.get_db() as conn:
            rows = _db.fetchall(conn, "SELECT name, file_path, is_active FROM agents")
        inactive = [r["name"] for r in rows if not r["is_active"]]
        if inactive:
            active_names: set[str] = set()
            for r in rows:
                if not r["is_active"]:
                    continue
                active_names.add(r["name"])
                # Also add the file-derived name so discover_agents() matches even if
                # the DB name includes an "_agent" suffix that the parser strips.
                if r.get("file_path"):
                    file_derived = os.path.basename(r["file_path"]).replace("_agent.py", "")
                    active_names.add(file_derived)
            _ACTIVE_AGENTS = active_names
            print(f"[startup] Active agents loaded — {len(_ACTIVE_AGENTS)} names, {len(inactive)} inactive")
        else:
            _ACTIVE_AGENTS = None  # all active (default)
    except Exception as e:
        print(f"[startup] Could not load active agent state: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _db.init_db()
    restored = restore_agents_from_db()
    if restored:
        print(f"[startup] Restored {restored} agent file(s) from database")
    n = sync_agents_to_db()
    print(f"[startup] Database ready — {n} agent(s) synced")
    _load_active_state_from_db()
    yield


app = FastAPI(
    title="Agent Builder API",
    description="Generate, manage, and chat with your AI agents.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Static file serving for agent-generated files ──────────────────────────────
_GENERATED_IMAGES = os.path.join(_ROOT, "generated_images")
_GENERATED_FILES  = os.path.join(_ROOT, "generated_files")
os.makedirs(_GENERATED_IMAGES, exist_ok=True)
os.makedirs(_GENERATED_FILES,  exist_ok=True)

app.mount("/files/images", StaticFiles(directory=_GENERATED_IMAGES), name="gen_images")
app.mount("/files/docs",   StaticFiles(directory=_GENERATED_FILES),  name="gen_docs")

# Patterns to extract a generated file path from an agent's reply string.
# Each entry: (compiled regex, url sub-path)
_FILE_URL_PATTERNS = [
    # image_editor_agent: "Edited image saved: generated_images/timestamp_name.png"
    (re.compile(r'generated_images[/\\]([\w\-. ]+\.(?:png|jpg|jpeg|webp|gif))', re.I), "images"),
    # text_to_image agents: "Image saved to generated_image.png" (root dir)
    (re.compile(r'saved to\s+([\w\-. ]+\.(?:png|jpg|jpeg|webp|gif))', re.I), "images"),
    # PDF: "PDF saved: generated_files/xxx.pdf"
    (re.compile(r'generated_files[/\\]([\w\-. ]+\.pdf)', re.I), "docs"),
    # Fallback PDF: "PDF saved: output.pdf"
    (re.compile(r'PDF saved[:\s]+([\w\-. ]+\.pdf)', re.I), "docs"),
    # Any other known file extension mentioned in reply
    (re.compile(r'saved[:\s]+([\w\-./\\]+\.(?:mp3|mp4|wav|csv|xlsx|zip))', re.I), "docs"),
]


# Human-readable labels for each HF method — used only in inactive-agent messages.
_METHOD_LABELS: dict[str, str] = {
    "text_to_image":               "image generation",
    "text_to_video":               "video generation",
    "image_to_image":              "image editing",
    "automatic_speech_recognition":"speech-to-text",
    "text_to_speech":              "text-to-speech",
    "summarization":               "summarization",
    "question_answering":          "question answering",
    "text_generation":             "text generation",
    "chat_completion":             "chat",
}


def _inactive_agent_reply(agent: dict) -> str:
    """Return a helpful message when the router picked an agent that is inactive."""
    name   = agent["name"]
    label  = _METHOD_LABELS.get(agent.get("method", ""), agent.get("method", "this capability"))
    return (
        f'I have a {label} agent ("{name}") but it is currently inactive. '
        f'To use it, activate it: POST /chat/agents/{name} '
        f'— or create a new {label} agent via POST /agents/generate/start'
    )


def _extract_file_url(reply: str, base_url: str) -> str | None:
    """Return a public URL if the reply contains a path to a generated file."""
    base = base_url.rstrip("/")
    for pattern, subdir in _FILE_URL_PATTERNS:
        m = pattern.search(reply)
        if m:
            fname = os.path.basename(m.group(1)).strip(" .,;:")
            # For root-level image saves ("generated_image.png"), copy to generated_images
            if subdir == "images" and not os.path.exists(os.path.join(_GENERATED_IMAGES, fname)):
                src = os.path.join(_ROOT, fname)
                if os.path.exists(src):
                    import shutil
                    shutil.copy2(src, os.path.join(_GENERATED_IMAGES, fname))
            return f"{base}/files/{subdir}/{fname}"
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Agent management schemas + routes
# ══════════════════════════════════════════════════════════════════════════════

class AgentSummary(BaseModel):
    id:           int
    name:         str
    description:  str
    method:       str
    input_format: str
    file_path:    str
    synced_at:    str


class AgentDetail(AgentSummary):
    source_code: str


class AgentsListResponse(BaseModel):
    total:  int
    agents: list[AgentSummary]


class SyncResponse(BaseModel):
    total:   int
    message: str


class DeleteResponse(BaseModel):
    id:      int
    name:    str
    message: str


@app.get("/agents", response_model=AgentsListResponse, summary="List all agents")
def list_agents():
    sync_agents_to_db()
    with _db.get_db() as conn:
        rows = _db.fetchall(
            conn,
            "SELECT id, name, description, method, input_format, file_path, synced_at FROM agents ORDER BY id",
        )
    return {"total": len(rows), "agents": rows}


@app.get(
    "/agents/{agent_id}",
    response_model=AgentDetail,
    summary="Get agent by ID (includes full source code)",
)
def get_agent(agent_id: int):
    with _db.get_db() as conn:
        row = _db.fetchone(conn, "SELECT * FROM agents WHERE id = %s" if os.getenv("DATABASE_URL") else "SELECT * FROM agents WHERE id = ?", (agent_id,))
    if not row:
        raise HTTPException(status_code=404, detail=f"No agent with id={agent_id}")
    return row


@app.delete(
    "/agents/{agent_id}",
    response_model=DeleteResponse,
    summary="Delete agent — removes from database AND deletes the .py file from disk",
)
def delete_agent(agent_id: int):
    ph = "%s" if os.getenv("DATABASE_URL") else "?"
    with _db.get_db() as conn:
        row = _db.fetchone(conn, f"SELECT id, name, file_path FROM agents WHERE id = {ph}", (agent_id,))
        if not row:
            raise HTTPException(status_code=404, detail=f"No agent with id={agent_id}")

        full_path    = os.path.join(_ROOT, row["file_path"])
        file_deleted = False
        if os.path.exists(full_path):
            os.remove(full_path)
            file_deleted = True

        _db.execute(conn, f"DELETE FROM agents WHERE id = {ph}", (agent_id,))

    status = "deleted from database and disk" if file_deleted else "deleted from database (file was already missing)"
    return {"id": row["id"], "name": row["name"], "message": f"Agent '{row['name']}' {status}."}


@app.post("/agents/sync", response_model=SyncResponse, summary="Sync agents from disk to DB")
def sync():
    total = sync_agents_to_db()
    return {"total": total, "message": f"Sync complete — {total} agent(s) in database."}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Agent generation wizard (3-step, mirrors main.py)
# ══════════════════════════════════════════════════════════════════════════════

_GEN_SESSIONS: dict[str, dict] = {}


class GenerateStartRequest(BaseModel):
    description: str


class ModelOption(BaseModel):
    index:             int
    name:              str
    provider:          str | None
    method:            str | None
    has_chat_template: bool


class GenerateStartResponse(BaseModel):
    session_id: str
    step:       str
    models:     list[ModelOption]
    message:    str


class GenerateContinueRequest(BaseModel):
    session_id:   str
    model_choice: int | None        = None
    answers:      dict[str, str] | None = None
    tool_choices: list[str] | None  = None


class ClarifyAndToolsResponse(BaseModel):
    session_id:           str
    step:                 str
    selected_model:       str
    clarifying_questions: list[str]
    suggested_tools:      list[dict]
    message:              str


class GenerateDoneResponse(BaseModel):
    session_id: str
    step:       str
    agent_id:   int | None
    name:       str
    method:     str
    file_path:  str
    message:    str


@app.post(
    "/agents/generate/start",
    response_model=GenerateStartResponse,
    summary="Step 1 — describe your agent, get a list of models to choose from",
)
def generate_start(request: GenerateStartRequest):
    import sys
    sys.path.insert(0, _ROOT)
    from services.search import search_best_llms

    try:
        results, _ = search_best_llms(request.description)
    except Exception as e:
        err = str(e)
        if "429" in err or "Too Many Requests" in err or "rate limit" in err.lower():
            import re as _re
            wait = _re.search(r"Retry after (\d+) seconds", err)
            seconds = wait.group(1) if wait else "a few"
            raise HTTPException(
                status_code=429,
                detail=f"HuggingFace rate limit hit. Wait {seconds} seconds then retry. Make sure HF_TOKEN is in your env vars.",
            )
        raise HTTPException(status_code=500, detail=f"Model search failed: {err}")

    # Prefer models that are both free AND have an active inference provider.
    # Fall back to any free model if HuggingFace has restricted provider access
    # (common for text-generation — image models are usually still available).
    free_models = [m for m in results if m.is_free and m.has_provider][:15]
    if not free_models:
        free_models = [m for m in results if m.is_free][:15]
    if not free_models:
        raise HTTPException(status_code=404, detail="No free models found. Try a different description.")

    session_id = uuid.uuid4().hex[:10]
    _GEN_SESSIONS[session_id] = {
        "step":        "select_model",
        "description": request.description,
        "free_models": free_models,
    }

    return {
        "session_id": session_id,
        "step":       "select_model",
        "models": [
            {
                "index":             i + 1,
                "name":              m.name,
                "provider":          m.provider,
                "method":            m.inferred_method,
                "has_chat_template": m.has_chat_template,
            }
            for i, m in enumerate(free_models)
        ],
        "message": "Pick a model using its index number, then call /agents/generate/continue.",
    }


@app.post(
    "/agents/generate/start/stream",
    summary="Step 1 (streaming) — same as /start but streams live progress via Server-Sent Events",
    description=(
        "Returns a text/event-stream. Each event is a JSON object:\n\n"
        "  `{\"status\": \"planning\",  \"message\": \"...\"}` — stages before searching\n\n"
        "  `{\"status\": \"searching\", \"message\": \"...\"}` — one HuggingFace query\n\n"
        "  `{\"status\": \"analyzing\", \"message\": \"...\"}` — checking providers\n\n"
        "  `{\"status\": \"progress\",  \"message\": \"...\"}` — per-model progress\n\n"
        "  `{\"status\": \"ranking\",   \"message\": \"...\"}` — final sort\n\n"
        "  `{\"status\": \"done\", \"session_id\": \"...\", \"models\": [...]}` — final result\n\n"
        "  `{\"status\": \"error\", \"message\": \"...\", \"code\": N}` — on failure\n\n"
        "Use the `session_id` from the `done` event to call `/agents/generate/continue`."
    ),
)
async def generate_start_stream(request: GenerateStartRequest):
    import sys
    sys.path.insert(0, _ROOT)
    from services.search import search_best_llms

    progress_q: queue.Queue = queue.Queue()

    def _run_search() -> None:
        def cb(status: str, message: str) -> None:
            progress_q.put({"status": status, "message": message})

        try:
            results, _ = search_best_llms(request.description, progress_cb=cb)
            free_models = [m for m in results if m.is_free and m.has_provider][:15]
            if not free_models:
                free_models = [m for m in results if m.is_free][:15]

            if not free_models:
                progress_q.put({
                    "status":  "error",
                    "message": "No free models found. Try a different description.",
                    "code":    404,
                })
                return

            session_id = uuid.uuid4().hex[:10]
            _GEN_SESSIONS[session_id] = {
                "step":        "select_model",
                "description": request.description,
                "free_models": free_models,
            }

            progress_q.put({
                "status":     "done",
                "message":    f"Found {len(free_models)} models ready to use!",
                "session_id": session_id,
                "step":       "select_model",
                "models": [
                    {
                        "index":             i + 1,
                        "name":              m.name,
                        "provider":          m.provider,
                        "method":            m.inferred_method,
                        "has_chat_template": m.has_chat_template,
                    }
                    for i, m in enumerate(free_models)
                ],
            })

        except Exception as e:
            err = str(e)
            if "429" in err or "Too Many Requests" in err or "rate limit" in err.lower():
                import re as _re
                wait = _re.search(r"Retry after (\d+) seconds", err)
                secs = wait.group(1) if wait else "a few"
                progress_q.put({
                    "status":  "error",
                    "message": f"HuggingFace rate limit. Wait {secs}s then retry. Set HF_TOKEN env var.",
                    "code":    429,
                })
            else:
                progress_q.put({"status": "error", "message": f"Model search failed: {err}", "code": 500})
        finally:
            progress_q.put(None)  # sentinel — always last

    threading.Thread(target=_run_search, daemon=True).start()

    async def event_stream():
        loop = asyncio.get_running_loop()
        while True:
            try:
                item = await loop.run_in_executor(None, lambda: progress_q.get(timeout=120))
            except queue.Empty:
                yield f"data: {json.dumps({'status': 'error', 'message': 'Search timed out after 120 seconds.', 'code': 504})}\n\n"
                break
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",      # disable Render/nginx buffering
            "Connection":        "keep-alive",
        },
    )


@app.post(
    "/agents/generate/continue",
    summary="Step 2: pick a model  |  Step 3: answer questions and create the agent",
)
def generate_continue(request: GenerateContinueRequest):
    import sys
    sys.path.insert(0, _ROOT)

    session = _GEN_SESSIONS.get(request.session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{request.session_id}' not found. Call /agents/generate/start first.",
        )

    # ── Step 2 ─────────────────────────────────────────────────────────────────
    if session["step"] == "select_model":
        if request.model_choice is None:
            raise HTTPException(status_code=422, detail="Provide 'model_choice' (1-based index).")

        free_models = session["free_models"]
        if not (1 <= request.model_choice <= len(free_models)):
            raise HTTPException(status_code=422, detail=f"model_choice must be 1–{len(free_models)}.")

        model = free_models[request.model_choice - 1]

        from services.planner import generate_agent_prompt_plan
        from services.generator import decide_tools
        from tools.registery import ALL_TOOLS

        agent_plan      = generate_agent_prompt_plan(session["description"], model.name)
        suggested_tools = decide_tools(
            session["description"],
            agent_plan.get("agent_type", ""),
            model.inferred_method,
        )

        session.update({
            "step":            "clarify_and_tools",
            "selected_model":  model,
            "agent_plan":      agent_plan,
            "suggested_tools": suggested_tools,
        })

        return ClarifyAndToolsResponse(
            session_id           = request.session_id,
            step                 = "clarify_and_tools",
            selected_model       = model.name,
            clarifying_questions = agent_plan.get("clarifying_questions", []),
            suggested_tools      = [
                {"name": t, "description": ALL_TOOLS[t]["description"]}
                for t in suggested_tools if t in ALL_TOOLS
            ],
            message = "Answer the clarifying questions in 'answers', confirm 'tool_choices', then call /continue again.",
        )

    # ── Step 3 ─────────────────────────────────────────────────────────────────
    if session["step"] == "clarify_and_tools":
        if request.answers is None:
            raise HTTPException(status_code=422, detail="Provide 'answers' dict (can be {}).")

        from services.planner import generate_agent_code_plan, finalize_agent_plan
        from services.generator import generate_safe_agent_code

        model      = session["selected_model"]
        agent_plan = session["agent_plan"]
        tools      = request.tool_choices if request.tool_choices is not None else session["suggested_tools"]

        code_plan = generate_agent_code_plan(
            session["description"], model.name, model.supported_task,
            model.inferred_method, model.has_chat_template,
            agent_plan["draft_system_prompt"],
        )
        final_plan = finalize_agent_plan(
            session["description"], model.name, code_plan,
            request.answers, model.inferred_method, tools,
        )
        code = generate_safe_agent_code(
            plan=final_plan, selected_model=model.name, provider=model.provider,
            supported_task=model.supported_task, inferred_method=model.inferred_method,
            has_chat_template=model.has_chat_template, selected_tools=tools,
        )

        out_dir   = os.path.join(_ROOT, "generated_code")
        os.makedirs(out_dir, exist_ok=True)
        safe_name = agent_plan["agent_type"].lower().replace(" ", "_").replace("-", "_")
        full_path = os.path.join(out_dir, f"{safe_name}_agent.py")
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(code)

        sync_agents_to_db()

        ph = "%s" if os.getenv("DATABASE_URL") else "?"
        with _db.get_db() as conn:
            row = _db.fetchone(conn, f"SELECT id FROM agents WHERE name = {ph}", (safe_name,))

        del _GEN_SESSIONS[request.session_id]

        return GenerateDoneResponse(
            session_id = request.session_id,
            step       = "done",
            agent_id   = row["id"] if row else None,
            name       = safe_name,
            method     = model.inferred_method,
            file_path  = os.path.relpath(full_path, _ROOT).replace("\\", "/"),
            message    = f"Agent '{safe_name}' created and saved to database.",
        )

    raise HTTPException(status_code=422, detail=f"Unknown session step: {session['step']}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Chat
# ══════════════════════════════════════════════════════════════════════════════

_CHAT_SESSIONS: dict[str, list] = {}

# None = all agents active. A set means only those names are active.
_ACTIVE_AGENTS: set[str] | None = None


class ChatAgentStatus(BaseModel):
    name:        str
    description: str
    method:      str
    active:      bool


@app.get(
    "/chat/agents",
    response_model=list[ChatAgentStatus],
    summary="List all agents with their active/inactive status for chat",
)
def list_chat_agents():
    with _db.get_db() as conn:
        rows = _db.fetchall(conn, "SELECT name, description, method, is_active FROM agents ORDER BY name")
    return [
        {
            "name":        r["name"],
            "description": r["description"],
            "method":      r["method"],
            "active":      bool(r["is_active"]),
        }
        for r in rows
    ]


@app.post(
    "/chat/agents/{name}",
    summary="Add an agent to the active chat pool",
)
def activate_chat_agent(name: str):
    global _ACTIVE_AGENTS
    ph = "%s" if os.getenv("DATABASE_URL") else "?"
    with _db.get_db() as conn:
        row = _db.fetchone(conn, f"SELECT name, file_path FROM agents WHERE name = {ph}", (name,))
        if not row:
            raise HTTPException(status_code=404, detail=f"No agent named '{name}' in database.")
        _db.execute(conn, f"UPDATE agents SET is_active = {ph} WHERE name = {ph}", (True, name))

    if _ACTIVE_AGENTS is not None:
        _ACTIVE_AGENTS.add(name)
        # Also add the file-derived name to handle DB/discover naming mismatches
        if row.get("file_path"):
            _ACTIVE_AGENTS.add(os.path.basename(row["file_path"]).replace("_agent.py", ""))

    active = sorted(_ACTIVE_AGENTS) if _ACTIVE_AGENTS is not None else []
    return {"message": f"Agent '{name}' activated.", "active_agents": active}


@app.delete(
    "/chat/agents/{name}",
    summary="Remove an agent from the active chat pool",
)
def deactivate_chat_agent(name: str):
    global _ACTIVE_AGENTS
    ph = "%s" if os.getenv("DATABASE_URL") else "?"
    with _db.get_db() as conn:
        row = _db.fetchone(conn, f"SELECT name FROM agents WHERE name = {ph}", (name,))
        if not row:
            raise HTTPException(status_code=404, detail=f"No agent named '{name}' in database.")
        _db.execute(conn, f"UPDATE agents SET is_active = {ph} WHERE name = {ph}", (False, name))

    if _ACTIVE_AGENTS is None:
        # First deactivation — initialise cache with all agents except this one
        with _db.get_db() as conn:
            all_rows = _db.fetchall(conn, "SELECT name FROM agents")
        _ACTIVE_AGENTS = {r["name"] for r in all_rows} - {name}
    else:
        _ACTIVE_AGENTS.discard(name)

    return {"message": f"Agent '{name}' deactivated.", "active_agents": sorted(_ACTIVE_AGENTS)}


class ChatRequest(BaseModel):
    message:    str
    session_id: str = "default"


class ChatResponse(BaseModel):
    reply:      str
    session_id: str
    file_url:   str | None = None


@app.post("/chat", response_model=ChatResponse, summary="Chat with the orchestrator agent")
def chat(request: ChatRequest, req: Request):
    import sys
    sys.path.insert(0, _ROOT)

    try:
        import orchestrator_agent as _oa
        from orchestrator_agent import discover_agents, memory_pass, route, execute, synthesize

        all_agents    = discover_agents()
        active_agents = (
            [a for a in all_agents if a["name"] in _ACTIVE_AGENTS]
            if _ACTIVE_AGENTS is not None
            else all_agents
        )
        _oa.AGENTS     = active_agents   # active only — used by fast-path + execute
        _oa.ALL_AGENTS = all_agents      # all agents — shown in router prompt with [INACTIVE] tags
        print(f"  [chat] active={[a['name'] for a in active_agents]}  "
              f"all={[a['name'] for a in all_agents]}")

        history       = _CHAT_SESSIONS.setdefault(request.session_id, [])
        mem_ctx       = memory_pass(request.message)
        route_result  = route(request.message)

        # If the router picked an inactive agent, return a helpful message immediately
        # without trying to execute it.  This is fully dynamic — no keyword lists needed.
        if route_result.get("action") == "agent":
            target       = route_result.get("target")
            active_names = {a["name"] for a in active_agents}
            if target and target not in active_names:
                inactive = next((a for a in all_agents if a["name"] == target), None)
                if inactive:
                    reply = _inactive_agent_reply(inactive)
                    history.append({"role": "user",      "content": request.message})
                    history.append({"role": "assistant", "content": reply})
                    return {"reply": reply, "session_id": request.session_id, "file_url": None}

        action_result = execute(route_result, request.message)
        reply         = synthesize(request.message, action_result, mem_ctx, route_action=route_result.get("action", "chat"))

        print(f"  [chat] route={route_result}")
        print(f"  [chat] action_result={action_result!r}")

        history.append({"role": "user",      "content": request.message})
        history.append({"role": "assistant", "content": reply})
        if len(history) > 20:
            history[:] = history[-20:]

        base = str(req.base_url)
        file_url = _extract_file_url(action_result, base) or _extract_file_url(reply, base)
        print(f"  [chat] file_url={file_url!r}")
        return {"reply": reply, "session_id": request.session_id, "file_url": file_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Dev server ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
