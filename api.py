"""
api.py — REST API for the Agent Builder project.

Agent management:
  GET    /agents                   — list all agents + total count
  GET    /agents/{id}              — get agent details + full source code
  DELETE /agents/{id}              — delete agent from DB and disk
  POST   /agents/sync              — rescan disk and update the database

Agent generation (3-step wizard):
  POST   /agents/generate/start    — step 1: describe → get model list
  POST   /agents/generate/continue — step 2: pick model → get questions + tools
  POST   /agents/generate/continue — step 3: answer questions → agent created

Chat:
  POST   /chat                     — send a message to the orchestrator

Run locally:
  pip install fastapi uvicorn psycopg2-binary
  python api.py

Production (Render):
  Set DATABASE_URL env var to your Supabase / Neon PostgreSQL connection string.
"""

import glob
import os
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
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
        if found_names:
            placeholders = ",".join(["%s" if os.getenv("DATABASE_URL") else "?"] * len(found_names))
            _db.execute(conn, f"DELETE FROM agents WHERE name NOT IN ({placeholders})", tuple(found_names))
        else:
            _db.execute(conn, "DELETE FROM agents")

        for meta in found_metas:
            _db.upsert_agent(conn, meta, now)

        total = _db.count_agents(conn)

    return total


# ── App lifespan ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _db.init_db()
    restored = restore_agents_from_db()
    if restored:
        print(f"[startup] Restored {restored} agent file(s) from database")
    n = sync_agents_to_db()
    print(f"[startup] Database ready — {n} agent(s) synced")
    yield


app = FastAPI(
    title="Agent Builder API",
    description="Generate, manage, and chat with your AI agents.",
    version="1.0.0",
    lifespan=lifespan,
)


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

    free_models = [m for m in results if m.is_free and m.has_provider][:15]
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
        rows = _db.fetchall(conn, "SELECT name, description, method FROM agents ORDER BY name")
    return [
        {
            "name":        r["name"],
            "description": r["description"],
            "method":      r["method"],
            "active":      _ACTIVE_AGENTS is None or r["name"] in _ACTIVE_AGENTS,
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
        row = _db.fetchone(conn, f"SELECT name FROM agents WHERE name = {ph}", (name,))
    if not row:
        raise HTTPException(status_code=404, detail=f"No agent named '{name}' in database.")

    if _ACTIVE_AGENTS is None:
        # All were active — keep all, nothing changes
        return {"message": f"Agent '{name}' is already active (all agents are active by default)."}

    _ACTIVE_AGENTS.add(name)
    return {"message": f"Agent '{name}' added to chat.", "active_agents": sorted(_ACTIVE_AGENTS)}


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

    if _ACTIVE_AGENTS is None:
        # First deactivation — initialise the set with all agents except this one
        with _db.get_db() as conn:
            all_rows = _db.fetchall(conn, "SELECT name FROM agents")
        _ACTIVE_AGENTS = {r["name"] for r in all_rows} - {name}
    else:
        _ACTIVE_AGENTS.discard(name)

    return {"message": f"Agent '{name}' removed from chat.", "active_agents": sorted(_ACTIVE_AGENTS)}


class ChatRequest(BaseModel):
    message:    str
    session_id: str = "default"


class ChatResponse(BaseModel):
    reply:      str
    session_id: str


@app.post("/chat", response_model=ChatResponse, summary="Chat with the orchestrator agent")
def chat(request: ChatRequest):
    import sys
    sys.path.insert(0, _ROOT)

    try:
        import orchestrator_agent as _oa
        from orchestrator_agent import discover_agents, memory_pass, route, execute, synthesize

        all_agents = discover_agents()
        _oa.AGENTS  = (
            [a for a in all_agents if a["name"] in _ACTIVE_AGENTS]
            if _ACTIVE_AGENTS is not None
            else all_agents
        )
        history       = _CHAT_SESSIONS.setdefault(request.session_id, [])
        mem_ctx       = memory_pass(request.message)
        route_result  = route(request.message)
        action_result = execute(route_result, request.message)
        reply         = synthesize(request.message, action_result, mem_ctx)

        history.append({"role": "user",      "content": request.message})
        history.append({"role": "assistant", "content": reply})
        if len(history) > 20:
            history[:] = history[-20:]

        return {"reply": reply, "session_id": request.session_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Dev server ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
