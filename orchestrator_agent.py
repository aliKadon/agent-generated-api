"""
orchestrator_agent.py — Master chat agent that routes to sub-agents and tools.

Per-message flow:
  1. Memory pass  — store/recall user facts
  2. Router LLM   — decides: agent / tool / chat
  3. Executor     — calls the chosen agent or tool
  4. Synthesizer  — produces a natural reply using the result + history
"""

import glob
import importlib.util
import json
import os
import re
import sys
import asyncio as _asyncio
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────

HF_TOKEN        = os.getenv("HF_TOKEN")
ROUTER_MODEL    = "Qwen/Qwen2.5-7B-Instruct"
ROUTER_PROVIDER = "together"

HISTORY: list = []
_AGENT_CACHE: dict = {}

client = InferenceClient(provider=ROUTER_PROVIDER, api_key=HF_TOKEN)

# ── Agent auto-discovery ───────────────────────────────────────────────────────

_ROOT      = os.path.dirname(os.path.abspath(__file__))
AGENTS:     list[dict] = []   # active agents only — used for fast-path routing + execution
ALL_AGENTS: list[dict] = []   # all agents including inactive — shown in router prompt

# Maps HF method names to a human-readable input format hint used by the router.
_METHOD_INPUT_FORMAT = {
    "image_to_image": "image_path|||edit description",
    "text_to_image":  "text prompt describing the image",
}

def _extract_agent_meta(file_path: str) -> dict | None:
    """
    Parse an agent .py file without importing it.
    Extracts SYSTEM_PROMPT and the method from PLAN to build a registry entry.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()

        # Derive a clean name from the filename: essay_writer_agent.py → essay_writer
        name = os.path.basename(file_path).replace("_agent.py", "")

        # Extract SYSTEM_PROMPT — try triple-quotes first, then single-line quotes
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

        # Extract method from PLAN dict
        m = re.search(r'"method"\s*:\s*"([^"]+)"', source)
        method = m.group(1) if m else "chat_completion"

        # Build a short description from the first sentence of the system prompt
        if system_prompt:
            first_sentence = system_prompt.split(".")[0].strip()
            description = first_sentence[:120] + ("..." if len(first_sentence) > 120 else "")
        else:
            description = f"Agent using {method}"

        return {
            "name":         name,
            "description":  description,
            "file":         os.path.relpath(file_path, _ROOT).replace("\\", "/"),
            "input_format": _METHOD_INPUT_FORMAT.get(method, "text"),
            "method":       method,
        }
    except Exception as e:
        print(f"  [discovery] skipping {file_path}: {e}")
        return None


def discover_agents() -> list[dict]:
    """
    Scan all *_agent.py files under the project root.
    Skips orchestrator_agent.py itself. Returns a fresh list every call
    so adding or deleting agents takes effect without restarting.
    """
    found = []
    for file_path in glob.glob(os.path.join(_ROOT, "**", "*_agent.py"), recursive=True):
        if os.path.basename(file_path) == "orchestrator_agent.py":
            continue
        meta = _extract_agent_meta(file_path)
        if meta:
            found.append(meta)
    return found

# ── Memory setup ───────────────────────────────────────────────────────────────

try:
    from google.adk.memory import InMemoryMemoryService as _MemSvc
    from google.adk.events import Event as _Event
    from google.genai import types as _genai_types
    _MEM       = _MemSvc()
    _MEMORY_OK = True
except ImportError:
    _MEMORY_OK = False

_MEM_APP  = "orchestrator"
_MEM_USER = "user"


def _run_async(coro):
    try:
        loop = _asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(_asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return _asyncio.run(coro)


def mem_store(text: str) -> None:
    if not _MEMORY_OK:
        return
    event = _Event(
        content=_genai_types.Content(parts=[_genai_types.Part(text=text)], role="user"),
        author="user",
    )
    _run_async(_MEM.add_events_to_memory(app_name=_MEM_APP, user_id=_MEM_USER, events=[event]))


def mem_search(query: str) -> str:
    if not _MEMORY_OK:
        return ""
    result = _run_async(_MEM.search_memory(app_name=_MEM_APP, user_id=_MEM_USER, query=query))
    if not result.memories:
        return ""
    lines = []
    for m in result.memories:
        for part in m.content.parts:
            if part.text:
                lines.append(part.text)
    return "\n".join(lines)


# ── Tools ──────────────────────────────────────────────────────────────────────

TOOLS_META = {
    "web_search":    "Search the web for current facts, news, or information",
    "calculator":    "Evaluate a math expression",
    "datetime":      "Get the current date and time",
    "weather":       "Get current weather for a city",
    "translator":    "Translate text to another language",
    "pdf_generator": "Save the last assistant reply as a PDF file",
}


def tool_web_search(query: str, max_results: int = 3) -> str:
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."
        lines = []
        for r in results:
            lines.append(f"Title  : {r.get('title', '')}")
            lines.append(f"URL    : {r.get('href', '')}")
            lines.append(f"Snippet: {r.get('body', '')}")
            lines.append("")
        return "\n".join(lines)
    except ImportError:
        return "web_search requires: pip install ddgs"
    except Exception as e:
        return f"web_search error: {e}"


def tool_calculator(expression: str) -> str:
    import math
    safe = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    safe.update({"abs": abs, "round": round, "pow": pow})
    try:
        return str(eval(expression, {"__builtins__": {}}, safe))
    except Exception as e:
        return f"calculator error: {e}"


def tool_datetime() -> str:
    return datetime.now().strftime("Date: %A, %B %d, %Y | Time: %H:%M:%S")


def tool_weather(city: str) -> str:
    try:
        import requests
        r = requests.get(f"https://wttr.in/{city}?format=3", timeout=10)
        return r.text.strip() if r.status_code == 200 else f"No weather data for: {city}"
    except Exception as e:
        return f"weather error: {e}"


def tool_translator(text: str, target_lang: str = "en") -> str:
    try:
        import requests
        r = requests.get(
            "https://api.mymemory.translated.net/get",
            params={"q": text[:500], "langpair": f"auto|{target_lang}"},
            timeout=10,
        )
        return r.json()["responseData"]["translatedText"]
    except Exception as e:
        return f"translator error: {e}"


def tool_pdf_generator(text: str, filename: str = "output.pdf") -> str:
    try:
        from fpdf import FPDF
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_files")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.basename(filename))
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for line in text.split("\n"):
            pdf.multi_cell(0, 10, line)
        pdf.output(out_path)
        return f"PDF saved: generated_files/{os.path.basename(filename)}"
    except ImportError:
        return "pdf_generator requires: pip install fpdf2"
    except Exception as e:
        return f"pdf_generator error: {e}"


# ── Sub-agent loader ───────────────────────────────────────────────────────────

def _load_agent_module(agent_cfg: dict):
    name = agent_cfg["name"]
    if name in _AGENT_CACHE:
        return _AGENT_CACHE[name]
    file_path = os.path.join(_ROOT, agent_cfg["file"].replace("/", os.sep))
    spec   = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _AGENT_CACHE[name] = module
    return module


def call_agent(agent_name: str, user_input: str) -> str:
    cfg = next((a for a in AGENTS if a["name"] == agent_name), None)
    if not cfg:
        return f"No agent named '{agent_name}' found in registry."
    try:
        module = _load_agent_module(cfg)
        return module.run_agent(user_input)
    except Exception as e:
        return f"Agent '{agent_name}' error: {e}"


# ── Router ─────────────────────────────────────────────────────────────────────

def _router_system_prompt() -> str:
    # Use ALL_AGENTS so the router knows every agent that exists, including
    # inactive ones — it marks them clearly so it can still pick the right
    # capability even if it's inactive.  The caller checks after routing.
    source = ALL_AGENTS if ALL_AGENTS else AGENTS
    active_names = {a["name"] for a in AGENTS}

    if source:
        lines = []
        for a in source:
            status = "" if a["name"] in active_names else "  [INACTIVE]"
            lines.append(
                f'  - {a["name"]}: {a["description"]}  '
                f'[input format: {a["input_format"]}]{status}'
            )
        agents_block = "\n".join(lines)
    else:
        agents_block = "  (none — no agent files found)"

    tools_block = "\n".join(f"  - {k}: {v}" for k, v in TOOLS_META.items())
    return f"""You are a routing AI. Given a user message, decide the best action.

Available sub-agents:
{agents_block}

Available tools:
{tools_block}

Return ONLY valid JSON — no markdown, no extra text:
{{
  "action": "agent" | "tool" | "chat",
  "target": "<name or null>",
  "input":  "<what to pass to the agent/tool, or null>",
  "reason": "<one-line reason>"
}}

Rules:
- Use "agent" only when the request clearly matches a sub-agent's specialty, even if it is [INACTIVE].
- Use "tool" for web search, math, date/time, weather, translation, or PDF export.
- Use "chat" for everything else — greetings, opinions, general questions, or if no agent fits.
- If no agent is listed for what the user wants, set action="chat".
- For any agent with input format "image_path|||edit description": the input MUST follow
  that format. If the user has not provided a file path, set action="chat" and ask for it.
- Never invent names — only use names from the lists above.
"""


# Maps HF method → trigger keywords for fast-path routing.
# Must stay in sync with _CAPABILITY_MAP in api.py.
_METHOD_KEYWORDS: dict[str, list[str]] = {
    "text_to_image": [
        "generate image", "create image", "make image", "draw image",
        "image of", "picture of", "generate a picture", "create a picture",
        "draw a", "paint a", "render an image", "generate an image",
        "صورة", "ارسم", "صمم",
    ],
    "text_to_video": [
        "generate video", "create video", "make video", "video of",
        "animate", "video clip", "short video", "render a video",
        "generate a video", "create a video",
        "فيديو", "انشئ فيديو",
    ],
    "image_to_image": [
        "edit image", "edit this image", "modify image", "change the image",
        "update the image", "transform image",
    ],
    "automatic_speech_recognition": [
        "transcribe", "speech to text", "convert audio", "audio to text",
        "recognize speech", "transcription",
    ],
    "summarization": [
        "summarize", "summarise", "give me a summary", "tldr",
    ],
}


def route(user_input: str) -> dict:
    # Fast-path: bypass the LLM router when the request clearly targets a specific
    # method. Small LLMs often mis-route these, so check keywords first.
    user_lower = user_input.lower()
    for method, keywords in _METHOD_KEYWORDS.items():
        if any(kw in user_lower for kw in keywords):
            matched = next((a for a in AGENTS if a.get("method") == method), None)
            if matched:
                print(f"  [router] keyword match ({method}) → {matched['name']}")
                return {
                    "action": "agent",
                    "target": matched["name"],
                    "input":  user_input,
                    "reason": f"{method} keyword match",
                }

    try:
        r = client.chat_completion(
            model=ROUTER_MODEL,
            messages=[
                {"role": "system", "content": _router_system_prompt()},
                {"role": "user",   "content": user_input},
            ],
            max_tokens=200,
            temperature=0.2,
        )
        text = r.choices[0].message.content.strip()
        s = text.find("{"); e = text.rfind("}") + 1
        if s == -1 or e == 0:
            raise ValueError("No JSON found in router response")
        return json.loads(text[s:e])
    except Exception as ex:
        print(f"  [router] fallback: {ex}")
        return {"action": "chat", "target": None, "input": None, "reason": "router error"}


# ── Memory pass ────────────────────────────────────────────────────────────────

def memory_pass(user_input: str) -> str:
    """Store/recall personal facts. Returns a context string for the synthesizer."""
    ctx        = []
    user_lower = user_input.lower()

    name_m = re.search(
        r'(?:my name(?:\s+is)?|call me|i am|im|remember me as)\s+([\w]+)',
        user_input, re.IGNORECASE,
    )
    if name_m:
        mem_store(f"User's name is {name_m.group(1)}")
        ctx.append(f"[Memory] Stored: user name = {name_m.group(1)}")

    fact_m = re.search(
        r'remember(?:\s+that)?\s+(.+?)\s+(?:is|=)\s+(.+)',
        user_input, re.IGNORECASE,
    )
    if fact_m:
        fact = f"{fact_m.group(1).strip()} is {fact_m.group(2).strip()}"
        mem_store(fact)
        ctx.append(f"[Memory] Stored: {fact}")

    if re.search(r'what[\W]*(is|s) my|do you know my|recall|what did i', user_lower):
        mem = mem_search(user_input)
        if mem:
            ctx.append(f"[Memory]\n{mem}")
    elif any(w in user_lower for w in ["my ", "i ", "i'm", "me "]):
        mem = mem_search(user_input)
        if mem:
            ctx.append(f"[Memory context]\n{mem}")

    return "\n".join(ctx)


# ── Executor ───────────────────────────────────────────────────────────────────

def execute(route_result: dict, user_input: str) -> str:
    """Run the routed action and return a raw result string."""
    action = route_result.get("action", "chat")
    target = route_result.get("target")
    inp    = route_result.get("input") or user_input

    if action == "agent":
        print(f"  [executor] → agent: {target}")
        return call_agent(target, inp)

    if action == "tool":
        print(f"  [executor] → tool: {target}")
        if target == "web_search":
            return tool_web_search(inp)
        if target == "calculator":
            m = re.search(r'[\d][\d\s+\-*/^().]+[\d]', inp)
            expr = m.group().strip() if m else inp
            return tool_calculator(expr)
        if target == "datetime":
            return tool_datetime()
        if target == "weather":
            m = re.search(r'(?:in|for)\s+([A-Za-z ]+)', inp)
            city = m.group(1).strip() if m else inp
            return tool_weather(city)
        if target == "translator":
            m    = re.search(r'\bin\s+(\w+)', inp, re.IGNORECASE)
            lang = m.group(1)[:2].lower() if m else "en"
            text = re.sub(r'(?i)translate\s*', "", inp).strip()
            text = re.sub(r'(?i)\bin\s+\w+$', "", text).strip()
            return tool_translator(text, lang)
        if target == "pdf_generator":
            last = next(
                (m["content"] for m in reversed(HISTORY) if m["role"] == "assistant"),
                inp,
            )
            return tool_pdf_generator(last)
        return f"Unknown tool: {target}"

    return ""  # "chat" — no pre-result needed


# ── Synthesizer ────────────────────────────────────────────────────────────────

def _synth_system_prompt() -> str:
    if AGENTS:
        agent_list = ", ".join(a["name"] for a in AGENTS)
        agents_note = f"You currently have these agents available: {agent_list}."
    else:
        agents_note = "You currently have NO specialized agents available."
    return (
        "You are a friendly, helpful AI assistant backed by specialized agents and tools.\n"
        f"{agents_note}\n"
        "When presenting results, be natural and concise:\n"
        "- Image results: confirm what was created. Do NOT mention file paths — the API returns the URL separately.\n"
        "- Web search: summarize the key points from the results.\n"
        "- Math / weather / time: give the answer directly and clearly.\n"
        "- Essays or long text: give a short intro then present the content.\n"
        "IMPORTANT: Never invent, assume, or hallucinate an action. "
        "If no [Result from action] is present in this prompt, NO agent or tool ran — "
        "do not pretend otherwise. Never output fake URLs like example.com or placeholder links.\n"
        "If the user asks for something that none of your available agents or tools can do, "
        "tell them clearly that you don't have that capability right now.\n"
        "Never reveal internal routing JSON or implementation details to the user."
    )


def _clean_reply(text: str) -> str:
    """Strip artefacts that small models echo back from the system prompt."""
    # Remove echoed system labels
    for prefix in ("[Result from action]", "[System]", "[Memory]", "[Memory context]"):
        if text.startswith(prefix):
            text = text[len(prefix):].lstrip("\n: ")
    # Strip hallucinated "[Image URL: https://...]" text blocks
    text = re.sub(r'\[Image URL[:\s]+https?://(?!agent-generated-api)[^\]]+\]', '', text)
    # Strip hallucinated markdown image/link syntax pointing to external URLs
    # (real file URLs are delivered via the file_url field, not embedded in the reply)
    text = re.sub(r'!\[[^\]]*\]\(https?://(?!agent-generated-api)[^\)]+\)', '', text)
    text = re.sub(r'\[[^\]]*\]\(https?://(?!agent-generated-api)[^\)]+\)', '', text)
    # Strip hallucinated file-save lines — real paths come through action_result,
    # the synthesizer must never echo them (it's told not to, but small models ignore that).
    text = re.sub(r'(?i)(Image|File|PDF|Output)\s+saved[:\s]+[^\n]+\.(png|jpg|jpeg|webp|gif|pdf|mp3|mp4|wav)', '', text)
    text = re.sub(r'generated_images[/\\][\w\-. ]+\.(?:png|jpg|jpeg|webp|gif)', '', text, flags=re.I)
    text = re.sub(r'generated_files[/\\][\w\-. ]+\.(?:pdf|mp3|mp4|wav|csv|zip)', '', text, flags=re.I)
    return text.strip()


def synthesize(user_input: str, action_result: str, mem_ctx: str, route_action: str = "chat") -> str:
    sys_msg = _synth_system_prompt()

    # Build the final user message — put context here, NOT in system message.
    # Small models echo system message content; putting results in the user turn
    # avoids that.
    context_parts = []
    if mem_ctx:
        context_parts.append(mem_ctx)
    if action_result:
        context_parts.append(f"Tool/agent result:\n{action_result}")
    elif route_action in ("agent", "tool"):
        context_parts.append(
            "Note: the agent/tool was called but returned no result. "
            "Tell the user the action could not be completed."
        )
    else:
        # route_action == "chat": no agent or tool ran at all
        context_parts.append(
            "[System] No agent or tool was invoked. "
            "Do NOT invent, simulate, or pretend to perform any action (especially image generation). "
            "Either answer from general knowledge or tell the user this capability is not available."
        )

    final_user_msg = "\n\n".join(context_parts + [user_input]) if context_parts else user_input

    msgs = [{"role": "system", "content": sys_msg}] + HISTORY + [{"role": "user", "content": final_user_msg}]
    try:
        r = client.chat_completion(
            model=ROUTER_MODEL,
            messages=msgs,
            max_tokens=800,
            temperature=0.7,
        )
        return _clean_reply(r.choices[0].message.content)
    except Exception as e:
        # if synthesis fails but we have a direct result, return it
        return action_result if action_result else f"Error: {e}"


# ── Main orchestration loop ────────────────────────────────────────────────────

def run_orchestrator(user_input: str) -> str:
    global AGENTS
    AGENTS = discover_agents()   # refresh every call — picks up new or deleted agents

    mem_ctx       = memory_pass(user_input)
    route_result  = route(user_input)
    print(f"  [router] {json.dumps(route_result)}")
    action_result = execute(route_result, user_input)
    reply         = synthesize(user_input, action_result, mem_ctx, route_action=route_result.get("action", "chat"))

    HISTORY.append({"role": "user",      "content": user_input})
    HISTORY.append({"role": "assistant", "content": reply})
    if len(HISTORY) > 20:
        HISTORY[:] = HISTORY[-20:]

    return reply


def main():
    global AGENTS
    AGENTS      = discover_agents()
    agent_names = ", ".join(a["name"] for a in AGENTS) if AGENTS else "none (run main.py to create one)"
    tool_names  = ", ".join(TOOLS_META.keys())
    print("=" * 60)
    print("Orchestrator Agent — ready")
    print(f"Agents : {agent_names}")
    print(f"Tools  : {tool_names}")
    print(f"Memory : {'enabled' if _MEMORY_OK else 'disabled (pip install google-adk)'}")
    print("Type 'exit' to quit.")
    print("=" * 60)

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        try:
            reply = run_orchestrator(user_input)
            print(f"\nAssistant: {reply}")
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
