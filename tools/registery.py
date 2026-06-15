"""
tools/registry.py — static tool metadata and source-code templates.

ALL_TOOLS  : metadata the planner uses to select and describe tools.
TOOL_CODE  : the actual Python source that gets injected into generated agents.
"""

ALL_TOOLS: dict[str, dict] = {
    "web_search": {
        "description": "Search DuckDuckGo for current information (free, no API key)",
        "pip": "ddgs",
        "trigger_keywords": ["search", "find", "what is", "who is", "latest", "news", "current", "today"],
    },
    "calculator": {
        "description": "Safely evaluate math expressions",
        "pip": None,
        "trigger_keywords": ["calculate", "math", "compute", "sum", "multiply", "divide", "equation", "how much is"],
    },
    "datetime": {
        "description": "Get current date and time",
        "pip": None,
        "trigger_keywords": ["time", "date", "today", "now", "current time", "what day"],
    },
    "file_reader": {
        "description": "Read content from a local file (PDF, txt, docx, etc.)",
        "pip": "pypdf",
        "trigger_keywords": ["read file", "open file", "file content", "load file",
                             ".pdf", ".txt", ".docx", "from file", "file path", "analyze file"],
    },
    "pdf_generator": {
        "description": "Save output as a PDF file",
        "pip": "fpdf2",
        "trigger_keywords": ["save pdf", "export pdf", "generate pdf", "pdf report"],
    },
    "memory": {
        "description": "Semantic memory via Google ADK InMemoryMemoryService (free, no API key)",
        "pip": "google-adk",
        "trigger_keywords": ["remember", "recall", "my name", "call me", "i am", "what is my", "what did i"],
    },
    "translator": {
        "description": "Translate text via MyMemory free API (no API key, 5000 chars/day)",
        "pip": "requests",
        "trigger_keywords": ["translate", "translation", "in french", "in spanish", "in arabic", "in german"],
    },
    "weather": {
        "description": "Get current weather via wttr.in (free, no API key)",
        "pip": "requests",
        "trigger_keywords": ["weather", "temperature", "forecast", "rain", "sunny", "hot", "cold"],
    },
    "image_saver": {
        "description": "Save generated image with timestamp and metadata (free, no API key)",
        "pip": "pillow",
        "trigger_keywords": [],  # always runs for image agents — no keyword trigger needed
    },
    "image_upscaler": {
        "description": "Upscale image 2x using Replicate free API",
        "pip": "requests",
        "trigger_keywords": ["upscale", "larger", "bigger", "hd", "high resolution", "enhance"],
    },
}

# ── Tool source code blocks ───────────────────────────────────────────────────
# Each value is a self-contained Python function (or group of functions) that
# gets embedded verbatim into the generated agent file.

TOOL_CODE: dict[str, str] = {

"web_search": '''\
def tool_web_search(query: str, max_results: int = 3) -> str:
    """DuckDuckGo search — free, no API key needed. pip install ddgs"""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."
        lines = []
        for r in results:
            lines.append(f"Title  : {r.get(\'title\', \'\')}")
            lines.append(f"URL    : {r.get(\'href\', \'\')}")
            lines.append(f"Snippet: {r.get(\'body\', \'\')}")
            lines.append("")
        return "\\n".join(lines)
    except ImportError:
        return "web_search requires: pip install ddgs"
    except Exception as e:
        return f"web_search error: {e}"
''',

"calculator": '''\
def tool_calculator(expression: str) -> str:
    """Safely evaluate a math expression using Python built-ins."""
    import math
    allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed.update({"abs": abs, "round": round, "pow": pow})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"calculator error: {e}"
''',

"datetime": '''\
def tool_datetime() -> str:
    """Return the current local date and time."""
    from datetime import datetime
    return datetime.now().strftime("Date: %A, %B %d, %Y | Time: %H:%M:%S")
''',

"file_reader": '''\
def tool_file_reader(filepath: str) -> str:
    """Read a local file (PDF or text) and return its content (first 8000 chars). pip install pypdf"""
    import os as _os
    # handle "file_path | question" format — extract just the path
    if "|" in filepath:
        filepath = filepath.split("|", 1)[0]
    filepath = filepath.strip().strip(\'"\').strip("\'")
    if not _os.path.exists(filepath):
        return f"File not found: {filepath}"
    try:
        if filepath.lower().endswith(".pdf"):
            try:
                from pypdf import PdfReader as _PR
                _rdr = _PR(filepath)
                text = "\\n".join(p.extract_text() or "" for p in _rdr.pages)
                return text[:8000] if text.strip() else "[PDF has no extractable text]"
            except ImportError:
                try:
                    from PyPDF2 import PdfReader as _PR
                    _rdr = _PR(filepath)
                    text = "\\n".join(p.extract_text() or "" for p in _rdr.pages)
                    return text[:8000] if text.strip() else "[PDF has no extractable text]"
                except ImportError:
                    return "[PDF reading requires: pip install pypdf]"
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()[:8000]
    except Exception as e:
        return f"file_reader error: {e}"
''',

"pdf_generator": '''\
def tool_pdf_generator(text: str, filename: str = "output.pdf") -> str:
    """Save text to a PDF file. pip install fpdf2"""
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for line in text.split("\\n"):
            pdf.multi_cell(0, 10, line)
        pdf.output(filename)
        return f"PDF saved: {filename}"
    except ImportError:
        return "pdf_generator requires: pip install fpdf2"
    except Exception as e:
        return f"pdf_generator error: {e}"
''',

"memory": '''\
# ── ADK InMemoryMemoryService (free, no API key) ─────────────────────────────
import asyncio as _asyncio
from google.adk.memory import InMemoryMemoryService as _MemSvc
from google.adk.events import Event as _Event
from google.genai import types as _genai_types

_MEM  = _MemSvc()
_APP  = "agent"
_USER = "user"

def _run(coro):
    """Run an async coroutine from sync code."""
    try:
        loop = _asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(_asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return _asyncio.run(coro)

def tool_memory_store(text: str) -> str:
    """Store a piece of text into ADK semantic memory."""
    event = _Event(
        content=_genai_types.Content(
            parts=[_genai_types.Part(text=text)],
            role="user",
        ),
        author="user",
    )
    _run(_MEM.add_events_to_memory(app_name=_APP, user_id=_USER, events=[event]))
    return f"Stored in memory: {text}"

def tool_memory_search(query: str) -> str:
    """Search semantic memory for relevant facts."""
    result = _run(_MEM.search_memory(app_name=_APP, user_id=_USER, query=query))
    if not result.memories:
        return "Nothing relevant found in memory."
    lines = []
    for m in result.memories:
        for part in m.content.parts:
            if part.text:
                lines.append(part.text)
    return "\\n".join(lines) if lines else "Nothing relevant found in memory."
''',

"translator": '''\
def tool_translator(text: str, target_lang: str = "en") -> str:
    """Translate text using MyMemory free API (no key, 5000 chars/day)."""
    try:
        import requests
        r = requests.get(
            "https://api.mymemory.translated.net/get",
            params={"q": text[:500], "langpair": f"auto|{target_lang}"},
            timeout=10,
        )
        data = r.json()
        return data["responseData"]["translatedText"]
    except ImportError:
        return "translator requires: pip install requests"
    except Exception as e:
        return f"translator error: {e}"
''',

"weather": '''\
def tool_weather(city: str) -> str:
    """Get current weather using wttr.in (free, no API key)."""
    try:
        import requests
        r = requests.get(f"https://wttr.in/{city}?format=3", timeout=10)
        return r.text.strip() if r.status_code == 200 else f"Could not get weather for: {city}"
    except ImportError:
        return "weather requires: pip install requests"
    except Exception as e:
        return f"weather error: {e}"
''',

# ── image_saver ───────────────────────────────────────────────────────────────
# Automatically saves every generated image with a timestamp in the filename
# and writes basic metadata (prompt, timestamp, model) into a sidecar .txt file.
# No API key needed — uses only Pillow (pip install pillow).
# trigger_keywords is empty because this runs unconditionally for image agents.
"image_saver": '''\
def tool_image_saver(image_obj, prompt: str, model_name: str = "unknown") -> str:
    """
    Save a PIL Image with a timestamp filename + a sidecar metadata .txt file.
    Returns the saved image path.
    pip install pillow
    """
    try:
        from PIL import Image as _Image
        from datetime import datetime as _dt
        import os as _os

        # ── output folder — walk up from __file__ until api.py is found ──────
        _here = _os.path.abspath(__file__)
        _proj_root = _os.path.dirname(_here)
        for _ in range(5):
            if _os.path.exists(_os.path.join(_proj_root, "api.py")):
                break
            _proj_root = _os.path.dirname(_proj_root)
        out_dir = _os.path.join(_proj_root, "generated_images")
        _os.makedirs(out_dir, exist_ok=True)

        # ── build filename from timestamp + first 40 chars of prompt ───────
        ts        = _dt.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in prompt[:40] if c.isalnum() or c in " _-")
        safe_name = safe_name.strip().replace(" ", "_") or "image"
        img_path  = _os.path.join(out_dir, f"{ts}_{safe_name}.png")
        txt_path  = _os.path.join(out_dir, f"{ts}_{safe_name}.txt")

        # ── save the image ─────────────────────────────────────────────────
        image_obj.save(img_path)

        # ── write sidecar metadata file ────────────────────────────────────
        with open(txt_path, "w", encoding="utf-8") as _f:
            _f.write(f"Timestamp : {_dt.now().isoformat()}\\n")
            _f.write(f"Model     : {model_name}\\n")
            _f.write(f"Prompt    : {prompt}\\n")
            _f.write(f"File      : {img_path}\\n")

        return f"Image saved: {img_path}\\nMetadata : {txt_path}"

    except ImportError:
        return "image_saver requires: pip install pillow"
    except Exception as e:
        return f"image_saver error: {e}"
''',

# ── image_upscaler ────────────────────────────────────────────────────────────
# Upscales a saved image 2× using the Replicate API (free tier available).
# Only triggered when the user's prompt contains keywords like "upscale", "hd".
# Requires a free REPLICATE_API_TOKEN in .env — sign up at replicate.com.
"image_upscaler": '''\
def tool_image_upscaler(image_path: str) -> str:
    """
    Upscale an image 2x using Replicate free tier (nightmareai/real-esrgan).
    Requires REPLICATE_API_TOKEN in .env — free account at replicate.com.
    pip install requests
    """
    try:
        import requests as _requests
        import os as _os
        import time as _time

        token = _os.getenv("REPLICATE_API_TOKEN")
        if not token:
            return "image_upscaler requires REPLICATE_API_TOKEN in .env (free at replicate.com)"

        # ── read image as base64 ───────────────────────────────────────────
        import base64 as _b64
        with open(image_path, "rb") as _f:
            img_b64 = _b64.b64encode(_f.read()).decode()
        data_uri = f"data:image/png;base64,{img_b64}"

        # ── start prediction ───────────────────────────────────────────────
        headers = {"Authorization": f"Token {token}", "Content-Type": "application/json"}
        r = _requests.post(
            "https://api.replicate.com/v1/predictions",
            headers=headers,
            json={
                "version": "42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b",
                "input": {"image": data_uri, "scale": 2},
            },
            timeout=30,
        )
        prediction = r.json()
        poll_url   = prediction.get("urls", {}).get("get")
        if not poll_url:
            return f"image_upscaler: failed to start — {prediction.get(\'detail\', \'unknown error\')}"

        # ── poll until complete (max 60s) ──────────────────────────────────
        for _ in range(20):
            _time.sleep(3)
            result = _requests.get(poll_url, headers=headers, timeout=15).json()
            status = result.get("status")
            if status == "succeeded":
                output_url = result["output"]
                # ── download upscaled image ────────────────────────────────
                img_data  = _requests.get(output_url, timeout=30).content
                out_path  = image_path.replace(".png", "_upscaled.png")
                with open(out_path, "wb") as _f:
                    _f.write(img_data)
                return f"Upscaled image saved: {out_path}"
            elif status == "failed":
                return f"image_upscaler: prediction failed — {result.get(\'error\', \'unknown\')}"

        return "image_upscaler: timed out waiting for result (>60s)"

    except FileNotFoundError:
        return f"image_upscaler: file not found — {image_path}"
    except ImportError:
        return "image_upscaler requires: pip install requests"
    except Exception as e:
        return f"image_upscaler error: {e}"
''',

}