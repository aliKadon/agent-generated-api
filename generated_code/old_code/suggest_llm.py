# pip install huggingface_hub python-dotenv requests ddgs google-adk

import os
import time
import random
from dotenv import load_dotenv
from huggingface_hub import HfApi, InferenceClient
from dataclasses import dataclass, field
import json

HF_MODEL_URL = "https://huggingface.co/"
load_dotenv()

PLANNER_MODELS = [
    "Qwen/Qwen2.5-3B-Instruct:featherless-ai",
    "Qwen/Qwen2.5-7B-Instruct:featherless-ai",
    "mistralai/Mistral-7B-Instruct-v0.3:featherless-ai",
]

TASK_TO_METHOD = {
    "text-generation":              "text_generation",
    "conversational":               "chat_completion",
    "text2text-generation":         "text_generation",
    "translation":                  "text_generation",
    "summarization":                "summarization",
    "question-answering":           "question_answering",
    "text-to-image":                "text_to_image",
    "image-to-text":                "image_to_text",
    "automatic-speech-recognition": "automatic_speech_recognition",
    "text-to-speech":               "text_to_speech",
    "text-to-audio":                "text_to_speech",
    "audio-classification":         "audio_classification",
    "image-classification":         "image_classification",
    "object-detection":             "object_detection",
    "token-classification":         "token_classification",
    "sentence-similarity":          "sentence_similarity",
    "fill-mask":                    "fill_mask",
}

CHAT_TEMPLATE_INDICATORS = ["conversational", "chat-template", "instruct"]

ALL_TOOLS = {
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
        "description": "Read content from a local file",
        "pip": None,
        "trigger_keywords": ["read file", "open file", "file content", "load file"],
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
}

# ── tool source code blocks ───────────────────────────────────────────────────

TOOL_CODE = {

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
    """Read a local file and return its content (first 4000 chars)."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()[:4000]
    except FileNotFoundError:
        return f"File not found: {filepath}"
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

# ADK-powered semantic memory — free, no API key, pure in-process
"memory": '''\
# ── ADK InMemoryMemoryService (free, no API key) ─────────────────────────────
import asyncio as _asyncio
from google.adk.memory import InMemoryMemoryService as _MemSvc
from google.adk.events import Event as _Event
from google.genai import types as _genai_types

_MEM = _MemSvc()
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

}

# ── tool router builder ───────────────────────────────────────────────────────

def build_tool_router_code(selected_tools: list) -> str:
    """
    Returns a standalone run_tools(user_input) -> str function
    with correct 4-space indentation throughout.
    """
    if not selected_tools:
        return 'def run_tools(user_input: str) -> str:\n    return ""\n'

    blocks = []

    for tool in selected_tools:
        info = ALL_TOOLS.get(tool)
        if not info:
            continue
        keywords = info.get("trigger_keywords", [])
        kw_check = " or ".join([f'"{kw}" in user_lower' for kw in keywords])

        if tool == "web_search":
            blocks.append(f'''\
    # web_search — DuckDuckGo, free, no key
    if {kw_check}:
        _r = tool_web_search(user_input)
        if _r:
            context_parts.append("[Web Search]\\n" + _r)
''')

        elif tool == "calculator":
            blocks.append('''\
    # calculator
    _calc_m = _re.search(r\'[\\d][\\d\\s+\\-*/^().]+[\\d]\', user_input)
    if _calc_m or any(w in user_lower for w in ["calculate", "compute", "math", "how much"]):
        _expr = _calc_m.group() if _calc_m else user_input
        context_parts.append("[Calculator]\\n" + tool_calculator(_expr))
''')

        elif tool == "datetime":
            blocks.append(f'''\
    # datetime
    if {kw_check}:
        context_parts.append("[Date & Time]\\n" + tool_datetime())
''')

        elif tool == "file_reader":
            blocks.append('''\
    # file_reader
    _path_m = _re.search(r\'[\\w./\\\\-]+\\.\\w+\', user_input)
    if _path_m and any(w in user_lower for w in ["read", "open", "load", "file"]):
        context_parts.append("[File]\\n" + tool_file_reader(_path_m.group()))
''')

        elif tool == "pdf_generator":
            blocks.append('''\
    # pdf_generator — call tool_pdf_generator(text, filename) manually after response
''')

        elif tool == "memory":
            blocks.append('''\
    # memory — ADK InMemoryMemoryService, semantic, free, no API key
    # STORE: "my name is Ali", "my name Ali", "call me Ali", "i am Ali", "remember X is Y"
    _name_m = _re.search(
        r\'(?:my name(?:\\s+is)?|call me|i am|im|remember me as)\\s+([\\w]+)\',
        user_input, _re.IGNORECASE
    )
    if _name_m:
        tool_memory_store(f"User\'s name is {_name_m.group(1)}")
        context_parts.append(f"[Memory] Stored: user name = {_name_m.group(1)}")
    _fact_m = _re.search(
        r\'remember(?:\\s+that)?\\s+(.+?)\\s+(?:is|=)\\s+(.+)\',
        user_input, _re.IGNORECASE
    )
    if _fact_m:
        tool_memory_store(f"{_fact_m.group(1).strip()} is {_fact_m.group(2).strip()}")
        context_parts.append(f"[Memory] Stored: {_fact_m.group(1).strip()} = {_fact_m.group(2).strip()}")
    # RECALL: "what is my name", "do you know my X", "recall", "what did i tell you"
    if _re.search(r\'what[\\W]*(is|s) my|do you know my|recall|what did i (tell|say|mention)\', user_lower):
        _mem_r = tool_memory_search(user_input)
        if _mem_r and "Nothing" not in _mem_r:
            context_parts.append("[Memory - What I remember]\\n" + _mem_r)
    # Also always do a semantic search so the LLM gets relevant context
    elif any(w in user_lower for w in ["my", "i ", "i\'m", "me"]):
        _mem_r = tool_memory_search(user_input)
        if _mem_r and "Nothing" not in _mem_r:
            context_parts.append("[Memory - Relevant context]\\n" + _mem_r)
''')

        elif tool == "translator":
            blocks.append(f'''\
    # translator — MyMemory free API, no key, 5000 chars/day
    if {kw_check}:
        _lang_m = _re.search(r\'in (\\w+)\', user_input, _re.IGNORECASE)
        _lang   = _lang_m.group(1)[:2].lower() if _lang_m else "en"
        _txt    = _re.sub(r\'translate\\s*\', "", user_input, flags=_re.IGNORECASE)
        _txt    = _re.sub(r\'in \\w+$\', "", _txt).strip()
        context_parts.append("[Translation]\\n" + tool_translator(_txt, _lang))
''')

        elif tool == "weather":
            blocks.append(f'''\
    # weather — wttr.in free API, no key
    if {kw_check}:
        _city_m = _re.search(r\'(?:in|for)\\s+([A-Za-z ]+)\', user_input)
        _city   = _city_m.group(1).strip() if _city_m else user_input
        context_parts.append("[Weather]\\n" + tool_weather(_city))
''')

    router_body = "\n".join(blocks) if blocks else "    pass\n"

    return (
        "def run_tools(user_input: str) -> str:\n"
        '    """Run relevant tools and return combined context string."""\n'
        "    import re as _re\n"
        "    context_parts = []\n"
        "    user_lower = user_input.lower()\n\n"
        + router_body
        + '\n    return "\\n".join(context_parts)\n'
    )


# ── ModelSuggestion ───────────────────────────────────────────────────────────

@dataclass
class ModelSuggestion:
    name: str
    url: str
    score: float
    is_free: bool
    has_provider: bool
    provider: str | None
    supported_task: str | None = None
    inferred_method: str | None = None
    has_chat_template: bool = False
    pipeline_tag: str | None = None
    tags: list = field(default_factory=list)


# ── retry wrapper ─────────────────────────────────────────────────────────────

def chat_completion_with_retry(client, messages, max_tokens=500, retries=3, base_delay=5.0):
    last_error = None
    for model in PLANNER_MODELS:
        for attempt in range(1, retries + 1):
            try:
                print(f"  [planner] {model}  attempt={attempt}")
                return client.chat_completion(model=model, messages=messages, max_tokens=max_tokens)
            except Exception as e:
                last_error = e
                err = str(e)
                transient = any(x in err for x in ["504", "502", "503", "timeout", "Gateway", "connection"])
                if transient and attempt < retries:
                    delay = base_delay * attempt + random.uniform(0, 2)
                    print(f"  ⚠️  Retrying in {delay:.1f}s…")
                    time.sleep(delay)
                elif attempt == retries:
                    print(f"  ❌ {model} failed. Trying next…")
                    break
                else:
                    raise
    raise RuntimeError(f"All planner models failed. Last: {last_error}")


# ── get_model_full_info ───────────────────────────────────────────────────────

def get_model_full_info(api, model_id, search_task):
    result = {
        "provider": None,
        "supported_task": search_task,
        "inferred_method": TASK_TO_METHOD.get(search_task, "text_generation"),
        "has_chat_template": False,
        "pipeline_tag": None,
        "tags": [],
    }
    try:
        info = api.model_info(model_id, expand="inferenceProviderMapping")
        result["pipeline_tag"] = getattr(info, "pipeline_tag", None)
        result["tags"] = list(getattr(info, "tags", []) or [])
        tag_text = " ".join(result["tags"]).lower()
        result["has_chat_template"] = any(
            ind in tag_text or ind in model_id.lower() for ind in CHAT_TEMPLATE_INDICATORS
        )
        mapping = getattr(info, "inference_provider_mapping", None)
        if not mapping:
            return result
        for pd in (mapping.values() if isinstance(mapping, dict) else mapping):
            status = pd.get("status") if isinstance(pd, dict) else getattr(pd, "status", None)
            prov   = pd.get("provider") if isinstance(pd, dict) else getattr(pd, "provider", None)
            ptask  = pd.get("task") if isinstance(pd, dict) else getattr(pd, "task", None)
            if status == "live":
                actual = ptask or search_task
                result["provider"] = prov
                result["supported_task"] = actual
                if actual == "conversational":
                    result["inferred_method"] = "chat_completion"
                elif actual == "text-generation" and result["has_chat_template"]:
                    result["inferred_method"] = "chat_completion"
                else:
                    result["inferred_method"] = TASK_TO_METHOD.get(actual, "text_generation")
                break
    except Exception as e:
        print(f"  ℹ️  model_info failed for {model_id}: {e}")
    return result


# ── generate_search_plan ──────────────────────────────────────────────────────

def generate_search_plan(prompt):
    client = InferenceClient(api_key=os.getenv("HF_TOKEN"))
    system = """
You are a JSON generator. Convert user request into HF model search plan.
Return ONLY valid JSON. No markdown.

Allowed HF tasks: text-generation, translation, text-to-image, image-to-text,
text-to-audio, text-to-speech, automatic-speech-recognition, audio-classification,
image-classification, object-detection, summarization, question-answering,
sentence-similarity, fill-mask, token-classification

JSON:
{
  "tasks": ["task-name"],
  "queries": ["q1", "q2", "q3"],
  "boost_words": ["w1", "w2"]
}

Rules:
- chat/conversation/coding → ["text-generation"]
- image → ["text-to-image"]
- queries MUST include 2+ of: instruct, chat, llm, qwen, mistral, llama
"""
    try:
        r = chat_completion_with_retry(client, [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ], max_tokens=200)
        text = r.choices[0].message.content.strip()
        print("Search plan:"); print(text)
        s = text.find("{"); e = text.rfind("}") + 1
        if s == -1 or e == 0: raise ValueError("No JSON")
        return json.loads(text[s:e])
    except Exception as ex:
        print(f"Search plan fallback ({ex})")
        if any(w in prompt.lower() for w in ["image", "picture", "photo"]):
            return {"tasks": ["text-to-image"], "queries": ["stable diffusion", "sdxl", "flux"], "boost_words": ["diffusion", "sdxl"]}
        return {"tasks": ["text-generation"], "queries": ["instruct", "chat", "llm", "qwen", "mistral"], "boost_words": ["instruct", "chat"]}


# ── search_best_llms ──────────────────────────────────────────────────────────

def search_best_llms(agent_prompt, limit=200):
    api  = HfApi()
    plan = generate_search_plan(agent_prompt)
    print("\nSearch plan result:"); print(plan)

    queries     = plan.get("queries",     ["chat", "instruct"])
    tasks       = plan.get("tasks",       ["text-generation"])
    boost_words = plan.get("boost_words", ["instruct", "chat"])
    if isinstance(tasks, str): tasks = [tasks]

    if any(w in agent_prompt.lower() for w in ["image", "picture", "photo"]):
        tasks = ["text-to-image"]; queries = ["stable diffusion", "sdxl", "flux"]; boost_words = ["diffusion", "sdxl"]

    suggestions = []
    for q in queries:
        for task in tasks:
            try:
                models = api.list_models(search=q, task=task, sort="downloads", direction=-1, limit=limit, full=True)
            except TypeError:
                models = api.list_models(search=q, filter=task, limit=limit, full=True)
            for model in models:
                mid = model.modelId
                if any(s.name == mid for s in suggestions): continue
                tags  = model.tags or []
                score = (model.downloads or 0) * 0.6 + (model.likes or 0) * 10
                for w in boost_words:
                    if w.lower() in mid.lower() or w.lower() in " ".join(tags).lower():
                        score += 5000
                is_free = not getattr(model, "gated", False)
                mi      = get_model_full_info(api, mid, task)
                suggestions.append(ModelSuggestion(
                    name=mid, url=HF_MODEL_URL + mid, score=score,
                    is_free=is_free, has_provider=mi["provider"] is not None,
                    provider=mi["provider"], supported_task=mi["supported_task"],
                    inferred_method=mi["inferred_method"], has_chat_template=mi["has_chat_template"],
                    pipeline_tag=mi["pipeline_tag"], tags=mi["tags"],
                ))
    suggestions.sort(key=lambda x: (not x.is_free, not x.has_provider, -x.score))
    return suggestions, tasks[0]


# ── generate_agent_prompt_plan ────────────────────────────────────────────────

def generate_agent_prompt_plan(user_agent_request, selected_model):
    client = InferenceClient(api_key=os.getenv("HF_TOKEN"))
    system = """
You are an expert AI agent prompt architect.
Return ONLY valid JSON. No markdown.

JSON:
{
  "agent_type": "short agent category",
  "clarifying_questions": ["question 1", "question 2"],
  "draft_system_prompt": "usable system prompt",
  "missing_info": ["info 1"],
  "recommended_settings": {"temperature": 0.7, "max_tokens": 800}
}
Ask 2-5 questions. Make draft_system_prompt immediately usable.
"""
    try:
        r = chat_completion_with_retry(client, [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Agent: {user_agent_request}\nModel: {selected_model}\nCreate plan."},
        ], max_tokens=500)
        text = r.choices[0].message.content.strip()
        s = text.find("{"); e = text.rfind("}") + 1
        if s == -1 or e == 0: raise ValueError("No JSON")
        return json.loads(text[s:e])
    except Exception as ex:
        print(f"Prompt plan fallback ({ex})")
        return {
            "agent_type": "general assistant",
            "clarifying_questions": ["What is the main task?", "Who uses this?", "What tone?", "Any rules?"],
            "draft_system_prompt": f"You are a professional AI assistant. Help the user with: {user_agent_request}.",
            "missing_info": ["tone", "rules"],
            "recommended_settings": {"temperature": 0.7, "max_tokens": 800},
        }


# ── decide_tools ──────────────────────────────────────────────────────────────

def decide_tools(user_agent_request, agent_type, inferred_method):
    client    = InferenceClient(api_key=os.getenv("HF_TOKEN"))
    available = list(ALL_TOOLS.keys())
    descs     = {k: v["description"] for k, v in ALL_TOOLS.items()}
    system = f"""
You are an AI tool selector.
Return ONLY valid JSON. No markdown.

Available tools: {json.dumps(available)}
Descriptions: {json.dumps(descs, indent=2)}

JSON: {{"selected_tools": ["tool1", "tool2"]}}

Rules:
- Only pick from the available list.
- chat/text agents: consider web_search, datetime, memory.
- math/finance: include calculator.
- translation: include translator.
- documents: include file_reader, pdf_generator.
- weather queries: include weather.
- method: {inferred_method}
- If method is text_to_image or audio: return {{"selected_tools": []}}
- Max 4 tools.
"""
    CHAT_DEFAULT_TOOLS = ["web_search", "memory", "datetime"]
    IMAGE_AUDIO_METHODS = ("text_to_image", "text_to_speech", "automatic_speech_recognition",
                           "audio_classification", "image_classification")

    try:
        r = chat_completion_with_retry(client, [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Agent: {agent_type}\nRequest: {user_agent_request}"},
        ], max_tokens=200)
        text = r.choices[0].message.content.strip()
        s = text.find("{"); e = text.rfind("}") + 1
        data = json.loads(text[s:e])
        validated = [t for t in data.get("selected_tools", []) if t in ALL_TOOLS]

        # If AI returns empty for a chat agent, force the defaults
        if not validated and inferred_method in ("chat_completion", "text_generation"):
            validated = CHAT_DEFAULT_TOOLS
            print(f"  AI returned no tools for chat agent — using defaults: {validated}")
        else:
            # Always ensure memory + web_search for chat agents (add if missing)
            if inferred_method in ("chat_completion", "text_generation"):
                for must_have in ["memory", "web_search"]:
                    if must_have not in validated:
                        validated.append(must_have)
                        print(f"  Auto-added required tool: {must_have}")

        print(f"  Tools selected: {validated}")
        return validated
    except Exception as ex:
        print(f"  Tool selection fallback ({ex})")
        if inferred_method in ("chat_completion", "text_generation"):
            return CHAT_DEFAULT_TOOLS
        return []


# ── generate_agent_code_plan ──────────────────────────────────────────────────

def generate_agent_code_plan(user_agent_request, selected_model, supported_task, inferred_method, has_chat_template, system_prompt):
    client = InferenceClient(api_key=os.getenv("HF_TOKEN"))
    system = f"""
You are a HF InferenceClient planner. Return ONLY valid JSON.
Method is fixed: "{inferred_method}" — do NOT change it.

JSON:
{{
  "method": "{inferred_method}",
  "input_kind": "text",
  "input_label": "short label max 8 words",
  "system_prompt": "system prompt",
  "prompt_template": "template using {{user_input}}",
  "output_kind": "text",
  "temperature": 0.7,
  "max_tokens": 800
}}

If text_generation: prompt_template = "<system>\\n\\nUser: {{user_input}}\\nAssistant:"
If chat_completion: prompt_template = "{{user_input}}"
"""
    try:
        r = chat_completion_with_retry(client, [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Request: {user_agent_request}\nModel: {selected_model}\nTask: {supported_task}\nSystem: {system_prompt}"},
        ], max_tokens=500)
        text = r.choices[0].message.content.strip()
        s = text.find("{"); e = text.rfind("}") + 1
        result = json.loads(text[s:e])
        result["method"] = inferred_method
        return result
    except Exception as ex:
        print(f"Code plan fallback ({ex})")
        pt = "{user_input}" if inferred_method == "chat_completion" else f"{system_prompt}\n\nUser: {{user_input}}\nAssistant:"
        return {"method": inferred_method, "input_kind": "text", "input_label": "Enter your message",
                "system_prompt": system_prompt, "prompt_template": pt, "output_kind": "text",
                "temperature": 0.7, "max_tokens": 800}


# ── finalize_agent_plan ───────────────────────────────────────────────────────

def finalize_agent_plan(user_agent_request, selected_model, agent_plan, answers, inferred_method, selected_tools):
    client     = InferenceClient(api_key=os.getenv("HF_TOKEN"))
    tools_json = json.dumps([{"name": t, "reason": ALL_TOOLS[t]["description"]} for t in selected_tools])
    system = f"""
You are an expert AI agent architect. Return ONLY valid JSON.
Method fixed: "{inferred_method}" — do NOT change it.
Tools fixed: {tools_json} — do NOT change them.

JSON:
{{
  "final_system_prompt": "professional system prompt",
  "method": "{inferred_method}",
  "tools": {tools_json},
  "routing_rules": ["rule 1"],
  "temperature": 0.7,
  "max_tokens": 800
}}
"""
    try:
        r = chat_completion_with_retry(client, [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Request: {user_agent_request}\nAnswers: {json.dumps(answers)}\nPlan: {json.dumps(agent_plan)}"},
        ], max_tokens=700)
        text = r.choices[0].message.content.strip()
        # Strip markdown code fences the AI sometimes wraps around JSON
        text = text.replace("```json", "").replace("```", "").strip()
        s = text.find("{"); e = text.rfind("}") + 1
        if s == -1 or e == 0: raise ValueError("No JSON found in response")
        result = json.loads(text[s:e])
        result["method"] = inferred_method
        result["tools"]  = [{"name": t, "reason": ALL_TOOLS[t]["description"]} for t in selected_tools]
        return result
    except Exception as ex:
        print(f"Finalize fallback ({ex})")
        return {
            "final_system_prompt": agent_plan.get("system_prompt", "You are a helpful assistant."),
            "method": inferred_method,
            "tools": [{"name": t, "reason": ALL_TOOLS[t]["description"]} for t in selected_tools],
            "routing_rules": [],
            "temperature": 0.7,
            "max_tokens": 800,
        }


# ── generate_safe_agent_code ──────────────────────────────────────────────────

def generate_safe_agent_code(plan, selected_model, provider, supported_task, inferred_method, has_chat_template, selected_tools):
    input_label = plan.get("input_label", "Enter your message")
    input_label = input_label.replace("\n", " ").replace('"', "'").strip()[:80] or "Enter your message"

    system_prompt = (
        plan.get("final_system_prompt")
        or plan.get("system_prompt")
        or "You are a helpful assistant."
    )

    if inferred_method == "chat_completion":
        prompt_template = plan.get("prompt_template", "{user_input}")
    else:
        prompt_template = plan.get("prompt_template", f"{system_prompt}\n\nUser: {{user_input}}\nAssistant:")
        if "{user_input}" not in prompt_template:
            prompt_template = f"{system_prompt}\n\nUser: {{user_input}}\nAssistant:"

    plan_copy = dict(plan)
    plan_copy["method"]          = inferred_method
    plan_copy["prompt_template"] = prompt_template
    plan_copy["system_prompt"]   = system_prompt

    tool_impl   = "\n".join(TOOL_CODE[t] for t in selected_tools if t in TOOL_CODE)
    tool_router = build_tool_router_code(selected_tools)
    has_tools   = bool(selected_tools)

    pip_pkgs    = [ALL_TOOLS[t]["pip"] for t in selected_tools if ALL_TOOLS[t].get("pip")]
    pip_comment = f"# pip install {' '.join(pip_pkgs)}\n" if pip_pkgs else ""

    # ── inference body (no nested f-strings) ─────────────────────────────────
    if inferred_method == "chat_completion":
        if has_tools:
            inference_body = '''\
        tool_ctx = run_tools(user_input)
        sys_msg  = plan.get("final_system_prompt") or plan.get("system_prompt") or SYSTEM_PROMPT
        if tool_ctx:
            sys_msg += "\\n\\n[Tool Results]\\n" + tool_ctx
        msgs = [{"role": "system", "content": sys_msg}] + HISTORY + [{"role": "user", "content": user_input}]
        response = client.chat_completion(model=model_name, messages=msgs,
                       max_tokens=plan.get("max_tokens", 800), temperature=plan.get("temperature", 0.7))
        reply = response.choices[0].message.content
        HISTORY.append({"role": "user", "content": user_input})
        HISTORY.append({"role": "assistant", "content": reply})
        if len(HISTORY) > 20: HISTORY[:] = HISTORY[-20:]
        return reply'''
        else:
            inference_body = '''\
        msgs = [{"role": "system", "content": plan.get("final_system_prompt") or plan.get("system_prompt") or SYSTEM_PROMPT}] + HISTORY + [{"role": "user", "content": user_input}]
        response = client.chat_completion(model=model_name, messages=msgs,
                       max_tokens=plan.get("max_tokens", 800), temperature=plan.get("temperature", 0.7))
        reply = response.choices[0].message.content
        HISTORY.append({"role": "user", "content": user_input})
        HISTORY.append({"role": "assistant", "content": reply})
        if len(HISTORY) > 20: HISTORY[:] = HISTORY[-20:]
        return reply'''

    elif inferred_method == "text_to_image":
        inference_body = '''\
        image = client.text_to_image(prompt=user_input, model=model_name)
        import os as _os
        out_dir = "generated_images"
        _os.makedirs(out_dir, exist_ok=True)
        # use first 40 chars of prompt as filename
        safe = "".join(c for c in user_input[:40] if c.isalnum() or c in " _-").strip().replace(" ", "_")
        out_path = f"{out_dir}/{safe}.png"
        image.save(out_path)
        return f"Image saved to {out_path}"'''

    elif inferred_method == "text_generation":
        if has_tools:
            inference_body = '''\
        tool_ctx = run_tools(user_input)
        prompt   = plan["prompt_template"].replace("{user_input}", user_input)
        if tool_ctx:
            prompt = tool_ctx + "\\n\\n" + prompt
        return client.text_generation(prompt, model=model_name,
                   max_new_tokens=plan.get("max_tokens", 800), temperature=plan.get("temperature", 0.7))'''
        else:
            inference_body = '''\
        prompt = plan["prompt_template"].replace("{user_input}", user_input)
        return client.text_generation(prompt, model=model_name,
                   max_new_tokens=plan.get("max_tokens", 800), temperature=plan.get("temperature", 0.7))'''

    else:
        # summarization, question_answering, automatic_speech_recognition, image_to_text, etc.
        inference_body = '''\
        return f"Method {plan['method']} is handled below."'''


    method_str       = json.dumps(inferred_method)
    provider_str     = json.dumps(provider)
    model_str        = json.dumps(selected_model)
    system_str       = json.dumps(system_prompt)
    plan_str         = json.dumps(plan_copy, indent=4)
    tools_repr       = json.dumps(selected_tools)  # double quotes inside, safe in single-quoted print()

    lines = [
        "import os",
        "from dotenv import load_dotenv",
        "from huggingface_hub import InferenceClient",
        pip_comment,
        "load_dotenv()",
        "",
        f"HF_TOKEN   = os.getenv('HF_TOKEN')",
        f"MODEL_NAME = {model_str}",
        f"PROVIDER   = {provider_str}",
        "",
        f"# HF task: {supported_task} | method: {inferred_method} | chat_template: {has_chat_template}",
        f"# Tools  : {tools_repr}",
        "",
        f"SYSTEM_PROMPT = {system_str}",
        "",
        f"PLAN = {plan_str}",
        "",
        "# Conversation history — keeps context across turns (max 20 messages = 10 turns)",
        "HISTORY: list = []",
        "",
        f"client = InferenceClient(provider=PROVIDER, api_key=HF_TOKEN)",
        "",
        "# ── tool implementations ─────────────────────────────────────────────────",
        tool_impl,
        "",
        "# ── tool router ──────────────────────────────────────────────────────────",
        tool_router,
        "",
        "# ── inference ────────────────────────────────────────────────────────────",
        "def run_inference(client, model_name, plan, user_input):",
        f"    if plan['method'] == {method_str}:",
        inference_body,
        "",
        "    # ── fallback handlers for other method types ──────────────────────────",
        "    if plan['method'] == 'automatic_speech_recognition':",
        "        return client.automatic_speech_recognition(user_input, model=model_name)",
        "",
        "    if plan['method'] == 'image_to_text':",
        "        return client.image_to_text(user_input, model=model_name)",
        "",
        "    if plan['method'] == 'summarization':",
        "        return client.summarization(user_input, model=model_name)",
        "",
        "    if plan['method'] == 'question_answering':",
        "        parts = user_input.split('|', 1)",
        "        return client.question_answering(",
        "            question=parts[0].strip(),",
        "            context=parts[1].strip() if len(parts) > 1 else '',",
        "            model=model_name,",
        "        )",
        "",
        "    raise ValueError(f\"Unsupported method: {plan['method']}\")",
        "",
        "",
        "def run_agent(user_input: str):",
        "    return run_inference(client=client, model_name=MODEL_NAME, plan=PLAN, user_input=user_input)",
        "",
        "",
        "def main():",
        "    print('Agent ready. Type exit or quit to stop.')",
        f"    print('Model   : {selected_model}')",
        f"    print('Provider: {provider}')",
        f"    print('Method  : {inferred_method}')",
        f"    print('Tools   : {tools_repr}')",
        "    while True:",
        f"        user_input = input('\\n{input_label}: ').strip()",
        "        if user_input.lower() in ['exit', 'quit']:",
        "            break",
        "        try:",
        "            result = run_agent(user_input)",
        "            print('\\nResult:')",
        "            print(result)",
        "        except Exception as e:",
        "            print('\\nError:', e)",
        "",
        "",
        "if __name__ == '__main__':",
        "    main()",
    ]

    return "\n".join(lines)


# ── create_agent_python_file ──────────────────────────────────────────────────

def create_agent_python_file(agent_name, code):
    safe  = agent_name.lower().replace(" ", "_").replace("-", "_")
    fname = f"{safe}_agent.py"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"Agent file created: {fname}")
    return fname


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Describe the AI agent you want to build:")
    prompt = input("> ").strip()
    if not prompt:
        print("Please enter a valid description."); return

    results, selected_task = search_best_llms(prompt)
    free_models = [m for m in results if m.is_free and m.has_provider]
    paid_models = [m for m in results if not m.is_free]

    print("\nBest free / open models:")
    if free_models:
        for i, m in enumerate(free_models[:20], 1):
            tag = " [chat✓]" if m.has_chat_template else ""
            print(f"{i}. {m.name}")
            print(f"   provider={m.provider} | task={m.supported_task} | method={m.inferred_method}{tag}")
            print(f"   {m.url}")
    else:
        print("No free models found.")

    print("\nPaid / gated:")
    for m in paid_models[:5]:
        print(f"- {m.name}")

    if not free_models:
        print("No free model available."); return

    selected_model = None
    while True:
        print("\nChoose a model by number:")
        c = input("> ").strip()
        try:
            selected_model = free_models[int(c) - 1]
        except Exception:
            print("Invalid choice."); continue
        if not selected_model.has_provider:
            print("❌ No provider. 1=Try again  2=Exit")
            if input("> ").strip() == "1": continue
            return
        break

    print(f"\n✅ {selected_model.name}")
    print(f"   provider={selected_model.provider} | task={selected_model.supported_task} | method={selected_model.inferred_method}")

    if not os.getenv("HF_TOKEN"):
        print("⚠️  HF_TOKEN not found in .env"); return

    print("\nCreating agent prompt plan…")
    agent_plan = generate_agent_prompt_plan(prompt, selected_model.name)
    print("\nAgent plan:"); print(json.dumps(agent_plan, indent=2))

    answers = {}
    for q in agent_plan.get("clarifying_questions", []):
        print(f"\n{q}")
        answers[q] = input("> ").strip()

    system_prompt = agent_plan["draft_system_prompt"]

    print("\nSelecting tools…")
    selected_tools = decide_tools(prompt, agent_plan.get("agent_type", ""), selected_model.inferred_method)

    if selected_tools:
        print("\nTools to be added:")
        for t in selected_tools:
            free_note = "free, no API key" if not ALL_TOOLS[t].get("pip") or t == "memory" else f"pip install {ALL_TOOLS[t]['pip']}"
            print(f"  ✅ {t} — {ALL_TOOLS[t]['description']}")
            print(f"     ({free_note})")
        print("\nKeep? (y / n / comma-separated e.g. web_search,calculator)")
        choice = input("> ").strip().lower()
        if choice == "n":
            selected_tools = []
        elif choice not in ("y", ""):
            selected_tools = [t.strip() for t in choice.split(",") if t.strip() in ALL_TOOLS]
            print(f"Using: {selected_tools}")

    code_plan = generate_agent_code_plan(
        prompt, selected_model.name, selected_model.supported_task,
        selected_model.inferred_method, selected_model.has_chat_template, system_prompt,
    )

    final_plan = finalize_agent_plan(
        prompt, selected_model.name, code_plan, answers,
        selected_model.inferred_method, selected_tools,
    )
    print("\nFinal plan:"); print(json.dumps(final_plan, indent=2))

    code = generate_safe_agent_code(
        plan=final_plan,
        selected_model=selected_model.name,
        provider=selected_model.provider,
        supported_task=selected_model.supported_task,
        inferred_method=selected_model.inferred_method,
        has_chat_template=selected_model.has_chat_template,
        selected_tools=selected_tools,
    )

    fname = create_agent_python_file(agent_plan["agent_type"], code)
    print(f"\n✅ Agent file: {fname}")
    pip_pkgs = list({ALL_TOOLS[t]["pip"] for t in selected_tools if ALL_TOOLS[t].get("pip")})
    if pip_pkgs:
        print(f"📦 Install dependencies: pip install {' '.join(pip_pkgs)}")


# if __name__ == "__main__":
#     main()