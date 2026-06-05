import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
# pip install google-adk ddgs

load_dotenv()

HF_TOKEN   = os.getenv('HF_TOKEN')
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
PROVIDER   = "together"

# HF task: conversational | method: chat_completion | chat_template: True
# Tools  : ["memory", "web_search"]

SYSTEM_PROMPT = "system prompt"

PLAN = {
    "final_system_prompt": "system prompt",
    "method": "chat_completion",
    "tools": [
        {
            "name": "memory",
            "reason": "Semantic memory via Google ADK InMemoryMemoryService (free, no API key)"
        },
        {
            "name": "web_search",
            "reason": "Search DuckDuckGo for current information (free, no API key)"
        }
    ],
    "routing_rules": [],
    "temperature": 0.7,
    "max_tokens": 800,
    "prompt_template": "{user_input}",
    "system_prompt": "system prompt"
}

# Conversation history — keeps context across turns (max 20 messages = 10 turns)
HISTORY: list = []

client = InferenceClient(provider=PROVIDER, api_key=HF_TOKEN)

# ── tool implementations ─────────────────────────────────────────────────
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
    return "\n".join(lines) if lines else "Nothing relevant found in memory."

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
            lines.append(f"Title  : {r.get('title', '')}")
            lines.append(f"URL    : {r.get('href', '')}")
            lines.append(f"Snippet: {r.get('body', '')}")
            lines.append("")
        return "\n".join(lines)
    except ImportError:
        return "web_search requires: pip install ddgs"
    except Exception as e:
        return f"web_search error: {e}"


# ── tool router ──────────────────────────────────────────────────────────
def run_tools(user_input: str) -> str:
    """Run relevant tools and return combined context string."""
    import re as _re
    context_parts = []
    user_lower = user_input.lower()

    # memory — ADK InMemoryMemoryService, semantic, free, no API key
    # STORE: "my name is Ali", "my name Ali", "call me Ali", "i am Ali", "remember X is Y"
    _name_m = _re.search(
        r'(?:my name(?:\s+is)?|call me|i am|im|remember me as)\s+([\w]+)',
        user_input, _re.IGNORECASE
    )
    if _name_m:
        tool_memory_store(f"User's name is {_name_m.group(1)}")
        context_parts.append(f"[Memory] Stored: user name = {_name_m.group(1)}")
    _fact_m = _re.search(
        r'remember(?:\s+that)?\s+(.+?)\s+(?:is|=)\s+(.+)',
        user_input, _re.IGNORECASE
    )
    if _fact_m:
        tool_memory_store(f"{_fact_m.group(1).strip()} is {_fact_m.group(2).strip()}")
        context_parts.append(f"[Memory] Stored: {_fact_m.group(1).strip()} = {_fact_m.group(2).strip()}")
    # RECALL: "what is my name", "do you know my X", "recall", "what did i tell you"
    if _re.search(r'what[\W]*(is|s) my|do you know my|recall|what did i (tell|say|mention)', user_lower):
        _mem_r = tool_memory_search(user_input)
        if _mem_r and "Nothing" not in _mem_r:
            context_parts.append("[Memory - What I remember]\n" + _mem_r)
    # Also always do a semantic search so the LLM gets relevant context
    elif any(w in user_lower for w in ["my", "i ", "i'm", "me"]):
        _mem_r = tool_memory_search(user_input)
        if _mem_r and "Nothing" not in _mem_r:
            context_parts.append("[Memory - Relevant context]\n" + _mem_r)

    # web_search — DuckDuckGo, free, no key
    if "search" in user_lower or "find" in user_lower or "what is" in user_lower or "who is" in user_lower or "latest" in user_lower or "news" in user_lower or "current" in user_lower or "today" in user_lower:
        _r = tool_web_search(user_input)
        if _r:
            context_parts.append("[Web Search]\n" + _r)

    return "\n".join(context_parts)


# ── inference ────────────────────────────────────────────────────────────
def run_inference(client, model_name, plan, user_input):
    if plan['method'] == "chat_completion":
        tool_ctx = run_tools(user_input)
        sys_msg  = plan.get("final_system_prompt") or plan.get("system_prompt") or SYSTEM_PROMPT
        if tool_ctx:
            sys_msg += "\n\n[Tool Results]\n" + tool_ctx
        msgs = [{"role": "system", "content": sys_msg}] + HISTORY + [{"role": "user", "content": user_input}]
        response = client.chat_completion(model=model_name, messages=msgs,
                       max_tokens=plan.get("max_tokens", 800), temperature=plan.get("temperature", 0.7))
        reply = response.choices[0].message.content
        HISTORY.append({"role": "user", "content": user_input})
        HISTORY.append({"role": "assistant", "content": reply})
        if len(HISTORY) > 20: HISTORY[:] = HISTORY[-20:]
        return reply

    if plan['method'] == 'text_to_image':
        image = client.text_to_image(prompt=user_input, model=model_name)
        image.save('generated_image.png')
        return 'Image saved to generated_image.png'

    if plan['method'] == 'automatic_speech_recognition':
        return client.automatic_speech_recognition(user_input, model=model_name)

    if plan['method'] == 'image_to_text':
        return client.image_to_text(user_input, model=model_name)

    if plan['method'] == 'summarization':
        return client.summarization(user_input, model=model_name)

    if plan['method'] == 'question_answering':
        parts = user_input.split('|', 1)
        return client.question_answering(
            question=parts[0].strip(),
            context=parts[1].strip() if len(parts) > 1 else '',
            model=model_name,
        )

    raise ValueError(f"Unsupported method: {plan['method']}")


def run_agent(user_input: str):
    return run_inference(client=client, model_name=MODEL_NAME, plan=PLAN, user_input=user_input)


def main():
    print('Agent ready. Type exit or quit to stop.')
    print('Model   : Qwen/Qwen2.5-7B-Instruct')
    print('Provider: together')
    print('Method  : chat_completion')
    print('Tools   : ["memory", "web_search"]')
    while True:
        user_input = input('\nEnter your message: ').strip()
        if user_input.lower() in ['exit', 'quit']:
            break
        try:
            result = run_agent(user_input)
            print('\nResult:')
            print(result)
        except Exception as e:
            print('\nError:', e)


if __name__ == '__main__':
    main()