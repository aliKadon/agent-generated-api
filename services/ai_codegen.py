"""
services/ai_codegen.py — AI-driven agent code generation (Plan A).

Instead of one hardcoded template per inference method (see
services/generator.py — kept intact as Plan B), a code LLM writes the
run_inference() function for the specific request. The generated function
is validated (syntax, contract, forbidden imports/calls) and repaired in a
loop, then spliced into the same trusted skeleton used by the template
generator (constants, tool implementations, tool router, CLI loop).

Plan A / Plan B switching:
  - config.USE_AI_CODEGEN = False  → generate_agent_code() goes straight to
    the template generator (Plan B). One-line revert, nothing else changes.
  - Any Plan A failure (LLM down, validation never passes) also falls back
    to Plan B automatically, so the worst case equals current behavior.

Responsibilities:
  - generate_agent_code()    : dispatcher — Plan A first, Plan B fallback
  - generate_agent_code_ai() : prompt → validate → repair → assemble file
"""

import ast
import inspect
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import InferenceClient

from config import (
    USE_AI_CODEGEN,
    CODEGEN_MODELS,
    CODEGEN_MAX_TOKENS,
    CODEGEN_REPAIR_ATTEMPTS,
    CODEGEN_FORBIDDEN_IMPORTS,
    CODEGEN_FORBIDDEN_CALLS,
    HISTORY_MAX_MESSAGES,
)
from tools.registery import ALL_TOOLS, TOOL_CODE
from tools.router import build_tool_router_code
from utils.llm import chat_completion_with_retry
from services.generator import generate_safe_agent_code


# Names the skeleton defines; AI code must not rebind them at module level.
_RESERVED_GLOBALS = frozenset({
    "client", "HISTORY", "PLAN", "SYSTEM_PROMPT",
    "MODEL_NAME", "PROVIDER", "HF_TOKEN",
    "run_tools", "run_agent", "main",
})

# Modules the AI code is allowed to import (top-level package name).
_ALLOWED_IMPORT_ROOTS = frozenset({
    "os", "io", "re", "json", "math", "time", "base64", "datetime",
    "typing", "textwrap", "random", "string", "pathlib", "urllib",
    "huggingface_hub", "PIL", "dotenv", "requests", "pypdf", "PyPDF2",
})


# ── real API signature of the target InferenceClient method ──────────────────
# Code LLMs hallucinate parameters by analogy (e.g. text_to_video(height=...)
# borrowed from text_to_image). Grounding the prompt in the INSTALLED method
# signature — and rejecting unknown kwargs in the validator — prevents agents
# that only fail at runtime.

def _inference_method_params(method_name: str) -> tuple[set[str] | None, str | None]:
    """
    Return (allowed_kwarg_names, signature_string) for the installed
    InferenceClient method, or (None, None) when it can't be introspected
    (unknown method, C-level callable, or the method takes **kwargs —
    in which case strict validation would give false positives).
    """
    fn = getattr(InferenceClient, method_name, None)
    if fn is None or not callable(fn):
        return None, None
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return None, None

    allowed: set[str] = set()
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return None, str(sig)   # **kwargs → anything goes, skip strict check
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      inspect.Parameter.KEYWORD_ONLY):
            allowed.add(name)
    return (allowed or None), str(sig)


# ── prompt building ───────────────────────────────────────────────────────────

_EXAMPLE_CHAT = '''\
def run_inference(client, model_name, plan, user_input):
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

_EXAMPLE_MEDIA = '''\
def run_inference(client, model_name, plan, user_input):
    image = client.text_to_image(prompt=user_input, model=model_name)
    import os as _os
    from datetime import datetime as _dt
    root = _os.path.dirname(_os.path.abspath(__file__))
    for _ in range(5):
        if _os.path.exists(_os.path.join(root, "api.py")):
            break
        root = _os.path.dirname(root)
    out_dir = _os.path.join(root, "generated_images")
    _os.makedirs(out_dir, exist_ok=True)
    ts   = _dt.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c for c in user_input[:40] if c.isalnum() or c in " _-").strip().replace(" ", "_") or "image"
    out_path = _os.path.join(out_dir, ts + "_" + safe + ".png")
    image.save(out_path)
    return "Image saved: " + out_path'''


def _build_codegen_system(
    inferred_method: str,
    selected_tools: list[str],
    method_sig: str | None = None,
) -> str:
    sig_note = ""
    if method_sig:
        sig_note = (
            f"\nEXACT signature of client.{inferred_method} in the INSTALLED "
            f"huggingface_hub version:\n"
            f"    client.{inferred_method}{method_sig}\n"
            "Use ONLY parameters from this signature. Do NOT borrow parameters "
            "from similar methods (e.g. text_to_video does NOT take height/width "
            "even though text_to_image does).\n"
        )
    tools_note = (
        "- run_tools(user_input) -> str : runs the agent's tools and returns "
        'combined context text ("" when nothing triggered). Prepend it to the '
        "system prompt / prompt when non-empty.\n"
        if selected_tools else
        '- run_tools(user_input) -> str : exists but always returns "" (no tools selected).\n'
    )
    return f"""\
You are an expert Python engineer. You write ONE function that will be embedded
into a generated AI agent file. Output ONLY Python code — no markdown fences,
no explanations.

You MUST define exactly this function (this exact signature):

def run_inference(client, model_name, plan, user_input):

Already defined in the file BEFORE your code (do NOT redefine or reassign them):
- client        : huggingface_hub.InferenceClient, provider already configured
- model_name    : model id string (also global MODEL_NAME)
- plan          : dict — keys include "method", "system_prompt", "final_system_prompt",
                  "prompt_template", "max_tokens", "temperature" (also global PLAN)
- SYSTEM_PROMPT : str global
- HISTORY       : list global for conversation history — trim to the last {HISTORY_MAX_MESSAGES} messages
- HF_TOKEN      : str global
{tools_note}
{sig_note}
Hard rules:
1. The inference call MUST use the InferenceClient method "{inferred_method}"
   on `client` (e.g. client.{inferred_method}(...)).
2. run_inference MUST always return a string.
3. Binary outputs must be saved to disk and the saved path returned:
   images → "generated_images/", audio/video/other files → "generated_files/".
   Find the project root by walking up from __file__ until a directory
   containing "api.py" (max 5 levels), like the example below.
4. If the method needs extra inputs (a file path, a question, an edit prompt),
   parse them from user_input: "path | question" or "path|||edit description".
   Validate the file exists and return a helpful message if not.
5. You may add module-level imports and small helper functions, but only import
   from: {", ".join(sorted(_ALLOWED_IMPORT_ROOTS))}.
6. NEVER import or use: {", ".join(CODEGEN_FORBIDDEN_IMPORTS)};
   never call: {", ".join(CODEGEN_FORBIDDEN_CALLS)}.
7. Wrap risky steps in try/except and return a clear error message string
   instead of letting the function crash.

Reference examples of good run_inference functions (adapt to THIS request,
do not copy blindly):

# Example A — chat_completion agent:
{_EXAMPLE_CHAT}

# Example B — text_to_image agent:
{_EXAMPLE_MEDIA}
"""


def _build_codegen_user(
    plan: dict,
    selected_model: str,
    provider: str | None,
    supported_task: str,
    inferred_method: str,
    selected_tools: list[str],
) -> str:
    tool_lines = [
        f"- {t}: {ALL_TOOLS[t]['description']}"
        for t in selected_tools if t in ALL_TOOLS
    ]
    return (
        f"Model: {selected_model}\n"
        f"Provider: {provider}\n"
        f"HF task: {supported_task}\n"
        f"InferenceClient method (fixed): {inferred_method}\n"
        f"Tools available via run_tools():\n"
        + ("\n".join(tool_lines) if tool_lines else "(none)") + "\n\n"
        f"Agent plan JSON:\n{json.dumps(plan, indent=2)}\n\n"
        "Write the run_inference function for this agent now."
    )


# ── extraction & validation ───────────────────────────────────────────────────

def _extract_code(text: str) -> str:
    """Strip markdown fences / prose and return the Python code block."""
    text = text.strip()
    if "```" in text:
        # take the largest fenced block
        parts, blocks = text.split("```"), []
        for i in range(1, len(parts), 2):
            block = parts[i]
            if block.startswith(("python", "py")):
                block = block.split("\n", 1)[1] if "\n" in block else ""
            blocks.append(block)
        if blocks:
            text = max(blocks, key=len).strip()
    # drop any prose lines before the first code-looking line
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        if ln.startswith(("import ", "from ", "def ", "@", "#")):
            return "\n".join(lines[i:]).strip()
    return text


def _dotted_name(node: ast.AST) -> str:
    """ast.Attribute / ast.Name → dotted string like 'os.system'."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _validate_ai_code(
    code: str,
    inferred_method: str | None = None,
    allowed_kwargs: set[str] | None = None,
) -> str | None:
    """
    Return an error description, or None if the code passes all checks.
    When inferred_method + allowed_kwargs are given, calls to that method are
    also checked for hallucinated keyword arguments against the real signature.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as ex:
        return f"SyntaxError: {ex}"

    # contract: run_inference(client, model_name, plan, user_input)
    fn = next(
        (n for n in tree.body
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "run_inference"),
        None,
    )
    if fn is None:
        return "Missing top-level `def run_inference(client, model_name, plan, user_input):`"
    got_args = [a.arg for a in fn.args.args]
    if got_args != ["client", "model_name", "plan", "user_input"]:
        return f"run_inference must take exactly (client, model_name, plan, user_input), got {got_args}"

    # module level: only imports / defs / constants; no rebinding reserved names
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                for name_node in ast.walk(t):
                    if isinstance(name_node, ast.Name) and name_node.id in _RESERVED_GLOBALS:
                        return f"Module-level code must not reassign reserved name '{name_node.id}'"

    for node in ast.walk(tree):
        # forbidden / non-allowlisted imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in CODEGEN_FORBIDDEN_IMPORTS:
                    return f"Forbidden import: {alias.name}"
                if root not in _ALLOWED_IMPORT_ROOTS:
                    return f"Import '{alias.name}' is not in the allowed list"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in CODEGEN_FORBIDDEN_IMPORTS:
                return f"Forbidden import: from {node.module}"
            if root and node.level == 0 and root not in _ALLOWED_IMPORT_ROOTS:
                return f"Import 'from {node.module}' is not in the allowed list"
        # forbidden calls (eval / exec / os.system / shutil.rmtree / …)
        elif isinstance(node, ast.Call):
            name = _dotted_name(node.func)
            if name in CODEGEN_FORBIDDEN_CALLS:
                return f"Forbidden call: {name}()"
            short = name.split(".")[-1]
            if short in ("eval", "exec", "__import__"):
                return f"Forbidden call: {name}()"
            # hallucinated-parameter check against the real installed signature
            if (
                inferred_method and allowed_kwargs
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == inferred_method
            ):
                for kw in node.keywords:
                    if kw.arg and kw.arg not in allowed_kwargs:
                        return (
                            f"client.{inferred_method}() has no parameter '{kw.arg}' "
                            f"in the installed huggingface_hub. Allowed parameters: "
                            f"{', '.join(sorted(allowed_kwargs))}"
                        )

    return None


# ── file assembly (same trusted skeleton as the template generator) ──────────

def _assemble_agent_file(
    ai_code: str,
    plan_copy: dict,
    selected_model: str,
    provider: str | None,
    supported_task: str,
    inferred_method: str,
    has_chat_template: bool,
    selected_tools: list[str],
    input_label: str,
    system_prompt: str,
) -> str:
    tool_impl   = "\n".join(TOOL_CODE[t] for t in selected_tools if t in TOOL_CODE)
    tool_router = build_tool_router_code(selected_tools)

    pip_pkgs    = [ALL_TOOLS[t]["pip"] for t in selected_tools if ALL_TOOLS[t].get("pip")]
    pip_comment = f"# pip install {' '.join(pip_pkgs)}\n" if pip_pkgs else ""

    # ACCEPTED_EXTENSIONS — parsed by the orchestrator to route uploaded files
    _ext_tool_map: dict[str, list[str]] = {
        "pdf_reader":  [".pdf"],
        "docx_reader": [".docx", ".doc"],
        "file_reader": [".txt", ".csv"],
    }
    accepted_exts: list[str] = []
    for _t in selected_tools:
        accepted_exts.extend(_ext_tool_map.get(_t, []))
    accepted_exts_line = (
        f"ACCEPTED_EXTENSIONS = {json.dumps(accepted_exts)}"
        if accepted_exts else ""
    )

    provider_str = json.dumps(provider)
    model_str    = json.dumps(selected_model)
    system_str   = json.dumps(system_prompt)
    plan_str     = json.dumps(plan_copy, indent=4)
    tools_repr   = json.dumps(selected_tools)

    lines = [
        "import os",
        "from dotenv import load_dotenv",
        "from huggingface_hub import InferenceClient",
        pip_comment,
        "load_dotenv()",
        "",
        "HF_TOKEN   = os.getenv('HF_TOKEN')",
        f"MODEL_NAME = {model_str}",
        f"PROVIDER   = {provider_str}",
        "",
        f"# codegen: ai (plan A) | HF task: {supported_task} | method: {inferred_method} | chat_template: {has_chat_template}",
        f"# Tools  : {tools_repr}",
        "",
        f"SYSTEM_PROMPT = {system_str}",
        "",
        f"PLAN = {plan_str}",
        *([accepted_exts_line, ""] if accepted_exts_line else []),
        f"# Conversation history — keeps context across turns (max {HISTORY_MAX_MESSAGES} messages)",
        "HISTORY: list = []",
        "",
        "client = InferenceClient(provider=PROVIDER, api_key=HF_TOKEN) if PROVIDER else InferenceClient(api_key=HF_TOKEN)",
        "",
        "# ── tool implementations ─────────────────────────────────────────────────",
        tool_impl,
        "",
        "# ── tool router ──────────────────────────────────────────────────────────",
        tool_router,
        "",
        "# ── inference (AI-generated) ─────────────────────────────────────────────",
        ai_code,
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
        # image_to_image and document_question_answering need two separate inputs
        *(
            [
                "        image_path  = input('\\nImage path (drag & drop): ').strip().strip('\"')",
                "        edit_prompt = input('Edit description         : ').strip()",
                "        if image_path.lower() in ['exit','quit'] or edit_prompt.lower() in ['exit','quit']:",
                "            break",
                "        user_input = image_path + '|||' + edit_prompt",
            ]
            if inferred_method == "image_to_image"
            else (
                [
                    "        doc_path = input('\\nDocument path (.pdf / .txt): ').strip().strip('\"')",
                    "        doc_q    = input('Question about the document : ').strip()",
                    "        if doc_path.lower() in ['exit','quit'] or doc_q.lower() in ['exit','quit']:",
                    "            break",
                    "        user_input = doc_path + ' | ' + doc_q",
                ]
                if inferred_method == "document_question_answering"
                else [
                    f"        user_input = input('\\n{input_label}: ').strip()",
                ]
            )
        ),
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


# ── generate_agent_code_ai (Plan A core) ──────────────────────────────────────

def generate_agent_code_ai(
    plan: dict,
    selected_model: str,
    provider: str | None,
    supported_task: str,
    inferred_method: str,
    has_chat_template: bool,
    selected_tools: list[str],
) -> str:
    """
    Ask a code LLM to write run_inference(), validate + repair it, and
    return the complete agent file. Raises RuntimeError if no valid code
    is produced within CODEGEN_REPAIR_ATTEMPTS repairs.
    """
    # Sanitise plan the same way the template generator does, so PLAN in the
    # produced file looks identical regardless of which path generated it.
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

    client       = InferenceClient(api_key=os.getenv("HF_TOKEN"))
    allowed_kwargs, method_sig = _inference_method_params(inferred_method)
    system_msg   = _build_codegen_system(inferred_method, selected_tools, method_sig)
    user_msg     = _build_codegen_user(
        plan_copy, selected_model, provider, supported_task,
        inferred_method, selected_tools,
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]

    last_error = "no attempt made"
    for attempt in range(1, CODEGEN_REPAIR_ATTEMPTS + 2):   # 1 generate + N repairs
        print(f"  [ai-codegen] attempt {attempt} …")
        r = chat_completion_with_retry(
            client, messages,
            max_tokens=CODEGEN_MAX_TOKENS,
            models=CODEGEN_MODELS,
        )
        raw     = r.choices[0].message.content or ""
        ai_code = _extract_code(raw)

        error = _validate_ai_code(ai_code, inferred_method, allowed_kwargs)
        if error is None:
            full = _assemble_agent_file(
                ai_code, plan_copy, selected_model, provider, supported_task,
                inferred_method, has_chat_template, selected_tools,
                input_label, system_prompt,
            )
            try:
                compile(full, "<generated_agent>", "exec")
            except SyntaxError as ex:
                error = f"Assembled file does not compile: {ex}"
            else:
                print("  ✅ AI-generated code passed validation")
                return full

        last_error = error
        print(f"  ⚠️  Validation failed: {error}")
        # feed the broken code + error back for a repair round
        messages = messages[:2] + [
            {"role": "assistant", "content": ai_code},
            {"role": "user", "content": (
                f"Your code failed validation:\n{error}\n\n"
                "Fix it and output ONLY the corrected Python code, "
                "keeping the exact run_inference signature and all the hard rules."
            )},
        ]

    raise RuntimeError(f"AI codegen did not pass validation: {last_error}")


# ── generate_agent_code (dispatcher: Plan A → Plan B) ─────────────────────────

def generate_agent_code_with_mode(
    plan: dict,
    selected_model: str,
    provider: str | None,
    supported_task: str,
    inferred_method: str,
    has_chat_template: bool,
    selected_tools: list[str],
) -> tuple[str, str]:
    """
    Like generate_agent_code() but also reports which path produced the code.

    Returns (code, mode) where mode is:
      "ai"       — Plan A: AI-written run_inference(), validated + repaired
      "template" — Plan B: the original template generator (USE_AI_CODEGEN
                   is False, or Plan A failed and we fell back)
    """
    if USE_AI_CODEGEN:
        try:
            code = generate_agent_code_ai(
                plan=plan, selected_model=selected_model, provider=provider,
                supported_task=supported_task, inferred_method=inferred_method,
                has_chat_template=has_chat_template, selected_tools=selected_tools,
            )
            return code, "ai"
        except Exception as ex:
            print(f"  ⚠️  AI codegen failed ({ex}) — falling back to templates (plan B)")

    code = generate_safe_agent_code(
        plan=plan, selected_model=selected_model, provider=provider,
        supported_task=supported_task, inferred_method=inferred_method,
        has_chat_template=has_chat_template, selected_tools=selected_tools,
    )
    return code, "template"


def generate_agent_code(
    plan: dict,
    selected_model: str,
    provider: str | None,
    supported_task: str,
    inferred_method: str,
    has_chat_template: bool,
    selected_tools: list[str],
) -> str:
    """Drop-in replacement for generate_safe_agent_code(). Plan A → Plan B."""
    code, _mode = generate_agent_code_with_mode(
        plan=plan, selected_model=selected_model, provider=provider,
        supported_task=supported_task, inferred_method=inferred_method,
        has_chat_template=has_chat_template, selected_tools=selected_tools,
    )
    return code
