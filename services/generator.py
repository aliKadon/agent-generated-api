"""
services/generator.py — agent code generation and tool selection.

Responsibilities:
  - decide_tools()              : ask the planner which tools to include
  - generate_safe_agent_code()  : assemble the final agent .py source
  - create_agent_python_file()  : write that source to disk
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import InferenceClient

from config import (
    CHAT_DEFAULT_TOOLS,
    CHAT_REQUIRED_TOOLS,
    IMAGE_AUDIO_METHODS,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    HISTORY_MAX_MESSAGES,
)
from tools.registery import ALL_TOOLS, TOOL_CODE
from tools.router import build_tool_router_code
from utils.llm import chat_completion_with_retry


# ── decide_tools ──────────────────────────────────────────────────────────────

def decide_tools(
    user_agent_request: str,
    agent_type: str,
    inferred_method: str,
) -> list[str]:
    """
    Ask the planner LLM to pick tools from ALL_TOOLS.
    Falls back to CHAT_DEFAULT_TOOLS for chat agents, [] for media agents.
    """
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
    is_chat = inferred_method in ("chat_completion", "text_generation")

    try:
        r = chat_completion_with_retry(
            client,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Agent: {agent_type}\nRequest: {user_agent_request}"},
            ],
            max_tokens=200,
        )
        text = r.choices[0].message.content.strip()
        s = text.find("{"); e = text.rfind("}") + 1
        data      = json.loads(text[s:e])
        validated = [t for t in data.get("selected_tools", []) if t in ALL_TOOLS]

        if not validated and is_chat:
            validated = list(CHAT_DEFAULT_TOOLS)
            print(f"  AI returned no tools for chat agent — using defaults: {validated}")
        elif is_chat:
            for must_have in CHAT_REQUIRED_TOOLS:
                if must_have not in validated:
                    validated.append(must_have)
                    print(f"  Auto-added required tool: {must_have}")

        print(f"  Tools selected: {validated}")
        return validated

    except Exception as ex:
        print(f"  Tool selection fallback ({ex})")
        return list(CHAT_DEFAULT_TOOLS) if is_chat else []


# ── generate_safe_agent_code ──────────────────────────────────────────────────

def generate_safe_agent_code(
    plan: dict,
    selected_model: str,
    provider: str | None,
    supported_task: str,
    inferred_method: str,
    has_chat_template: bool,
    selected_tools: list[str],
) -> str:
    """
    Assemble a complete, runnable agent .py file as a string.
    """
    # ── Sanitise inputs ───────────────────────────────────────────────────────
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

    # ── Tool source + router ──────────────────────────────────────────────────
    tool_impl   = "\n".join(TOOL_CODE[t] for t in selected_tools if t in TOOL_CODE)
    tool_router = build_tool_router_code(selected_tools)
    has_tools   = bool(selected_tools)

    pip_pkgs    = [ALL_TOOLS[t]["pip"] for t in selected_tools if ALL_TOOLS[t].get("pip")]
    pip_comment = f"# pip install {' '.join(pip_pkgs)}\n" if pip_pkgs else ""

    # Build ACCEPTED_EXTENSIONS from tools that handle specific file types.
    # This variable is parsed by the orchestrator at discovery time to route
    # uploaded files to this agent automatically.
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

    if plan.get("route") == "tool" or inferred_method == "tool_pipeline":
        plan_copy["route"] = "tool"
        plan_copy["method"] = "tool_pipeline"
        plan_copy["pipeline"] = selected_tools
        method_str = json.dumps("tool_pipeline")
        system_str = json.dumps(system_prompt)
        plan_str = json.dumps(plan_copy, indent=4)
        tools_repr = json.dumps(selected_tools)
        source_fmt = json.dumps(str(plan.get("source_format") or "input"))
        target_fmt = json.dumps(str(plan.get("target_format") or "output"))

        lines = [
            "import os",
            "from dotenv import load_dotenv",
            pip_comment,
            "load_dotenv()",
            "",
            f"# Tool-only pipeline: {selected_tools}",
            f"# Tools  : {tools_repr}",
            "",
            f"SYSTEM_PROMPT = {system_str}",
            "",
            f"PLAN = {plan_str}",
            *([ accepted_exts_line, "" ] if accepted_exts_line else []),
            "",
            "# ── tool implementations ─────────────────────────────────────────────────",
            tool_impl,
            "",
            "# ── tool pipeline ────────────────────────────────────────────────────────",
            f"SOURCE_FORMAT = {source_fmt}",
            f"TARGET_FORMAT = {target_fmt}",
            "",
            "def _project_dir(subdir: str) -> str:",
            "    root = os.path.dirname(os.path.abspath(__file__))",
            "    for _ in range(5):",
            "        if os.path.exists(os.path.join(root, 'api.py')):",
            "            break",
            "        root = os.path.dirname(root)",
            "    out_dir = os.path.join(root, subdir)",
            "    os.makedirs(out_dir, exist_ok=True)",
            "    return out_dir",
            "",
            "def _clean_input(value: str) -> str:",
            "    value = (value or '').strip().strip('\"').strip(\"'\")",
            "    if '|' in value and os.path.exists(value.split('|', 1)[0].strip().strip('\"').strip(\"'\")):",
            "        value = value.split('|', 1)[0].strip().strip('\"').strip(\"'\")",
            "    return value",
            "",
            "def _safe_stem(value: str) -> str:",
            "    base = os.path.splitext(os.path.basename(value))[0] if value else 'output'",
            "    safe = ''.join(c for c in base[:40] if c.isalnum() or c in ' _-').strip().replace(' ', '_')",
            "    return safe or 'output'",
            "",
            "def run_agent(user_input: str):",
            "    data = _clean_input(user_input)",
            "    source_path = data if os.path.exists(data) else ''",
            "    for tool_name in PLAN.get('pipeline', []):",
            "        if tool_name == 'docx_reader':",
            "            if not source_path:",
            "                return 'Please provide a valid .docx file path.'",
            "            data = tool_docx_reader(source_path)",
            "        elif tool_name == 'pdf_reader':",
            "            if not source_path:",
            "                return 'Please provide a valid .pdf file path.'",
            "            data = tool_pdf_reader(source_path)",
            "        elif tool_name == 'file_reader':",
            "            if source_path:",
            "                data = tool_file_reader(source_path)",
            "        elif tool_name == 'pdf_generator':",
            "            from datetime import datetime as _dt",
            "            out_dir = _project_dir('generated_files')",
            "            stem = _safe_stem(source_path or data)",
            "            ts = _dt.now().strftime('%Y%m%d_%H%M%S')",
            "            out_path = os.path.join(out_dir, f'{ts}_{stem}.pdf')",
            "            data = tool_pdf_generator(str(data), out_path)",
            "        else:",
            "            return f'Unsupported tool in pipeline: {tool_name}'",
            "    return data",
            "",
            "",
            "def main():",
            "    print('Tool pipeline agent ready. Type exit or quit to stop.')",
            f"    print('Method  : {method_str}')",
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

    # ── Inference body ────────────────────────────────────────────────────────
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
        # Shared image-call snippet: try the configured provider first,
        # fall back to default HF routing if the provider rejects the model.
        _img_call = '''\
        try:
            image = client.text_to_image(prompt=user_input, model=model_name)
        except Exception as _pe:
            from huggingface_hub import InferenceClient as _FbIC
            image = _FbIC(api_key=HF_TOKEN).text_to_image(prompt=user_input, model=model_name)'''
        if "image_saver" in selected_tools:
            inference_body = _img_call + '''
        return tool_image_saver(image, prompt=user_input, model_name=model_name)'''
        else:
            inference_body = _img_call + '''
        import os as _os
        from datetime import datetime as _dt
        _here = _os.path.abspath(__file__)
        _proj_root = _os.path.dirname(_here)
        for _ in range(5):
            if _os.path.exists(_os.path.join(_proj_root, "api.py")):
                break
            _proj_root = _os.path.dirname(_proj_root)
        out_dir  = _os.path.join(_proj_root, "generated_images")
        _os.makedirs(out_dir, exist_ok=True)
        ts       = _dt.now().strftime("%Y%m%d_%H%M%S")
        safe     = "".join(c for c in user_input[:40] if c.isalnum() or c in " _-").strip().replace(" ", "_") or "image"
        out_path = _os.path.join(out_dir, f"{ts}_{safe}.png")
        image.save(out_path)
        return f"Image saved: {out_path}"'''

    elif inferred_method == "image_to_image":
        # image-to-image: user provides an image path + edit description
        # the generated agent asks for both inputs separately
        inference_body = '''\
        # user_input format: "image_path|||edit description"
        # split on ||| separator
        parts      = user_input.split("|||", 1)
        image_path = parts[0].strip().strip('"')
        edit_text  = parts[1].strip() if len(parts) > 1 else ""
        if not edit_text:
            return "Please provide an edit description after |||"
        import os as _os
        if not _os.path.exists(image_path):
            return f"Image file not found: {image_path}"
        with open(image_path, "rb") as _f:
            image_bytes = _f.read()
        result_img = client.image_to_image(
            image=image_bytes,
            prompt=edit_text,
            model=model_name,
        )
        from datetime import datetime as _dt
        _here = _os.path.abspath(__file__)
        _proj_root = _os.path.dirname(_here)
        for _ in range(5):
            if _os.path.exists(_os.path.join(_proj_root, "api.py")):
                break
            _proj_root = _os.path.dirname(_proj_root)
        out_dir  = _os.path.join(_proj_root, "generated_images")
        _os.makedirs(out_dir, exist_ok=True)
        ts       = _dt.now().strftime("%Y%m%d_%H%M%S")
        safe     = "".join(c for c in edit_text[:40] if c.isalnum() or c in " _-").strip().replace(" ", "_")
        out_path = _os.path.join(out_dir, f"{ts}_{safe}.png")
        result_img.save(out_path)
        return f"Edited image saved: {out_path}"'''

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

    elif inferred_method == "document_question_answering":
        inference_body = '''\
        # user_input format: "file_path | question"
        parts     = user_input.split("|", 1)
        file_path = parts[0].strip().strip(\'"\').strip("\'")
        question  = parts[1].strip() if len(parts) > 1 else "Please summarize this document."
        import os as _os
        if not file_path or not _os.path.exists(file_path):
            return f"File not found: {file_path!r}\\nFormat: file_path | question"
        # ── read document content ──────────────────────────────────────────
        doc_text = ""
        if file_path.lower().endswith(".pdf"):
            try:
                from pypdf import PdfReader as _PR
                _rdr = _PR(file_path)
                doc_text = "\\n".join(p.extract_text() or "" for p in _rdr.pages)[:8000]
            except ImportError:
                try:
                    from PyPDF2 import PdfReader as _PR
                    _rdr = _PR(file_path)
                    doc_text = "\\n".join(p.extract_text() or "" for p in _rdr.pages)[:8000]
                except ImportError:
                    doc_text = "[PDF reading requires: pip install pypdf]"
        else:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as _f:
                    doc_text = _f.read()[:8000]
            except Exception as _err:
                doc_text = f"[Could not read file: {_err}]"
        # ── send document + question to the LLM ───────────────────────────
        sys_msg = plan.get("final_system_prompt") or plan.get("system_prompt") or SYSTEM_PROMPT
        if doc_text:
            sys_msg += f"\\n\\n[Document Content]\\n{doc_text}"
        msgs = [{"role": "system", "content": sys_msg}] + HISTORY + [{"role": "user", "content": question}]
        response = client.chat_completion(model=model_name, messages=msgs,
                       max_tokens=plan.get("max_tokens", 800), temperature=plan.get("temperature", 0.7))
        reply = response.choices[0].message.content
        HISTORY.append({"role": "user", "content": user_input})
        HISTORY.append({"role": "assistant", "content": reply})
        if len(HISTORY) > 20: HISTORY[:] = HISTORY[-20:]
        return reply'''

    elif inferred_method == "text_to_video":
        inference_body = '''\
        video_data = client.text_to_video(user_input, model=model_name)
        # video_data may be bytes or a file-like object depending on the provider
        video_bytes = video_data.read() if hasattr(video_data, "read") else bytes(video_data)
        import os as _os
        from datetime import datetime as _dt
        _here = _os.path.abspath(__file__)
        _proj_root = _os.path.dirname(_here)
        for _ in range(5):
            if _os.path.exists(_os.path.join(_proj_root, "api.py")):
                break
            _proj_root = _os.path.dirname(_proj_root)
        out_dir  = _os.path.join(_proj_root, "generated_files")
        _os.makedirs(out_dir, exist_ok=True)
        ts       = _dt.now().strftime("%Y%m%d_%H%M%S")
        safe     = "".join(c for c in user_input[:40] if c.isalnum() or c in "_-").strip() or "video"
        out_path = _os.path.join(out_dir, f"{ts}_{safe}.mp4")
        with open(out_path, "wb") as _f:
            _f.write(video_bytes)
        return f"Video saved: generated_files/{_os.path.basename(out_path)}"'''

    elif inferred_method in ("text_to_speech", "text_to_audio"):
        inference_body = '''\
        audio_data = client.text_to_speech(user_input, model=model_name)
        # audio_data may be bytes or a file-like object depending on the provider
        audio_bytes = audio_data.read() if hasattr(audio_data, "read") else bytes(audio_data)
        import os as _os
        from datetime import datetime as _dt
        _here = _os.path.abspath(__file__)
        _proj_root = _os.path.dirname(_here)
        for _ in range(5):
            if _os.path.exists(_os.path.join(_proj_root, "api.py")):
                break
            _proj_root = _os.path.dirname(_proj_root)
        out_dir  = _os.path.join(_proj_root, "generated_files")
        _os.makedirs(out_dir, exist_ok=True)
        ts       = _dt.now().strftime("%Y%m%d_%H%M%S")
        safe     = "".join(c for c in user_input[:40] if c.isalnum() or c in "_-").strip() or "audio"
        out_path = _os.path.join(out_dir, f"{ts}_{safe}.wav")
        with open(out_path, "wb") as _f:
            _f.write(audio_bytes)
        return f"Audio saved: generated_files/{_os.path.basename(out_path)}"'''

    else:
        # ── [DYNAMIC FALLBACK] universal handler ─────────────────────────────
        # Any method without a dedicated template above lands here (new/unknown
        # HF tasks, plus summarization / image_to_text / classification / etc.).
        # It calls the InferenceClient method BY NAME and dispatches on the
        # RESULT TYPE, so novel tasks work without a per-task template.
        # To revert: replace this whole block with
        #     inference_body = '''\\
        #     return f"Method {plan['method']} is not supported yet."'''
        inference_body = '''\
        method_fn = getattr(client, plan["method"], None)
        if method_fn is None or not callable(method_fn):
            return "Method " + str(plan["method"]) + " is not available on this InferenceClient version."
        # primary input passed positionally; some methods want it as `text=`
        try:
            result = method_fn(user_input, model=model_name)
        except TypeError:
            result = method_fn(text=user_input, model=model_name)
        import os as _os, json as _json
        from datetime import datetime as _dt
        def _proj_dir(sub):
            _p = _os.path.dirname(_os.path.abspath(__file__))
            for _ in range(5):
                if _os.path.exists(_os.path.join(_p, "api.py")):
                    break
                _p = _os.path.dirname(_p)
            _d = _os.path.join(_p, sub)
            _os.makedirs(_d, exist_ok=True)
            return _d
        # ── dispatch on output type (text / image / binary / structured) ────
        if result is None:
            return "No output returned by the model."
        if isinstance(result, str):
            return result
        if hasattr(result, "save"):              # PIL.Image-like → save .png
            _ts  = _dt.now().strftime("%Y%m%d_%H%M%S")
            _out = _os.path.join(_proj_dir("generated_images"), _ts + "_output.png")
            result.save(_out)
            return "Image saved: " + _out
        if isinstance(result, (bytes, bytearray)) or hasattr(result, "read"):
            _data = result.read() if hasattr(result, "read") else bytes(result)
            _m    = plan["method"]
            _ext  = ".mp4" if "video" in _m else (".wav" if ("speech" in _m or "audio" in _m) else ".bin")
            _ts   = _dt.now().strftime("%Y%m%d_%H%M%S")
            _out  = _os.path.join(_proj_dir("generated_files"), _ts + "_output" + _ext)
            with open(_out, "wb") as _f:
                _f.write(_data)
            return "File saved: " + _out
        if isinstance(result, (list, dict)):     # classification / detection / NER
            return _json.dumps(result, ensure_ascii=False, indent=2, default=str)
        return str(result)'''
        # ─────────────────────────────────────────────────────────────────────

    # ── Render JSON values as safe string literals ────────────────────────────
    method_str   = json.dumps(inferred_method)
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
        *([ accepted_exts_line, "" ] if accepted_exts_line else []),
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
        "# ── inference ────────────────────────────────────────────────────────────",
        "def run_inference(client, model_name, plan, user_input):",
        f"    if plan['method'] == {method_str}:",
        inference_body,
        "",
        "    # ── fallback handlers ─────────────────────────────────────────────────",
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


# ── create_agent_python_file ──────────────────────────────────────────────────

def create_agent_python_file(agent_name: str, code: str) -> str:
    """Write the generated agent source to disk. Returns the filename."""
    safe  = agent_name.lower().replace(" ", "_").replace("-", "_")
    fname = f"{safe}_agent.py"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"Agent file created: {fname}")
    return fname
