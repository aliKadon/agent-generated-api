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
        if "image_saver" in selected_tools:
            inference_body = '''\
        image = client.text_to_image(prompt=user_input, model=model_name)
        return tool_image_saver(image, prompt=user_input, model_name=model_name)'''
        else:
            inference_body = '''\
        image = client.text_to_image(prompt=user_input, model=model_name)
        import os as _os
        out_dir = "generated_images"
        _os.makedirs(out_dir, exist_ok=True)
        safe = "".join(c for c in user_input[:40] if c.isalnum() or c in " _-").strip().replace(" ", "_")
        out_path = f"{out_dir}/{safe}.png"
        image.save(out_path)
        return f"Image saved to {out_path}"'''

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
        out_dir  = "generated_images"
        _os.makedirs(out_dir, exist_ok=True)
        ts       = _dt.now().strftime("%Y%m%d_%H%M%S")
        safe     = "".join(c for c in edit_text[:40] if c.isalnum() or c in " _-").strip().replace(" ", "_")
        out_path = f"{out_dir}/{ts}_{safe}.png"
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

    else:
        inference_body = '''\
        return f"Method {plan['method']} is handled below."'''

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
        "",
        f"# Conversation history — keeps context across turns (max {HISTORY_MAX_MESSAGES} messages)",
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
        # image_to_image needs two separate inputs
        *(
            [
                "        image_path  = input('\\nImage path (drag & drop): ').strip().strip('\"')",
                "        edit_prompt = input('Edit description         : ').strip()",
                "        if image_path.lower() in ['exit','quit'] or edit_prompt.lower() in ['exit','quit']:",
                "            break",
                "        user_input = image_path + '|||' + edit_prompt",
            ]
            if inferred_method == "image_to_image"
            else [
                f"        user_input = input('\\n{input_label}: ').strip()",
            ]
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