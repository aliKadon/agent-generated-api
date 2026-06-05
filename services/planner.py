"""
services/planner.py — LLM-powered prompt and plan generation.

Responsibilities:
  - generate_agent_prompt_plan() : draft a system prompt + clarifying questions
  - generate_agent_code_plan()   : decide method, template, and settings
  - finalize_agent_plan()        : merge user answers into a final plan
"""

import json
import os
import sys

from huggingface_hub import InferenceClient

from config import DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS
from utils.llm import chat_completion_with_retry

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



# ── generate_agent_prompt_plan ────────────────────────────────────────────────

def generate_agent_prompt_plan(user_agent_request: str, selected_model: str) -> dict:
    """
    Ask the planner to produce:
      - agent_type
      - clarifying_questions
      - draft_system_prompt
      - missing_info
      - recommended_settings
    """
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
        r = chat_completion_with_retry(
            client,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Agent: {user_agent_request}\nModel: {selected_model}\nCreate plan."},
            ],
            max_tokens=500,
        )
        text = r.choices[0].message.content.strip()
        s = text.find("{"); e = text.rfind("}") + 1
        if s == -1 or e == 0:
            raise ValueError("No JSON found")
        return json.loads(text[s:e])
    except Exception as ex:
        print(f"Prompt plan fallback ({ex})")
        return {
            "agent_type": "general assistant",
            "clarifying_questions": [
                "What is the main task?",
                "Who uses this?",
                "What tone?",
                "Any rules?",
            ],
            "draft_system_prompt": (
                f"You are a professional AI assistant. "
                f"Help the user with: {user_agent_request}."
            ),
            "missing_info": ["tone", "rules"],
            "recommended_settings": {
                "temperature": DEFAULT_TEMPERATURE,
                "max_tokens": DEFAULT_MAX_TOKENS,
            },
        }


# ── generate_agent_code_plan ──────────────────────────────────────────────────

def generate_agent_code_plan(
    user_agent_request: str,
    selected_model: str,
    supported_task: str,
    inferred_method: str,
    has_chat_template: bool,
    system_prompt: str,
) -> dict:
    """
    Ask the planner to decide input/output kind, prompt template, and
    generation settings. The method is locked to inferred_method.
    """
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
        r = chat_completion_with_retry(
            client,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": (
                    f"Request: {user_agent_request}\n"
                    f"Model: {selected_model}\n"
                    f"Task: {supported_task}\n"
                    f"System: {system_prompt}"
                )},
            ],
            max_tokens=500,
        )
        text = r.choices[0].message.content.strip()
        s = text.find("{"); e = text.rfind("}") + 1
        result = json.loads(text[s:e])
        result["method"] = inferred_method   # always lock the method
        return result
    except Exception as ex:
        print(f"Code plan fallback ({ex})")
        if inferred_method == "chat_completion":
            pt = "{user_input}"
        else:
            pt = f"{system_prompt}\n\nUser: {{user_input}}\nAssistant:"
        return {
            "method": inferred_method,
            "input_kind": "text",
            "input_label": "Enter your message",
            "system_prompt": system_prompt,
            "prompt_template": pt,
            "output_kind": "text",
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
        }


# ── finalize_agent_plan ───────────────────────────────────────────────────────

def finalize_agent_plan(
    user_agent_request: str,
    selected_model: str,
    agent_plan: dict,
    answers: dict,
    inferred_method: str,
    selected_tools: list[str],
) -> dict:
    """
    Merge clarifying-question answers into a final plan.
    Method and tools are locked — the planner cannot change them.
    """
    from tools.registery import ALL_TOOLS   # local import avoids circular deps

    client     = InferenceClient(api_key=os.getenv("HF_TOKEN"))
    tools_json = json.dumps([
        {"name": t, "reason": ALL_TOOLS[t]["description"]}
        for t in selected_tools
    ])
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
        r = chat_completion_with_retry(
            client,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": (
                    f"Request: {user_agent_request}\n"
                    f"Answers: {json.dumps(answers)}\n"
                    f"Plan: {json.dumps(agent_plan)}"
                )},
            ],
            max_tokens=700,
        )
        text = r.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        s = text.find("{"); e = text.rfind("}") + 1
        if s == -1 or e == 0:
            raise ValueError("No JSON found")
        result = json.loads(text[s:e])
        # Always enforce locked values
        result["method"] = inferred_method
        result["tools"]  = [
            {"name": t, "reason": ALL_TOOLS[t]["description"]}
            for t in selected_tools
        ]
        return result
    except Exception as ex:
        print(f"Finalize fallback ({ex})")
        return {
            "final_system_prompt": agent_plan.get(
                "system_prompt", "You are a helpful assistant."
            ),
            "method": inferred_method,
            "tools": [
                {"name": t, "reason": ALL_TOOLS[t]["description"]}
                for t in selected_tools
            ],
            "routing_rules": [],
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
        }