"""
main.py — CLI entrypoint for the agent builder.

Orchestrates the interactive flow:
  1. Search for candidate models
  2. Let the user pick one
  3. Generate a prompt plan + ask clarifying questions
  4. Select and confirm tools
  5. Build and write the agent file
"""

import json
import os
import sys

# Ensure the project root (main_code/) is always on the path,
# regardless of which directory the user runs the script from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

from services.search import search_best_llms
from services.planner import (
    generate_agent_prompt_plan,
    generate_agent_code_plan,
    finalize_agent_plan,
)
from services.generator import (
    decide_tools,
    create_agent_python_file,
)
# generate_agent_code_with_mode: AI-written code (Plan A) with automatic
# fallback to the original template generator (Plan B). Returns (code, mode)
# where mode is "ai" or "template". Toggle via config.USE_AI_CODEGEN.
from services.ai_codegen import generate_agent_code_with_mode
from tools.registery import ALL_TOOLS

load_dotenv()


def main() -> None:
    # ── 1. Get the user's idea ────────────────────────────────────────────────
    print("Describe the AI agent you want to build:")
    prompt = input("> ").strip()
    if not prompt:
        print("Please enter a valid description.")
        return

    # ── 2. Search for models ──────────────────────────────────────────────────
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
        print("No free model available.")
        return

    # ── 3. User picks a model ─────────────────────────────────────────────────
    selected_model = None
    while True:
        print("\nChoose a model by number:")
        c = input("> ").strip()
        try:
            selected_model = free_models[int(c) - 1]
        except Exception:
            print("Invalid choice.")
            continue
        if not selected_model.has_provider:
            print("❌ No provider available.  1 = Try again   2 = Exit")
            if input("> ").strip() == "1":
                continue
            return
        break

    print(f"\n✅ {selected_model.name}")
    print(f"   provider={selected_model.provider} | task={selected_model.supported_task} | method={selected_model.inferred_method}")

    if not os.getenv("HF_TOKEN"):
        print("⚠️  HF_TOKEN not found in .env")
        return

    # ── 4. Generate prompt plan + ask clarifying questions ────────────────────
    print("\nCreating agent prompt plan…")
    agent_plan = generate_agent_prompt_plan(prompt, selected_model.name)
    print("\nAgent plan:"); print(json.dumps(agent_plan, indent=2))

    answers: dict[str, str] = {}
    for q in agent_plan.get("clarifying_questions", []):
        print(f"\n{q}")
        answers[q] = input("> ").strip()

    system_prompt = agent_plan["draft_system_prompt"]

    # ── 5. Select tools ───────────────────────────────────────────────────────
    print("\nSelecting tools…")
    selected_tools = decide_tools(prompt, agent_plan.get("agent_type", ""), selected_model.inferred_method)

    if selected_tools:
        print("\nTools to be added:")
        for t in selected_tools:
            pip = ALL_TOOLS[t].get("pip")
            free_note = "free, no API key" if not pip or t == "memory" else f"pip install {pip}"
            print(f"  ✅ {t} — {ALL_TOOLS[t]['description']}")
            print(f"     ({free_note})")

        print("\nKeep? (y / n / comma-separated subset e.g. web_search,calculator)")
        choice = input("> ").strip().lower()
        if choice == "n":
            selected_tools = []
        elif choice not in ("y", ""):
            selected_tools = [t.strip() for t in choice.split(",") if t.strip() in ALL_TOOLS]
            print(f"Using: {selected_tools}")

    # ── 6. Build the final plan ───────────────────────────────────────────────
    code_plan = generate_agent_code_plan(
        prompt,
        selected_model.name,
        selected_model.supported_task,
        selected_model.inferred_method,
        selected_model.has_chat_template,
        system_prompt,
    )

    final_plan = finalize_agent_plan(
        prompt,
        selected_model.name,
        code_plan,
        answers,
        selected_model.inferred_method,
        selected_tools,
    )
    print("\nFinal plan:"); print(json.dumps(final_plan, indent=2))

    # ── 7. Generate and write the agent file ──────────────────────────────────
    code, codegen_mode = generate_agent_code_with_mode(
        plan=final_plan,
        selected_model=selected_model.name,
        provider=selected_model.provider,
        supported_task=selected_model.supported_task,
        inferred_method=selected_model.inferred_method,
        has_chat_template=selected_model.has_chat_template,
        selected_tools=selected_tools,
    )

    fname = create_agent_python_file(agent_plan["agent_type"], code)
    print(f"\n✅ Agent file: {fname}  (codegen: {codegen_mode})")

    pip_pkgs = list({ALL_TOOLS[t]["pip"] for t in selected_tools if ALL_TOOLS[t].get("pip")})
    if pip_pkgs:
        print(f"📦 Install dependencies: pip install {' '.join(pip_pkgs)}")

    # Auto-save to database so the API reflects the new agent immediately
    try:
        from api import sync_agents_to_db
        total = sync_agents_to_db()
        print(f"💾 Saved to database ({total} agent(s) total)")
    except Exception as e:
        print(f"⚠️  Could not save to database: {e}")


if __name__ == "__main__":
    main()