"""
services/search.py — HuggingFace model discovery.

Responsibilities:
  - generate_search_plan()  : ask the planner LLM which HF tasks/queries to use
  - get_model_full_info()   : fetch provider + method details for one model
  - search_best_llms()      : orchestrate the full search and return ranked results
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import HfApi, InferenceClient

from config import (
    HF_MODEL_URL,
    TASK_TO_METHOD,
    CHAT_TEMPLATE_INDICATORS,
    SCORE_WEIGHT_DOWNLOADS,
    SCORE_WEIGHT_LIKES,
    SCORE_BOOST_KEYWORD,
    MODEL_SEARCH_LIMIT,
)
from models import ModelSuggestion
from utils.llm import chat_completion_with_retry


# ── generate_search_plan ──────────────────────────────────────────────────────

def generate_search_plan(prompt: str) -> dict:
    """
    Ask the planner LLM to produce a JSON search plan:
      { tasks, queries, boost_words }
    Falls back to sensible defaults on any error.
    """
    client = InferenceClient(api_key=os.getenv("HF_TOKEN"))
    system = """
You are a JSON generator. Convert user request into HF model search plan.
Return ONLY valid JSON. No markdown.

Allowed HF tasks: text-generation, translation, text-to-image, image-to-image,
image-to-text, text-to-audio, text-to-speech, automatic-speech-recognition,
audio-classification, image-classification, object-detection, summarization,
question-answering, sentence-similarity, fill-mask, token-classification

JSON:
{
  "tasks": ["task-name"],
  "queries": ["q1", "q2", "q3"],
  "boost_words": ["w1", "w2"]
}

Rules:
- chat/conversation/coding → ["text-generation"]
- generate/create image → ["text-to-image"]
- edit/modify/change existing image → ["image-to-image"]
- queries MUST include 2+ of: instruct, chat, llm, qwen, mistral, llama
"""
    try:
        r = chat_completion_with_retry(
            client,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
        )
        text = r.choices[0].message.content.strip()
        print("Search plan:"); print(text)
        s = text.find("{"); e = text.rfind("}") + 1
        if s == -1 or e == 0:
            raise ValueError("No JSON found in planner response")
        return json.loads(text[s:e])
    except Exception as ex:
        print(f"Search plan fallback ({ex})")
        prompt_lower = prompt.lower()
        edit_words = ["edit", "modify", "change", "alter", "adjust", "fix", "update", "transform"]
        if any(w in prompt_lower for w in edit_words) and any(w in prompt_lower for w in ["image", "picture", "photo"]):
            return {
                "tasks": ["image-to-image"],
                "queries": ["flux image editing", "stable diffusion img2img", "image to image"],
                "boost_words": ["edit", "img2img", "flux"],
            }
        if any(w in prompt_lower for w in ["image", "picture", "photo"]):
            return {
                "tasks": ["text-to-image"],
                "queries": ["stable diffusion", "sdxl", "flux"],
                "boost_words": ["diffusion", "sdxl"],
            }
        return {
            "tasks": ["text-generation"],
            "queries": ["instruct", "chat", "llm", "qwen", "mistral"],
            "boost_words": ["instruct", "chat"],
        }


# ── get_model_full_info ───────────────────────────────────────────────────────

def get_model_full_info(api: HfApi, model_id: str, search_task: str) -> dict:
    """
    Fetch provider mapping and chat-template status for a single model.
    Returns a dict with keys: provider, supported_task, inferred_method,
    has_chat_template, pipeline_tag, tags.
    """
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
            ind in tag_text or ind in model_id.lower()
            for ind in CHAT_TEMPLATE_INDICATORS
        )

        # Try multiple attribute names — huggingface_hub changed this in newer versions
        mapping = (
            getattr(info, "inference_provider_mapping", None)
            or getattr(info, "inference_providers", None)
            or getattr(info, "inferenceProviderMapping", None)
        )
        if not mapping:
            return result

        items = mapping.values() if isinstance(mapping, dict) else list(mapping)

        # Log the first entry so we can see the real status value in Render logs
        if items:
            first = next(iter(items))
            raw_status = first.get("status") if isinstance(first, dict) else getattr(first, "status", None)
            print(f"  [DEBUG] {model_id} provider status value: {raw_status!r}")

        # "live" is the historic value; accept any non-error/disabled status
        # so the code keeps working if HuggingFace renames it
        _DEAD_STATUSES = {"error", "disabled", "unsupported", "offline", "unavailable", None}

        for pd in items:
            status = pd.get("status") if isinstance(pd, dict) else getattr(pd, "status", None)
            prov   = pd.get("provider") if isinstance(pd, dict) else getattr(pd, "provider", None)
            ptask  = pd.get("task") if isinstance(pd, dict) else getattr(pd, "task", None)

            if status not in _DEAD_STATUSES and prov:
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


# ── search_best_llms ──────────────────────────────────────────────────────────

def search_best_llms(
    agent_prompt: str,
    limit: int = MODEL_SEARCH_LIMIT,
    progress_cb=None,
) -> tuple[list[ModelSuggestion], str]:
    """
    Search HuggingFace for models that fit the agent prompt.
    Returns (ranked_suggestions, primary_task).

    progress_cb(status, message) is called at each stage so callers
    can stream live progress to clients (e.g. via SSE).
    """
    def emit(status: str, message: str) -> None:
        if progress_cb:
            progress_cb(status, message)

    api = HfApi(token=os.getenv("HF_TOKEN"))

    emit("planning", "Analyzing your request and planning the search...")
    plan = generate_search_plan(agent_prompt)
    print("\nSearch plan result:"); print(plan)

    queries     = plan.get("queries",     ["chat", "instruct"])
    tasks       = plan.get("tasks",       ["text-generation"])
    boost_words = plan.get("boost_words", ["instruct", "chat"])
    if isinstance(tasks, str):
        tasks = [tasks]

    # Override for obvious image prompts
    if any(w in agent_prompt.lower() for w in ["image", "picture", "photo"]):
        tasks       = ["text-to-image"]
        queries     = ["stable diffusion", "sdxl", "flux"]
        boost_words = ["diffusion", "sdxl"]

    # ── Pass 1: collect raw candidates from list_models (quick) ─────────────────
    seen_ids: set[str] = set()
    raw: list[tuple]   = []   # (model_id, task, score, is_free)

    total_searches = len(queries) * len(tasks)
    search_num     = 0
    for q in queries:
        for task in tasks:
            search_num += 1
            emit("searching", f"Searching for '{q}' ({task}) [{search_num}/{total_searches}]...")
            try:
                models = api.list_models(
                    search=q, task=task,
                    sort="downloads", direction=-1,
                    limit=limit, full=True,
                )
            except TypeError:
                models = api.list_models(search=q, filter=task, limit=limit, full=True)

            for model in models:
                mid = model.modelId
                if mid in seen_ids:
                    continue
                seen_ids.add(mid)

                tags  = model.tags or []
                score = (
                    (model.downloads or 0) * SCORE_WEIGHT_DOWNLOADS
                    + (model.likes or 0) * SCORE_WEIGHT_LIKES
                )
                for w in boost_words:
                    if w.lower() in mid.lower() or w.lower() in " ".join(tags).lower():
                        score += SCORE_BOOST_KEYWORD

                is_free = not getattr(model, "gated", False)
                raw.append((mid, task, score, is_free))

    # ── Pass 2: fetch full provider/method info per model (the slow part) ────────
    total = len(raw)
    emit("analyzing", f"Found {total} candidates. Checking provider availability...")

    suggestions: list[ModelSuggestion] = []
    for i, (mid, task, score, is_free) in enumerate(raw):
        if i % 5 == 0:
            emit("progress", f"Checking model {i + 1}/{total}...")
        mi = get_model_full_info(api, mid, task)
        suggestions.append(ModelSuggestion(
            name=mid,
            url=HF_MODEL_URL + mid,
            score=score,
            is_free=is_free,
            has_provider=mi["provider"] is not None,
            provider=mi["provider"],
            supported_task=mi["supported_task"],
            inferred_method=mi["inferred_method"],
            has_chat_template=mi["has_chat_template"],
            pipeline_tag=mi["pipeline_tag"],
            tags=mi["tags"],
        ))

    emit("ranking", f"Ranking {len(suggestions)} models by score and availability...")
    suggestions.sort(key=lambda x: (not x.is_free, not x.has_provider, -x.score))
    return suggestions, tasks[0]