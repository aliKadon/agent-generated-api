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

import requests as _requests

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
    SUPPLEMENTAL_SEARCH_TASKS,
)
from models import ModelSuggestion
from utils.llm import chat_completion_with_retry


# ── generate_search_plan ──────────────────────────────────────────────────────

def generate_search_plan(prompt: str) -> dict:
    """
    Ask the planner LLM to produce a JSON search plan:
      { tasks, queries, boost_words }
    Falls back to keyword-based detection on any LLM error.
    """
    client = InferenceClient(api_key=os.getenv("HF_TOKEN"))
    system = """
You are a JSON generator. Given an AI agent description, produce a HuggingFace model search plan.
Return ONLY valid JSON. No markdown, no explanation.

Allowed HF tasks:
  text-generation, translation, summarization, question-answering,
  sentence-similarity, feature-extraction, fill-mask, token-classification,
  zero-shot-classification,
  text-to-image, image-to-image,
  image-to-text, visual-question-answering, document-question-answering,
  image-classification, zero-shot-image-classification, object-detection, depth-estimation,
  automatic-speech-recognition, text-to-speech, text-to-audio, audio-classification,
  text-to-video

JSON schema:
{
  "tasks": ["one-hf-task"],
  "queries": ["search term 1", "search term 2", "search term 3"],
  "boost_words": ["word1", "word2"]
}

INTENT → TASK mapping (use the agent's PURPOSE, not just keywords):
  chat / code / write / assistant / reasoning / general help       → text-generation
  translate text between languages                                 → translation
  summarize / condense / extract key points from TEXT              → summarization
  answer questions given a TEXT context / reading comprehension    → question-answering
  semantic search / compare sentences / embeddings / retrieval     → sentence-similarity
  compute vector embeddings / feature vectors                      → feature-extraction
  fill [MASK] token / masked language modelling                    → fill-mask
  extract named entities / NER / POS tagging / tag tokens          → token-classification
  classify text into categories without training examples          → zero-shot-classification
  GENERATE / CREATE / DRAW / PAINT a NEW image from text           → text-to-image
  EDIT / MODIFY / TRANSFORM / STYLIZE an EXISTING image            → image-to-image
  READ / DESCRIBE / ANALYZE / CAPTION an existing image            → image-to-text
  ANSWER QUESTIONS about the content of a specific image           → visual-question-answering
  READ / ANALYZE / EXTRACT information from a DOCUMENT or PDF      → document-question-answering
  CLASSIFY / LABEL what category an image belongs to               → image-classification
  classify image without predefined labels / zero-shot             → zero-shot-image-classification
  DETECT / LOCATE / COUNT objects / draw bounding boxes in image   → object-detection
  estimate DEPTH / 3-D structure from an image                     → depth-estimation
  TRANSCRIBE / convert SPEECH or AUDIO to TEXT                     → automatic-speech-recognition
  convert TEXT to SPOKEN audio / voice synthesis / TTS             → text-to-speech
  GENERATE MUSIC / SOUND EFFECTS / AUDIO from text                 → text-to-audio
  CLASSIFY audio / identify sounds / music genre                   → audio-classification
  GENERATE / CREATE a VIDEO from text                              → text-to-video

Critical distinctions — these are the most common mistakes:
  "read image" / "describe image" / "analyze image"  →  image-to-text  (NOT text-to-image)
  "generate image" / "draw" / "create image"         →  text-to-image  (NOT image-to-text)
  "edit image" / "modify photo"                      →  image-to-image (NOT text-to-image)
  "read pdf" / "analyze document"                    →  document-question-answering
  "transcribe audio"                                 →  automatic-speech-recognition (NOT text-to-speech)
  "speak text" / "text to voice"                     →  text-to-speech (NOT automatic-speech-recognition)

queries: use specific model/architecture names as they appear in HuggingFace model IDs.
  Examples: "flux" not "flux image editing", "whisper" not "speech recognition model",
  "stable-diffusion" not "image generation", "llama" not "chat assistant".
  Short architecture names rank much better in HF search than descriptive phrases.
"""
    try:
        r = chat_completion_with_retry(
            client,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=250,
        )
        text = r.choices[0].message.content.strip()
        print("Search plan:"); print(text)
        s = text.find("{"); e = text.rfind("}") + 1
        if s == -1 or e == 0:
            raise ValueError("No JSON found in planner response")
        return json.loads(text[s:e])
    except Exception as ex:
        print(f"Search plan fallback ({ex})")
        return _keyword_fallback_plan(prompt)


def _keyword_fallback_plan(prompt: str) -> dict:
    """
    Pure keyword-based fallback when the LLM call fails.
    Checked in order from most-specific to most-general.
    """
    p = prompt.lower()

    _img  = lambda: any(w in p for w in ["image", "picture", "photo", "img"])
    _vid  = lambda: any(w in p for w in ["video", "clip", "animate", "animation"])
    _aud  = lambda: any(w in p for w in ["audio", "sound", "music", "speech", "voice"])
    _doc  = lambda: any(w in p for w in ["document", "pdf", "file", "doc", "page"])
    _read = lambda: any(w in p for w in ["read", "describe", "analyze", "caption",
                                          "understand", "explain", "extract", "ocr",
                                          "recognize text", "tell me about"])
    _edit = lambda: any(w in p for w in ["edit", "modify", "change", "alter", "adjust",
                                          "enhance", "fix", "transform", "stylize",
                                          "restore", "colorize", "upscale"])
    _gen  = lambda: any(w in p for w in ["generate", "create", "make", "draw", "paint",
                                          "render", "produce", "design"])

    # ── image understanding (must come BEFORE image generation) ──────────────
    if _read() and _img():
        return {"tasks": ["image-to-text"],
                "queries": ["image to text", "blip", "llava", "image captioning"],
                "boost_words": ["blip", "llava", "captioning"]}

    if any(w in p for w in ["vqa", "visual question", "question about image",
                              "ask about image", "answer from image"]):
        return {"tasks": ["visual-question-answering"],
                "queries": ["visual question answering", "vqa", "blip-2", "idefics"],
                "boost_words": ["vqa", "blip"]}

    if _doc() or any(w in p for w in ["extract from file", "read file", "file reader"]):
        return {"tasks": ["document-question-answering"],
                "queries": ["document question answering", "layoutlm", "donut", "document understanding"],
                "boost_words": ["layoutlm", "donut"]}

    if any(w in p for w in ["object detection", "detect object", "find objects",
                              "bounding box", "yolo", "count objects", "locate objects"]):
        return {"tasks": ["object-detection"],
                "queries": ["yolo", "object detection", "detr", "rtdetr"],
                "boost_words": ["detection", "yolo"]}

    if any(w in p for w in ["classify image", "image classification", "label image",
                              "image category", "identify image"]):
        return {"tasks": ["image-classification"],
                "queries": ["image classification", "vit", "resnet", "efficientnet"],
                "boost_words": ["classification", "vit"]}

    if any(w in p for w in ["depth estimation", "depth map", "3d", "monocular depth"]):
        return {"tasks": ["depth-estimation"],
                "queries": ["depth estimation", "midas", "depth anything", "dpt"],
                "boost_words": ["depth", "midas"]}

    # ── image generation / editing ───────────────────────────────────────────
    if _edit() and _img():
        return {"tasks": ["image-to-image"],
                "queries": ["flux", "img2img", "instruct-pix2pix", "controlnet"],
                "boost_words": ["flux", "img2img", "edit"]}

    if (_gen() and _img()) or any(w in p for w in ["text to image", "text-to-image",
                                                     "artwork", "illustration", "صورة", "ارسم"]):
        return {"tasks": ["text-to-image"],
                "queries": ["flux", "stable-diffusion", "sdxl", "playground"],
                "boost_words": ["flux", "sdxl", "diffusion"]}

    # ── video ────────────────────────────────────────────────────────────────
    if _vid():
        return {"tasks": ["text-to-video"],
                "queries": ["text to video", "wan", "cogvideo", "video generation"],
                "boost_words": ["video", "animate"]}

    # ── audio / speech ───────────────────────────────────────────────────────
    if any(w in p for w in ["transcribe", "speech to text", "speech-to-text",
                              "audio to text", "recognize speech", "stt", "whisper"]):
        return {"tasks": ["automatic-speech-recognition"],
                "queries": ["whisper", "speech recognition", "asr", "wav2vec"],
                "boost_words": ["whisper", "wav2vec"]}

    if any(w in p for w in ["text to speech", "text-to-speech", "tts", "read aloud",
                              "voice synthesis", "speak text", "narrate", "voice generation"]):
        return {"tasks": ["text-to-speech"],
                "queries": ["text to speech", "tts", "coqui", "xtts", "speecht5"],
                "boost_words": ["tts", "coqui"]}

    if any(w in p for w in ["generate music", "music generation", "compose music",
                              "generate audio", "sound effect", "musicgen"]):
        return {"tasks": ["text-to-audio"],
                "queries": ["musicgen", "audio generation", "music", "audiocraft"],
                "boost_words": ["musicgen", "audiocraft"]}

    if any(w in p for w in ["audio classification", "classify audio", "sound classification",
                              "identify sound", "music genre", "emotion voice"]):
        return {"tasks": ["audio-classification"],
                "queries": ["audio classification", "sound classification", "ast"],
                "boost_words": ["classification", "audio"]}

    # ── text tasks ───────────────────────────────────────────────────────────
    if any(w in p for w in ["translate", "translation", "multilingual", "localize"]):
        return {"tasks": ["translation"],
                "queries": ["translation", "multilingual", "nllb", "opus-mt", "marian"],
                "boost_words": ["translation", "multilingual"]}

    if any(w in p for w in ["summarize", "summarization", "condense", "tldr",
                              "key points", "brief", "abstract"]):
        return {"tasks": ["summarization"],
                "queries": ["summarization", "bart", "pegasus", "led"],
                "boost_words": ["summarization", "bart"]}

    if any(w in p for w in ["similarity", "semantic search", "embedding", "embeddings",
                              "vector", "retrieval", "compare text", "dense retrieval"]):
        return {"tasks": ["sentence-similarity"],
                "queries": ["sentence similarity", "embedding", "sbert", "e5", "bge"],
                "boost_words": ["embedding", "sbert"]}

    if any(w in p for w in ["named entity", "ner", "entity recognition",
                              "pos tagging", "token classification", "extract entities"]):
        return {"tasks": ["token-classification"],
                "queries": ["ner", "named entity recognition", "bert ner", "token classification"],
                "boost_words": ["ner", "bert"]}

    if any(w in p for w in ["question answering", "reading comprehension",
                              "answer from context", "extract answer"]):
        return {"tasks": ["question-answering"],
                "queries": ["question answering", "bert qa", "roberta squad", "extractive qa"],
                "boost_words": ["qa", "squad"]}

    if any(w in p for w in ["fill mask", "fill blank", "masked language", "predict token"]):
        return {"tasks": ["fill-mask"],
                "queries": ["bert", "roberta", "fill mask", "masked language model"],
                "boost_words": ["bert", "roberta"]}

    # ── default: chat / text generation ──────────────────────────────────────
    return {"tasks": ["text-generation"],
            "queries": ["instruct", "chat", "llm", "qwen", "mistral", "llama"],
            "boost_words": ["instruct", "chat"]}


# ── get_model_full_info ───────────────────────────────────────────────────────

def get_model_full_info(api: HfApi, model_id: str, search_task: str,
                        strict_task: bool = False) -> dict:
    """
    Fetch provider mapping and metadata for a single model via direct HF HTTP API.

    Uses the raw REST endpoint directly so gated models (which raise
    GatedRepoError in huggingface_hub) are handled transparently — provider
    info is public metadata served regardless of gating or license agreement.

    strict_task=True: only accept a provider whose reported task matches
    search_task exactly (used for supplemental-task searches).
    """
    result = {
        "provider": None,
        "supported_task": search_task,
        "inferred_method": TASK_TO_METHOD.get(search_task, "text_generation"),
        "has_chat_template": False,
        "pipeline_tag": None,
        "tags": [],
        "license": None,
    }

    _DEAD_STATUSES = {"error", "disabled", "unsupported", "offline", "unavailable", None}

    # ── Primary: direct HF REST API (works for gated and non-gated models) ───
    try:
        url = (f"https://huggingface.co/api/models/{model_id}"
               f"?expand[]=inferenceProviderMapping&expand[]=pipeline_tag&expand[]=tags")
        headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN', '')}"}
        resp = _requests.get(url, headers=headers, timeout=10)
        print(f"  [DEBUG] {model_id} HTTP status={resp.status_code}")

        if resp.status_code == 200:
            data = resp.json()
            result["pipeline_tag"] = data.get("pipeline_tag")
            result["tags"] = data.get("tags") or []

            tag_text = " ".join(result["tags"]).lower()
            result["has_chat_template"] = any(
                ind in tag_text or ind in model_id.lower()
                for ind in CHAT_TEMPLATE_INDICATORS
            )
            license_tags = [t for t in result["tags"] if t.startswith("license:")]
            if license_tags:
                result["license"] = license_tags[0].replace("license:", "")

            mapping = data.get("inferenceProviderMapping") or {}
            print(f"  [DEBUG] {model_id} providers={list(mapping.keys())}")

            best_prov  = None
            best_ptask = None
            any_prov   = None
            any_ptask  = None

            for prov_name, pd in mapping.items():
                status = pd.get("status") if isinstance(pd, dict) else None
                ptask  = pd.get("task")   if isinstance(pd, dict) else None

                if status in _DEAD_STATUSES:
                    continue

                if any_prov is None:
                    any_prov  = prov_name
                    any_ptask = ptask

                if ptask == search_task:
                    best_prov  = prov_name
                    best_ptask = ptask
                    break

            chosen_prov  = best_prov  or (None if strict_task else any_prov)
            chosen_ptask = best_ptask or (None if strict_task else any_ptask)

            if chosen_prov:
                actual = chosen_ptask or search_task
                result["provider"] = chosen_prov
                result["supported_task"] = actual
                search_method = TASK_TO_METHOD.get(search_task, "text_generation")
                if actual == "conversational":
                    result["inferred_method"] = "chat_completion"
                elif actual == "text-generation" and result["has_chat_template"]:
                    result["inferred_method"] = "chat_completion"
                elif search_method != "text_generation":
                    result["inferred_method"] = search_method
                    result["supported_task"]  = search_task
                else:
                    result["inferred_method"] = TASK_TO_METHOD.get(actual, "text_generation")

            return result
    except Exception as e:
        print(f"  ℹ️  HTTP fetch failed for {model_id}: {e} — trying huggingface_hub fallback")

    # ── Fallback: huggingface_hub client (for non-gated models if HTTP failed) ─
    try:
        info = api.model_info(model_id, expand="inferenceProviderMapping")
        result["pipeline_tag"] = getattr(info, "pipeline_tag", None)
        result["tags"] = list(getattr(info, "tags", []) or [])

        tag_text = " ".join(result["tags"]).lower()
        result["has_chat_template"] = any(
            ind in tag_text or ind in model_id.lower()
            for ind in CHAT_TEMPLATE_INDICATORS
        )
        license_tags = [t for t in result["tags"] if t.startswith("license:")]
        if license_tags:
            result["license"] = license_tags[0].replace("license:", "")

        mapping = (
            getattr(info, "inference_provider_mapping", None)
            or getattr(info, "inference_providers", None)
            or getattr(info, "inferenceProviderMapping", None)
        ) or {}

        if isinstance(mapping, dict):
            items = list(mapping.items())
        else:
            items = [
                ((p.get("provider") if isinstance(p, dict) else getattr(p, "provider", None)), p)
                for p in mapping
            ]

        best_prov = best_ptask = any_prov = any_ptask = None
        for prov, pd in items:
            status = pd.get("status") if isinstance(pd, dict) else getattr(pd, "status", None)
            ptask  = pd.get("task")   if isinstance(pd, dict) else getattr(pd, "task",   None)
            if status in _DEAD_STATUSES or not prov:
                continue
            if any_prov is None:
                any_prov, any_ptask = prov, ptask
            if ptask == search_task:
                best_prov, best_ptask = prov, ptask
                break

        chosen_prov  = best_prov  or (None if strict_task else any_prov)
        chosen_ptask = best_ptask or (None if strict_task else any_ptask)

        if chosen_prov:
            actual = chosen_ptask or search_task
            result["provider"] = chosen_prov
            result["supported_task"] = actual
            search_method = TASK_TO_METHOD.get(search_task, "text_generation")
            if actual == "conversational":
                result["inferred_method"] = "chat_completion"
            elif actual == "text-generation" and result["has_chat_template"]:
                result["inferred_method"] = "chat_completion"
            elif search_method != "text_generation":
                result["inferred_method"] = search_method
                result["supported_task"]  = search_task
            else:
                result["inferred_method"] = TASK_TO_METHOD.get(actual, "text_generation")
    except Exception as e:
        print(f"  ℹ️  huggingface_hub fallback also failed for {model_id}: {e}")

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

    # ── Keyword-based task overrides ─────────────────────────────────────────────
    # Check these AFTER the LLM plan so we can correct it when needed.
    # Order matters: more specific patterns first.
    _p = agent_prompt.lower()

    _OVERRIDES = [
        # ── image understanding (most specific — check before generation) ────────
        # document / pdf reading — check before generic image-to-text
        {
            "match_any":    ["document question", "answer from document", "answer from pdf",
                             "read pdf", "analyze pdf", "extract from pdf", "pdf reader",
                             "document reader", "document understanding", "read file",
                             "extract from file", "layoutlm", "donut"],
            "task":         "document-question-answering",
            "queries":      ["document question answering", "layoutlm", "donut", "document understanding"],
            "boost":        ["layoutlm", "donut"],
        },
        # visual question answering — answer questions about an image
        {
            "match_any":    ["visual question", "vqa", "answer question about image",
                             "ask about image", "question about photo", "question about picture",
                             "what is in the image", "image question answering"],
            "task":         "visual-question-answering",
            "queries":      ["visual question answering", "vqa", "blip-2", "idefics"],
            "boost":        ["vqa", "blip"],
        },
        # image captioning / description / reading — reads/describes an image
        {
            "match_any":    ["read image", "describe image", "caption image", "image caption",
                             "analyze image", "understand image", "explain image",
                             "image to text", "image-to-text", "ocr", "extract text from image",
                             "image description", "blip", "llava", "image understanding"],
            "exclude_any":  ["generate", "create", "draw", "paint", "make", "render",
                             "edit", "modify"],
            "task":         "image-to-text",
            "queries":      ["image to text", "blip", "llava", "image captioning"],
            "boost":        ["blip", "llava", "captioning"],
        },
        # object detection
        {
            "match_any":    ["detect object", "object detection", "find objects",
                             "bounding box", "yolo", "count objects", "locate objects",
                             "draw bounding box", "detect bounding"],
            "task":         "object-detection",
            "queries":      ["yolo", "object detection", "detr", "rtdetr"],
            "boost":        ["detection", "yolo"],
        },
        # image classification
        {
            "match_any":    ["classify image", "image classification", "image category",
                             "recognize image", "identify image", "image label",
                             "what category is this image", "label image"],
            "task":         "image-classification",
            "queries":      ["image classification", "vit", "resnet", "efficientnet"],
            "boost":        ["classification", "vit"],
        },
        # depth estimation
        {
            "match_any":    ["depth estimation", "depth map", "3d depth", "monocular depth",
                             "depth from image", "depth anything", "midas"],
            "task":         "depth-estimation",
            "queries":      ["depth estimation", "midas", "depth anything", "dpt"],
            "boost":        ["depth", "midas"],
        },
        # ── image editing ────────────────────────────────────────────────────────
        {
            "match_all":    [["image", "picture", "photo", "img"]],
            "match_any":    ["edit", "modify", "change", "alter", "adjust", "transform",
                             "enhance", "fix", "restore", "colorize", "upscale", "stylize",
                             "inpaint", "outpaint", "restyle"],
            "task":         "image-to-image",
            "queries":      ["flux", "img2img", "instruct-pix2pix", "controlnet"],
            "boost":        ["flux", "img2img", "edit"],
        },
        # ── image generation ─────────────────────────────────────────────────────
        # exclude reading/analyzing words to avoid false positives
        {
            "match_any":    ["generate image", "create image", "make image", "draw",
                             "paint", "sketch", "artwork", "illustration",
                             "text to image", "text-to-image", "generate photo",
                             "image generation", "art generation", "dalle", "midjourney",
                             "stable diffusion", "flux"],
            "exclude_any":  ["read", "analyze", "describe", "understand", "caption",
                             "extract", "ocr", "edit", "modify", "question"],
            "task":         "text-to-image",
            "queries":      ["flux", "stable-diffusion", "sdxl", "playground"],
            "boost":        ["flux", "sdxl", "diffusion"],
        },
        # ── video ────────────────────────────────────────────────────────────────
        {
            "match_any":    ["generate video", "create video", "text to video",
                             "text-to-video", "video generation", "animate", "animation",
                             "cogvideo", "wan video", "video from text"],
            "task":         "text-to-video",
            "queries":      ["text to video", "wan", "cogvideo", "video generation"],
            "boost":        ["video", "animate"],
        },
        # ── audio / speech ───────────────────────────────────────────────────────
        # speech recognition — audio → text
        {
            "match_any":    ["transcribe", "speech to text", "speech-to-text",
                             "voice to text", "audio to text", "recognize speech",
                             "stt", "whisper", "asr", "transcription"],
            "task":         "automatic-speech-recognition",
            "queries":      ["whisper", "speech recognition", "asr", "wav2vec"],
            "boost":        ["whisper", "wav2vec"],
        },
        # text-to-speech — text → spoken audio
        {
            "match_any":    ["text to speech", "text-to-speech", "tts", "read aloud",
                             "voice synthesis", "speak text", "narrate", "voice generation",
                             "convert text to voice", "synthesize speech"],
            "task":         "text-to-speech",
            "queries":      ["text to speech", "tts", "coqui", "xtts", "speecht5"],
            "boost":        ["tts", "coqui"],
        },
        # music / sound generation — text → music/audio
        {
            "match_any":    ["generate music", "create music", "music generation",
                             "compose music", "generate audio", "create audio",
                             "sound effects", "audio generation", "musicgen",
                             "audiocraft", "text to music"],
            "exclude_any":  ["transcribe", "speech to text", "recognize speech", "tts",
                             "text to speech"],
            "task":         "text-to-audio",
            "queries":      ["musicgen", "audio generation", "music", "audiocraft"],
            "boost":        ["musicgen", "audiocraft"],
        },
        # audio classification
        {
            "match_any":    ["audio classification", "classify audio", "sound classification",
                             "identify sound", "music genre", "emotion in voice",
                             "audio tag", "sound recognition"],
            "task":         "audio-classification",
            "queries":      ["audio classification", "sound classification", "ast"],
            "boost":        ["classification", "audio"],
        },
        # ── text tasks ───────────────────────────────────────────────────────────
        # translation
        {
            "match_any":    ["translate", "translation", "convert language",
                             "language translation", "multilingual translate", "nllb", "opus-mt"],
            "task":         "translation",
            "queries":      ["translation", "multilingual", "nllb", "opus-mt", "marian"],
            "boost":        ["translation", "multilingual"],
        },
        # summarization
        {
            "match_any":    ["summarize", "summarization", "condense", "shorten text",
                             "tldr", "key points", "abstract", "extract summary"],
            "task":         "summarization",
            "queries":      ["summarization", "bart", "pegasus", "led"],
            "boost":        ["summarization", "bart"],
        },
        # sentence similarity / embeddings
        {
            "match_any":    ["similarity", "semantic search", "embedding", "embeddings",
                             "vector search", "retrieval", "compare text", "dense retrieval",
                             "sentence encoder", "bi-encoder", "sbert", "bge", "e5"],
            "task":         "sentence-similarity",
            "queries":      ["sentence similarity", "embedding", "sbert", "e5", "bge"],
            "boost":        ["embedding", "sbert"],
        },
        # NER / token classification
        {
            "match_any":    ["named entity", "ner", "entity recognition", "extract entity",
                             "pos tagging", "token classification", "extract entities",
                             "information extraction"],
            "task":         "token-classification",
            "queries":      ["ner", "named entity recognition", "bert ner", "token classification"],
            "boost":        ["ner", "bert"],
        },
        # extractive question answering (text context, not image)
        {
            "match_any":    ["question answering", "reading comprehension",
                             "answer from context", "extract answer", "squad", "qa"],
            "exclude_any":  ["image", "picture", "photo", "visual", "video", "audio"],
            "task":         "question-answering",
            "queries":      ["question answering", "bert qa", "roberta squad", "extractive qa"],
            "boost":        ["qa", "squad"],
        },
        # fill-mask
        {
            "match_any":    ["fill mask", "fill blank", "masked language", "predict token",
                             "mlm", "bert mask"],
            "task":         "fill-mask",
            "queries":      ["bert", "roberta", "fill mask", "masked language model"],
            "boost":        ["bert", "roberta"],
        },
        # zero-shot classification
        {
            "match_any":    ["zero shot classification", "zero-shot classification",
                             "classify without training", "classify text", "text classification",
                             "sentiment", "topic classification"],
            "exclude_any":  ["image", "picture", "photo"],
            "task":         "zero-shot-classification",
            "queries":      ["zero shot classification", "bart mnli", "nli", "text classification"],
            "boost":        ["classification", "nli"],
        },
    ]

    for override in _OVERRIDES:
        # match_all: every inner group must have at least one keyword match
        all_match = all(
            any(kw in _p for kw in group)
            for group in override.get("match_all", [])
        )
        # match_any: at least one positive keyword must match
        any_match = any(kw in _p for kw in override.get("match_any", []))
        # exclude_any: none of the negative keywords must match
        not_excluded = not any(kw in _p for kw in override.get("exclude_any", []))

        if all_match and any_match and not_excluded:
            tasks       = [override["task"]]
            queries     = override["queries"]
            boost_words = override["boost"]
            print(f"[search] Task override applied: {override['task']}")
            break

    # ── Pass 1: collect raw candidates from list_models (quick) ─────────────────
    seen_ids: set[str]  = set()
    raw: list[tuple]    = []   # (model_id, target_task, score, gated, strict)

    # Build full search list: primary tasks + supplemental tasks per primary.
    # Supplemental entries are flagged strict=True so pass 2 only keeps them
    # when their provider mapping explicitly reports the primary task.
    search_plan: list[tuple] = []   # (query, hf_search_task, target_task, strict)
    for q in queries:
        for task in tasks:
            search_plan.append((q, task, task, False))
            for supp in SUPPLEMENTAL_SEARCH_TASKS.get(task, []):
                search_plan.append((q, supp, task, True))

    total_searches = len(search_plan)
    for search_num, (q, hf_task, target_task, strict) in enumerate(search_plan, 1):
        label = f"{hf_task}→{target_task}" if strict else hf_task
        emit("searching", f"Searching for '{q}' ({label}) [{search_num}/{total_searches}]...")
        try:
            models = api.list_models(
                search=q, task=hf_task,
                sort="downloads", direction=-1,
                limit=limit, full=True,
            )
        except TypeError:
            models = api.list_models(search=q, filter=hf_task, limit=limit, full=True)

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

            gated = bool(getattr(model, "gated", False))
            raw.append((mid, target_task, score, gated, strict))

    # ── Pass 2: fetch full provider/method info per model (the slow part) ────────
    total = len(raw)
    emit("analyzing", f"Found {total} candidates. Checking provider availability...")

    suggestions: list[ModelSuggestion] = []
    for i, (mid, task, score, gated, strict) in enumerate(raw):
        if i % 5 == 0:
            emit("progress", f"Checking model {i + 1}/{total}...")
        mi = get_model_full_info(api, mid, task, strict_task=strict)
        suggestions.append(ModelSuggestion(
            name=mid,
            url=HF_MODEL_URL + mid,
            score=score,
            is_free=True,   # HF serverless inference is free with HF_TOKEN; gated ≠ paid
            has_provider=mi["provider"] is not None,
            provider=mi["provider"],
            supported_task=mi["supported_task"],
            inferred_method=mi["inferred_method"],
            has_chat_template=mi["has_chat_template"],
            pipeline_tag=mi["pipeline_tag"],
            tags=mi["tags"],
            gated=gated,
            license=mi.get("license"),
        ))

    emit("ranking", f"Ranking {len(suggestions)} models by score and availability...")
    # Sort: provider first, then ungated before gated, then by score
    suggestions.sort(key=lambda x: (not x.has_provider, x.gated, -x.score))
    return suggestions, tasks[0]