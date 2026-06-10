"""
config.py — all static constants for the agent builder.
No logic, no imports beyond the standard library.
"""

# ── Hugging Face ──────────────────────────────────────────────────────────────

HF_MODEL_URL = "https://huggingface.co/"

# Models tried in order when calling the planner (fallback chain).
PLANNER_MODELS = [
    "Qwen/Qwen2.5-3B-Instruct:featherless-ai",
    "Qwen/Qwen2.5-7B-Instruct:featherless-ai",
    "mistralai/Mistral-7B-Instruct-v0.3:featherless-ai",
]

# ── Task → InferenceClient method mapping ─────────────────────────────────────
# Keys are HF pipeline tags; values are the corresponding InferenceClient
# method names used in generated agent code.

TASK_TO_METHOD: dict[str, str] = {
    # ── text ──────────────────────────────────────────────────────────────────
    "text-generation":                  "text_generation",
    "conversational":                   "chat_completion",
    "text2text-generation":             "text_generation",
    "translation":                      "text_generation",
    "summarization":                    "summarization",
    "question-answering":               "question_answering",
    "sentence-similarity":              "sentence_similarity",
    "feature-extraction":               "feature_extraction",
    "fill-mask":                        "fill_mask",
    "token-classification":             "token_classification",
    "zero-shot-classification":         "zero_shot_classification",
    # ── image generation / editing ────────────────────────────────────────────
    "text-to-image":                    "text_to_image",
    "image-to-image":                   "image_to_image",
    # ── image understanding ───────────────────────────────────────────────────
    "image-to-text":                    "image_to_text",
    "visual-question-answering":        "visual_question_answering",
    "document-question-answering":      "document_question_answering",
    "image-classification":             "image_classification",
    "zero-shot-image-classification":   "zero_shot_image_classification",
    "object-detection":                 "object_detection",
    "depth-estimation":                 "depth_estimation",
    # ── audio ─────────────────────────────────────────────────────────────────
    "automatic-speech-recognition":     "automatic_speech_recognition",
    "text-to-speech":                   "text_to_speech",
    "text-to-audio":                    "text_to_audio",
    "audio-classification":             "audio_classification",
    # ── video ─────────────────────────────────────────────────────────────────
    "text-to-video":                    "text_to_video",
}

# ── Chat-template detection ───────────────────────────────────────────────────
# If any of these strings appear in a model's tags or ID, it supports
# chat_completion (vs plain text_generation).

CHAT_TEMPLATE_INDICATORS: list[str] = [
    "conversational",
    "chat-template",
    "instruct",
]

# ── Inference methods that should never receive tools ────────────────────────
# Used by decide_tools() to short-circuit tool selection for non-text tasks.

IMAGE_AUDIO_METHODS: tuple[str, ...] = (
    # image generation / editing — no text tools needed
    "text_to_image",
    "image_to_image",
    # image understanding — input is an image, output is text/labels
    "image_to_text",
    "visual_question_answering",
    "document_question_answering",
    "image_classification",
    "zero_shot_image_classification",
    "object_detection",
    "depth_estimation",
    # audio — binary input/output
    "text_to_speech",
    "text_to_audio",
    "automatic_speech_recognition",
    "audio_classification",
    # video generation
    "text_to_video",
    # embeddings — vector output, no text tools needed
    "feature_extraction",
)

# ── Chat agent defaults ───────────────────────────────────────────────────────
# Tools added automatically when the AI returns an empty list for a chat agent.

CHAT_DEFAULT_TOOLS: list[str] = ["web_search", "memory", "datetime"]

# Tools that must always be present for chat/text-generation agents.
CHAT_REQUIRED_TOOLS: list[str] = ["memory", "web_search"]

# ── Retry / networking ────────────────────────────────────────────────────────

PLANNER_RETRIES: int   = 3
PLANNER_BASE_DELAY: float = 5.0   # seconds; multiplied by attempt number

# HTTP status codes / substrings that indicate a transient error worth retrying.
TRANSIENT_ERROR_MARKERS: tuple[str, ...] = (
    "504", "502", "503", "timeout", "Gateway", "connection",
)

# ── Generation defaults ───────────────────────────────────────────────────────

DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_MAX_TOKENS:  int   = 800
HISTORY_MAX_MESSAGES: int  = 20   # kept per session in generated agents

# ── Model search ──────────────────────────────────────────────────────────────

MODEL_SEARCH_LIMIT: int = 200     # max models fetched per HF search query

# When searching for a specific task, many capable models are tagged with a
# different primary pipeline_tag on HuggingFace (e.g. FLUX is "text-to-image"
# but supports image-to-image via inference providers). These supplemental tasks
# are searched in addition to the primary task. Results are only kept if the
# provider mapping explicitly reports the primary task.
SUPPLEMENTAL_SEARCH_TASKS: dict = {
    "image-to-image":              ["text-to-image"],
    "image-to-text":               ["text-generation"],
    "visual-question-answering":   ["text-generation", "image-to-text"],
    "document-question-answering": ["text-generation", "image-to-text"],
    "text-to-video":               ["text-to-image"],
    "text-to-speech":              ["audio-to-audio"],
    "automatic-speech-recognition": ["audio-to-audio"],
    "depth-estimation":            ["image-to-image"],
    "object-detection":            ["image-classification"],
    "audio-classification":        ["audio-to-audio"],
    "image-classification":        ["image-to-image"],
}

# Score weights used when ranking candidate models.
SCORE_WEIGHT_DOWNLOADS: float = 0.6
SCORE_WEIGHT_LIKES:     float = 10.0
SCORE_BOOST_KEYWORD:    float = 5000.0