"""
utils/llm.py — shared LLM call helpers.
"""

import random
import time

from huggingface_hub import InferenceClient

from config import (
    PLANNER_MODELS,
    PLANNER_RETRIES,
    PLANNER_BASE_DELAY,
    TRANSIENT_ERROR_MARKERS,
)


def chat_completion_with_retry(
    client: InferenceClient,
    messages: list[dict],
    max_tokens: int = 500,
    retries: int = PLANNER_RETRIES,
    base_delay: float = PLANNER_BASE_DELAY,
) -> object:
    """
    Try each model in PLANNER_MODELS in order, retrying on transient errors.
    Raises RuntimeError if every model fails.
    """
    last_error = None

    for model in PLANNER_MODELS:
        for attempt in range(1, retries + 1):
            try:
                print(f"  [planner] {model}  attempt={attempt}")
                return client.chat_completion(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                last_error = e
                err = str(e)
                transient = any(marker in err for marker in TRANSIENT_ERROR_MARKERS)

                if transient and attempt < retries:
                    delay = base_delay * attempt + random.uniform(0, 2)
                    print(f"  ⚠️  Retrying in {delay:.1f}s…")
                    time.sleep(delay)
                elif attempt == retries:
                    print(f"  ❌ {model} failed. Trying next…")
                    break
                else:
                    raise

    raise RuntimeError(f"All planner models failed. Last error: {last_error}")