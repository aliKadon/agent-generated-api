import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN   = os.getenv('HF_TOKEN')
MODEL_NAME = "black-forest-labs/FLUX.2-klein-4B"
PROVIDER   = "fal-ai"

# HF task: image-to-image | method: image_to_image | chat_template: False
# Tools  : []

SYSTEM_PROMPT = "I expect you to edit the image according to the given text instructions. The input to the system will be an image and a text string describing the desired changes. The output will also be an image."

PLAN = {
    "final_system_prompt": "I expect you to edit the image according to the given text instructions. The input to the system will be an image and a text string describing the desired changes. The output will also be an image.",
    "method": "image_to_image",
    "tools": [],
    "routing_rules": [],
    "temperature": 0.7,
    "max_tokens": 800,
    "prompt_template": "I expect you to edit the image according to the given text instructions. The input to the system will be an image and a text string describing the desired changes. The output will also be an image.\n\nUser: {user_input}\nAssistant:",
    "system_prompt": "I expect you to edit the image according to the given text instructions. The input to the system will be an image and a text string describing the desired changes. The output will also be an image."
}

# Conversation history — keeps context across turns (max 20 messages)
HISTORY: list = []

client = InferenceClient(provider=PROVIDER, api_key=HF_TOKEN)

# ── tool implementations ─────────────────────────────────────────────────


# ── tool router ──────────────────────────────────────────────────────────
def run_tools(user_input: str) -> str:
    return ""


# ── inference ────────────────────────────────────────────────────────────
def run_inference(client, model_name, plan, user_input):
    if plan['method'] == "image_to_image":
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
        return f"Edited image saved: {out_path}"

    # ── fallback handlers ─────────────────────────────────────────────────
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
    print('Model   : black-forest-labs/FLUX.2-klein-4B')
    print('Provider: fal-ai')
    print('Method  : image_to_image')
    print('Tools   : []')
    while True:
        image_path  = input('\nImage path (drag & drop): ').strip().strip('"')
        edit_prompt = input('Edit description         : ').strip()
        if image_path.lower() in ['exit','quit'] or edit_prompt.lower() in ['exit','quit']:
            break
        user_input = image_path + '|||' + edit_prompt
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