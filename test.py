import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_NAME = "black-forest-labs/FLUX.2-klein-4B"
PROVIDER = "fal-ai"

OUTPUT_DIR = Path("generated_images")


def create_client() -> InferenceClient:
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN is missing. Add it to your .env file.")

    return InferenceClient(
        provider=PROVIDER,
        api_key=HF_TOKEN,
    )


def validate_image_path(image_path: str) -> Path:
    path = Path(image_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    return path


def make_output_path(prompt: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    safe_prompt = "".join(
        char for char in prompt[:40]
        if char.isalnum() or char in " _-"
    ).strip().replace(" ", "_")

    filename = f"{timestamp}_{safe_prompt or 'edited_image'}.png"

    return OUTPUT_DIR / filename


def edit_image(
    client: InferenceClient,
    image_path: str,
    edit_prompt: str,
) -> Path:
    image_file = validate_image_path(image_path)

    if not edit_prompt.strip():
        raise ValueError("Edit description is required.")

    with open(image_file, "rb") as file:
        image_bytes = file.read()

    result_image = client.image_to_image(
        image=image_bytes,
        prompt=edit_prompt,
        model=MODEL_NAME,
    )

    output_path = make_output_path(edit_prompt)
    result_image.save(output_path)

    return output_path.resolve()


def main():
    client = create_client()

    print("Agent ready. Type exit or quit to stop.")
    print(f"Model   : {MODEL_NAME}")
    print(f"Provider: {PROVIDER}")
    print("Method  : image_to_image")

    while True:
        image_path = input("\nImage path: ").strip().strip('"')
        if image_path.lower() in {"exit", "quit"}:
            break

        edit_prompt = input("Edit description: ").strip()
        if edit_prompt.lower() in {"exit", "quit"}:
            break

        try:
            output_path = edit_image(client, image_path, edit_prompt)
            print(f"\nEdited image saved: {output_path}")

        except Exception as error:
            print(f"\nError: {error}")


if __name__ == "__main__":
    main()