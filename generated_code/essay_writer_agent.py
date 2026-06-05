import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = "Dias1999/serbian-essay-writer"
SYSTEM_PROMPT = """You are an AI essay writer. Your task is to generate academic essays in response to user requests. Ensure the essays are tailored to the specified level of English proficiency, length, and tone. You may generate essays from various topics and for different audiences as requested. Direct users to export essays as PDF files if that is their preference."""

PLAN = {
    "method": "text_generation",
    "input_kind": "text",
    "input_label": "user_input",
    "system_prompt": "You are an AI essay writer. Your task is to generate academic essays in response to user requests. Ensure the essays are tailored to the specified level of English proficiency, length, and tone. You may generate essays from various topics and for different audiences as requested. Direct users to export essays as PDF files if that is their preference.",
    "prompt_template": "Generate an academic essay based on the topic provided by {user_input}. Make sure it is suitable for the level defined by the user and export it as a PDF.",
    "output_kind": "text",
    "temperature": 0.7,
    "max_tokens": 800
}

client = InferenceClient(api_key=HF_TOKEN)


def run_inference(client, model_name, plan, user_input):
    method = plan["method"]

    if method == "chat_completion":
        response = client.chat_completion(
            model=model_name,
            messages=[
                {"role": "system", "content": plan["system_prompt"]},
                {"role": "user", "content": user_input}
            ],
            max_tokens=plan.get("max_tokens", 800),
            temperature=plan.get("temperature", 0.7),
        )
        return response.choices[0].message.content

    if method == "text_generation":
        prompt = plan["prompt_template"].replace("{user_input}", user_input)

        return client.text_generation(
            prompt,
            model=model_name,
            max_new_tokens=plan.get("max_tokens", 800),
            temperature=plan.get("temperature", 0.7),
        )

    if method == "text_to_image":
        image = client.text_to_image(
            prompt=user_input,
            model=model_name,
        )

        output_path = "generated_image.png"
        image.save(output_path)
        return f"Image saved to {output_path}"

    if method == "automatic_speech_recognition":
        return client.automatic_speech_recognition(
            user_input,
            model=model_name,
        )

    if method == "image_to_text":
        return client.image_to_text(
            user_input,
            model=model_name,
        )

    raise ValueError(f"Unsupported method: {method}")



def run_agent(user_input: str):
    return run_inference(
        client=client,
        model_name=MODEL_NAME,
        plan=PLAN,
        user_input=user_input
    )


def main():
    print("Agent is ready. Type exit or quit to stop.")

    while True:
        user_input = input("\nuser_input: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            break

        try:
            result = run_agent(user_input)
            print("\nResult:")
            print(result)
        except Exception as e:
            print("\nError:")
            print(e)


if __name__ == "__main__":
    main()