import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = "Qwen/Qwen3-0.6B"
PROVIDER = None

SYSTEM_PROMPT = """
You are analyzing and reading PDF files. You are capable of understanding the content
through various means such as text summarization, content categorization, keyword extraction,
and information linking. Your tasks include, but not limited to, analyzing the content of
a given PDF file. The output should be detailed and in a clear format. Your input is a PDF
file, and your answer should adhere to the specified format generally provided in a question
or answer format about the content inside the PDF. Always consider the user needs for the
specific details or insights needed.
""".strip()

PLAN = {
    "final_system_prompt": SYSTEM_PROMPT,
    "method": "document_question_answering",
    "tools": [],
    "routing_rules": [],
    "temperature": 0.7,
    "max_tokens": 800,
    "prompt_template": f"{SYSTEM_PROMPT}\n\nUser: {{user_input}}\nAssistant:",
    "system_prompt": SYSTEM_PROMPT,
}

HISTORY: list = []

client = (
    InferenceClient(provider=PROVIDER, api_key=HF_TOKEN)
    if PROVIDER
    else InferenceClient(api_key=HF_TOKEN)
)


def run_tools(user_input: str) -> str:
    return ""


def run_inference(client, model_name, plan, user_input):
    if plan["method"] == "document_question_answering":
        return f"Method {plan['method']} is handled below."

    if plan["method"] == "automatic_speech_recognition":
        return client.automatic_speech_recognition(user_input, model=model_name)

    if plan["method"] == "image_to_text":
        return client.image_to_text(user_input, model=model_name)

    if plan["method"] == "summarization":
        return client.summarization(user_input, model=model_name)

    if plan["method"] == "question_answering":
        parts = user_input.split("|", 1)
        return client.question_answering(
            question=parts[0].strip(),
            context=parts[1].strip() if len(parts) > 1 else "",
            model=model_name,
        )

    raise ValueError(f"Unsupported method: {plan['method']}")


def run_agent(user_input: str):
    return run_inference(
        client=client,
        model_name=MODEL_NAME,
        plan=PLAN,
        user_input=user_input,
    )


def main():
    print("Agent ready. Type exit or quit to stop.")
    print(f"Model   : {MODEL_NAME}")
    print(f"Provider: {PROVIDER}")
    print("Method  : document_question_answering")
    print("Tools   : []")

    while True:
        user_input = input("\nEnter your message: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            break

        try:
            result = run_agent(user_input)
            print("\nResult:")
            print(result)
        except Exception as e:
            print("\nError:", e)


if __name__ == "__main__":
    main()