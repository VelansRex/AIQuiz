
import os
import sys
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables from .env
load_dotenv()

# Azure OpenAI Client Configuration
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

deployment_name = os.getenv("DEPLOYMENT_NAME")
# Loading a Text File
def resource_path(relative_path):
    # Path to resource for PyInstaller
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# System Prompt
system_prompt = """
    You are a quiz assistant. Given a text, generate one multiple-choice question with four options (a, b, c, d),
    and mark the correct answer with a star (*).
    """

# Function for Generating a Question with Streaming
def generate_streamed_question(text):
    user_prompt_template = """
    Given this educational text, generate one multiple-choice question with four options (a, b, c, d).
    Mark the correct answer with an asterisk (*). For example:

    Question: What is the capital of France?
    a) Berlin
    b) Madrid
    c) Paris*
    d) Rome
    
    TEXT:
    {text}
    """
    prompt = user_prompt_template.format(text=text)

    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300,
        stream=True  # Small chunks of information from the model, part of a conversation is processed before waiting for the entire output.
    )

    full_output = ""
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            part = chunk.choices[0].delta.content
            print(part, end="", flush=True)
            full_output += part
    print()
    return full_output.strip()

# Application Section
if __name__ == "__main__":
    content = resource_path("ArtificialIntelligenceExamples.txt")

    print("Name: AI Quiz \nDescription: Azure OpenAI-Powered AI Quiz")
    print("About: The app uses Microsoft Azure OpenAI API and GPT-4o-mini model.")
    print("Instruction:")
    print("\
        1. Download AI Quiz: https://github.com/VelansRex/AIQuiz/releases/download/Use/AIQuiz.exe\n\
        2. Run File: AIQuiz.exe\n\
        3. Project Repo: https://github.com/VelansRex/AIQuiz.git\n\
        ")
    print("--------------------")

    if content:
        print("\nQuiz Question:\n")
        for i in range(5):
            print(f"\nQuestion {i+1}:")
            generate_streamed_question(content)

    print()
    input("Press Enter to exit...")

