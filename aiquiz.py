
"""
Name: AI Quiz
Description: Azure OpenAI-Powered Quiz

Instruction:
1. Download AI Quiz Git repository: git clone https://github.com/twoj-user/ai-quiz-app.git
2. Install openai and python-dotenv libraries: pip install -r requirements.txt
2. Run the app with command: python aiquiz.py

About App:
The app uses Microsoft Azure OpenAI API with free credits from a free Azure account, which are valid until 28-04-2025.
"""

import os
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
def read_text_from_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return ""

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
    content = read_text_from_file("ArtificialIntelligenceExamples.txt")

    if content:
        print("\nQuiz Question:\n")
        for i in range(5):
            print(f"\nQuestion {i+1}:")
            generate_streamed_question(content)