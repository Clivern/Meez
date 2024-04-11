# Copyright 2025 Clivern
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from meez.data import TextReader, JsonReader, FileReader
from meez.core.langchain import Langchain
from meez.core.respond import Respond


def create_sample_file():
    """Create a sample text file for demonstration"""
    sample_content = """
    Python Programming Language

    Python is a high-level, interpreted programming language known for its simplicity and readability.
    It was created by Guido van Rossum and first released in 1991.

    Key Features:
    - Easy to learn and use
    - Extensive standard library
    - Cross-platform compatibility
    - Strong community support
    - Used in web development, data science, AI, and automation

    Popular frameworks include Django, Flask, FastAPI for web development,
    and NumPy, Pandas, TensorFlow for data science and machine learning.
    """

    with open("sample_python_info.txt", "w") as f:
        f.write(sample_content)

    return "sample_python_info.txt"


def example_with_text_reader():
    """Example using TextReader with Respond"""
    print("=== Example 1: Using TextReader ===\n")

    # Sample text data
    text_data = """
    Artificial Intelligence (AI) is a branch of computer science that aims to create
    intelligent machines that work and react like humans. Some of the activities
    computers with artificial intelligence are designed for include speech recognition,
    learning, planning, and problem solving.

    Machine Learning is a subset of AI that enables computers to learn and improve
    from experience without being explicitly programmed. It focuses on developing
    algorithms that can access data and use it to learn for themselves.
    """

    # Create text reader
    text_reader = TextReader(text_data)

    # Create langchain instance (you'll need to set your OpenAI API key)
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return

    langchain = Langchain(openai_api_key)
    respond = Respond(langchain)

    # Example questions to ask about the text
    questions = [
        "What is Artificial Intelligence?",
        "How does Machine Learning relate to AI?",
        "What are some applications of AI?",
    ]

    for question in questions:
        print(f"Question: {question}")

        try:
            response = respond.run(
                question=question,
                data=text_reader,
                model_name="gpt-4o-mini",
                temperature=0,
            )
            print(f"Answer: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")


def example_with_json_reader():
    """Example using JsonReader with Respond"""
    print("=== Example 2: Using JsonReader ===\n")

    # Sample JSON data
    json_data = {
        "company": {
            "name": "TechCorp",
            "founded": 2010,
            "industry": "Software Development",
            "employees": 250,
            "products": [
                {
                    "name": "CloudManager",
                    "type": "Cloud Infrastructure",
                    "users": 50000,
                },
                {
                    "name": "DataAnalytics",
                    "type": "Business Intelligence",
                    "users": 25000,
                },
            ],
            "locations": [
                {"city": "San Francisco", "country": "USA"},
                {"city": "London", "country": "UK"},
                {"city": "Tokyo", "country": "Japan"},
            ],
        }
    }

    # Create JSON reader
    json_reader = JsonReader(json_data)

    # Create langchain instance
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return

    langchain = Langchain(openai_api_key)
    respond = Respond(langchain)

    # Example questions about the JSON data
    questions = [
        "What is the company name and when was it founded?",
        "How many products does the company have and what are they?",
        "In which countries does the company have offices?",
    ]

    for question in questions:
        print(f"Question: {question}")

        try:
            response = respond.run(
                question=question,
                data=json_reader,
                model_name="gpt-4o-mini",
                temperature=0,
            )
            print(f"Answer: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")


def example_with_file_reader():
    """Example using FileReader with Respond"""
    print("=== Example 3: Using FileReader ===\n")

    # Create a sample file
    file_path = create_sample_file()

    try:
        # Create file reader
        file_reader = FileReader(file_path)

        # Create langchain instance
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("Please set OPENAI_API_KEY environment variable")
            return

        langchain = Langchain(openai_api_key)
        respond = Respond(langchain)

        # Example questions about the file content
        questions = [
            "Who created Python and when was it first released?",
            "What are the key features of Python?",
            "What are some popular Python frameworks mentioned?",
        ]

        for question in questions:
            print(f"Question: {question}")

            try:
                response = respond.run(
                    question=question,
                    data=file_reader,
                    model_name="gpt-4o-mini",
                    temperature=0,
                )
                print(f"Answer: {response}\n")
            except Exception as e:
                print(f"Error: {e}\n")

    finally:
        # Clean up the sample file
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up: {file_path}")


def main():
    """Main function to run all examples"""
    print("Meez Data and Respond Examples")
    print("=" * 50)
    print("This example demonstrates how to use different data readers")
    print("with the Respond class to answer questions based on context.\n")

    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Warning: OPENAI_API_KEY environment variable is not set.")
        print("   Please set it to run the examples with actual AI responses.")
        print("   Example: export OPENAI_API_KEY='your-api-key-here'\n")

    # Run examples
    example_with_text_reader()
    example_with_json_reader()
    example_with_file_reader()

    print("All examples completed!")


if __name__ == "__main__":
    main()
