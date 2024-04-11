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
from meez.core import Langchain, Intention


def intention_detection():
    """Simple example of intention detection"""

    # 1. Get your OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("Error: Please set OPENAI_API_KEY environment variable")
        return

    # 2. Create Langchain instance
    langchain = Langchain(openai_api_key=api_key)

    # 3. Define your possible intentions
    intentions = [
        "greeting",
        "weather",
        "joke",
        "help",
        "goodbye",
        "book_appointment",
        "cancel_appointment",
        "reschedule_appointment",
        "check_availability",
        "get_directions",
        "contact_support",
        "view_profile",
        "unknown",
    ]

    # 4. Create intention detector
    detector = Intention(langchain=langchain, intentions=intentions)

    # 5. Test some inputs
    test_texts = [
        "Hello there!",
        "What's the weather like?",
        "Tell me a joke",
        "I need help",
        "Goodbye!",
        "I need to book an appointment for next week",
        "Can you help me cancel my appointment?",
        "What time slots are available tomorrow?",
        "How do I get to your office?",
        "I want to pay my bill",
        "I need to talk to customer service",
        "Show me my account information",
        "Can I move my appointment to Friday?",
    ]

    print("Simple Intention Detection Example")
    print("==================================\n")

    for text in test_texts:
        try:
            intention = detector.detect(text)
            print(f"Input: '{text}' → Intention: {intention}")
        except Exception as e:
            print(f"Input: '{text}' → Error: {e}")


if __name__ == "__main__":
    intention_detection()
