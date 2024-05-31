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

from .langchain import Langchain


class GeneralRespond:
    """General Respond Class - Answers questions without requiring context data"""

    def __init__(self, langchain: Langchain):
        """
        Initialize the General Respond Class

        Args:
            langchain: The langchain instance
        """
        self._langchain = langchain

    def run(
        self,
        question: str,
        model_name="gpt-4o-mini",
        temperature=0,
        callbacks=[],
    ) -> str:
        """
        Respond to general user questions using AI knowledge

        Args:
            question: The specific question to answer
            model_name: The model name
            temperature: The temperature
            callbacks: The callbacks

        Returns:
            The response to the user question
        """
        chain = self._langchain.create_chat_chain(
            model_name,
            temperature,
            [
                (
                    "system",
                    "You are a knowledgeable and helpful AI assistant that answers user questions based on your training data. "
                    "Your role is to:\n"
                    "- Provide accurate and helpful answers to user questions\n"
                    "- Use your knowledge to give comprehensive responses\n"
                    "- Be concise but thorough in your answers\n"
                    "- Maintain a helpful and professional tone\n"
                    "- Use markdown formatting to structure your responses clearly\n"
                    "- Use headers, lists, bold, italic, and code formatting when appropriate\n"
                    "- Ensure proper spacing and readability in your markdown\n"
                    "- If you're unsure about something, acknowledge the limitations of your knowledge",
                ),
                (
                    "user",
                    f"Question: {question}\n\nPlease answer this question based on your knowledge.",
                ),
            ],
            callbacks,
        )

        return chain.invoke({})
