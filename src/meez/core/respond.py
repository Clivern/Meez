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

from meez.data import DataReader
from .langchain import Langchain


class Respond:
    """Respond Class"""

    def __init__(self, langchain: Langchain):
        """
        Initialize the Respond Class

        Args:
            langchain: The langchain instance
        """
        self._langchain = langchain

    def run(
        self,
        question: str,
        data: DataReader,
        model_name="gpt-4o-mini",
        temperature=0,
        callbacks=[],
    ) -> str:
        """
        Respond to the user questions using a provided context data

        Args:
            question: The specific question to answer
            data: The data to respond to
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
                    "You are a knowledgeable assistant that answers user questions based on the provided context data. "
                    "Your role is to:\n"
                    "- Provide accurate and helpful answers to user questions\n"
                    "- Use only the information available in the provided context\n"
                    "- If the context doesn't contain enough information to answer a question, clearly state what information is missing\n"
                    "- Be concise but thorough in your responses\n"
                    "- Maintain a helpful and professional tone",
                ),
                (
                    "user",
                    f"Context data: {data.to_string()}\n\nQuestion: {question}\n\nPlease answer the question based on the context data provided.",
                ),
            ],
            callbacks,
        )

        return chain.invoke({})
