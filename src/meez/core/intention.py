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

from typing import List
from .langchain import Langchain


class Intention:
    """Intention Class"""

    def __init__(self, langchain: Langchain, intentions: List[str]):
        """
        Initialize the Intention Class

        Args:
            langchain: The langchain instance
            intentions: The list of intentions
        """
        self._langchain = langchain
        self._intentions = intentions

    def detect(
        self, text: str, model_name="gpt-4o-mini", temperature=0, callbacks=[]
    ) -> str:
        """
        Detect the intention of the text

        Args:
            text: The text to detect the intention of
            model_name: The model name
            temperature: The temperature
            callbacks: The callbacks

        Returns:
            The intention of the text
        """
        chain = self._langchain.create_chat_chain(
            model_name,
            temperature,
            [
                (
                    "system",
                    "You are a helpful assistant that provides the user intent from a list of intents.",
                ),
                (
                    "user",
                    f"Provide the intent value ONLY of user prompt {text} from these intent list {self._intentions}",
                ),
            ],
            callbacks,
        )

        return chain.invoke({})
