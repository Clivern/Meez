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

import langchain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


langchain.verbose = False
langchain.debug = False
langchain.llm_cache = False


class Langchain:
    """Langchain Class"""

    def __init__(self, openai_api_key: str):
        """
        Initialize the Langchain Class

        Args:
            openai_api_key: The OpenAI API key
        """
        self._openai_api_key = openai_api_key

    def create_chat_chain(
        self,
        model_name="gpt-4o-mini",
        temperature=0,
        prompt_template=None,
        callbacks=[],
    ):
        """
        Create a chat chain

        Args:
            openai_api_key: The OpenAI API key
            model_name: The model name
            temperature: The temperature
            prompt_template: The prompt template
            callbacks: The callbacks

        Returns:
            The chat chain
        """
        prompt = ChatPromptTemplate.from_messages(prompt_template)

        llm = ChatOpenAI(
            openai_api_key=self._openai_api_key,
            model_name=model_name,
            temperature=temperature,
            callbacks=callbacks,
        )

        chain = prompt | llm | StrOutputParser()

        return chain
