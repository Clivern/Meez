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

import json
from meez.util import FileSystem
from abc import ABC, abstractmethod


class DataReader(ABC):
    """
    Data Reader Class
    """

    @abstractmethod
    def to_string(self) -> str:
        """
        Convert the data to a string representation.

        Returns:
            str: String representation of the data
        """
        pass


class FileReader(DataReader):
    """
    File Data Reader Class
    """

    def __init__(self, file_path: str):
        """
        Initialize the File Data Reader Class

        Args:
            file_path: The path to the file to read from
        """
        self._file_path = file_path

    def to_string(self) -> str:
        """
        Convert the data to a string representation.

        Returns:
            str: String representation of the data
        """
        return FileSystem.read_file(self._file_path)


class TextReader(DataReader):
    """
    Text Data Reader Class
    """

    def __init__(self, text: str):
        """
        Initialize the Text Data Reader Class
        """
        self._text = text

    def to_string(self) -> str:
        """
        Convert the data to a string representation.

        Returns:
            str: String representation of the data
        """
        return self._text.strip()


class JsonReader(DataReader):
    """
    Json Data Reader Class
    """

    def __init__(self, json_data: dict):
        """
        Initialize the Json Data Reader Class

        Args:
            json_data: The json data to read from
        """
        self._json_data = json_data

    def to_string(self) -> str:
        """
        Convert the data to a string representation.

        Returns:
            str: String representation of the data
        """
        return json.dumps(self._json_data).replace("{", "{{").replace("}", "}}").strip()


if __name__ == "__main__":
    file_reader = FileReader("test.txt")
    print(file_reader.to_string())

    text_reader = TextReader("Hello, world!")
    print(text_reader.to_string())

    json_reader = JsonReader({"name": "John", "age": 30})
    print(json_reader.to_string())
