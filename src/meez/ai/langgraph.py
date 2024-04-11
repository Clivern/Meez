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

from typing import Any, Callable, Dict, Generator, List, Optional, TypedDict
from langgraph.graph import StateGraph, END


class MainState(TypedDict):
    messages: list[Dict[str, str]]


class LangGraph:
    """
    LangGraph Class for creating and managing graph workflows with multiple actions
    """

    def __init__(self, state_type: Optional[type] = MainState):
        """
        Initialize the LangGraph Class

        Args:
            state_type: The type of state to use for the graph (default: Dict)
        """
        self._state_type = state_type
        self._graph = StateGraph(self._state_type)
        self._nodes = {}
        self._entry_point = None

    def add_node(
        self, name: str, action: Callable, description: str = ""
    ) -> "LangGraph":
        """
        Add a node to the graph

        Args:
            name: Name of the node
            action: Function to execute at this node
            description: Optional description of the node

        Returns:
            self for method chaining
        """
        self._nodes[name] = {"action": action, "description": description}

        self._graph.add_node(name, action)

        return self

    def add_edge(self, from_node: str, to_node: str) -> "LangGraph":
        """
        Add an edge between nodes

        Args:
            from_node: Source node name
            to_node: Target node name (can be END for terminal nodes)

        Returns:
            self for method chaining
        """
        self._graph.add_edge(from_node, to_node)

        return self

    def add_conditional_edge(self, from_node: str, condition: Callable) -> "LangGraph":
        """
        Add a conditional edge between nodes

        Args:
            from_node: Source node name
            condition: Conditional function to determine if edge should be taken

        Returns:
            self for method chaining
        """
        self._graph.add_conditional_edges(from_node, condition)

        return self

    def set_entry_point(self, node_name: str) -> "LangGraph":
        """
        Set the entry point of the graph

        Args:
            node_name: Name of the entry node

        Returns:
            self for method chaining
        """
        self._entry_point = node_name

        self._graph.set_entry_point(self._entry_point)

        return self

    def add_finish_point(self, node_name: str) -> "LangGraph":
        """
        Add a finish point to the graph

        Args:
            node_name: Name of the node that should finish the graph

        Returns:
            self for method chaining
        """
        self._graph.add_edge(node_name, END)

        return self

    def compile(self) -> Any:
        """
        Compile the graph into an executable workflow

        Returns:
            Compiled graph workflow
        """
        return self._graph.compile()

    def run(
        self, initial_state: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run the graph with initial state

        Args:
            initial_state: Initial state for the graph
            config: Optional configuration for the run

        Returns:
            Final state after graph execution
        """
        compiled_graph = self.compile()

        return compiled_graph.invoke(initial_state, config=config)

    def stream(
        self, initial_state: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream the graph execution step by step

        Args:
            initial_state: Initial state for the graph
            config: Optional configuration for the run

        Returns:
            Generator yielding each step of execution
        """
        compiled_graph = self.compile()

        return compiled_graph.stream(initial_state, config=config)

    def get_node_info(self, node_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific node

        Args:
            node_name: Name of the node

        Returns:
            Node information or None if not found
        """
        return self._nodes.get(node_name)

    def list_nodes(self) -> List[str]:
        """
        Get list of all node names

        Returns:
            List of node names
        """
        return list(self._nodes.keys())

    def validate_graph(self) -> bool:
        """
        Validate that the graph is properly configured

        Returns:
            True if graph is valid, False otherwise
        """
        if not self._entry_point:
            return False
        if not self._nodes:
            return False
        if self._entry_point not in self._nodes:
            return False
        return True
