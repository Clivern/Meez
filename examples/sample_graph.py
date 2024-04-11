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

from meez.core.langgraph import LangGraph, MainState


def step1(state: MainState) -> MainState:
    """First step: Process input"""

    print("Step 1: Processing...")

    messages = state.get("messages", [])

    if messages:
        user_input = messages[-1].get("content", "")
        result = f"Processed: {user_input.upper()}"

        return {"messages": messages + [{"role": "assistant", "content": result}]}

    return state


def step2(state: MainState) -> MainState:
    """Second step: Generate response"""

    print("Step 2: Generating response...")

    messages = state.get("messages", [])

    response = "Hello! I've processed your request."

    return {"messages": messages + [{"role": "assistant", "content": response}]}


def main():
    """Simple LangGraph example"""

    print("=== Simple LangGraph Example ===\n")

    # Create graph
    graph = LangGraph()

    # Add nodes
    graph.add_node("step1", step1, "Process input")
    graph.add_node("step2", step2, "Generate response")

    # Set entry point
    graph.set_entry_point("step1")

    # Add edges
    graph.add_edge("step1", "step2")
    graph.add_finish_point("step2")

    user_input = input("Enter your message: ")

    # Initial state
    initial_state = {
        "messages": [
            # Old Messages from the database
            {
                "role": "user",
                "content": "what is the company name?",
                "id": "caa5277e-e4f7-4b3a-af1d-c7307cef34a0",
                "createdAt": "2025-01-01 11:00:00",
            },
            {
                "role": "assistant",
                "content": "The company name is Meez",
                "id": "56dc6ff1-8e23-473f-9dd2-3801b10fb0e1",
                "createdAt": "2025-01-01 11:00:00",
            },
            # New Message from the user
            {
                "role": "user",
                "content": user_input,
                "id": "caa5277e-e4f7-4b3a-af1d-c7307cef34a0",
                "createdAt": "2025-01-01 12:00:00",
            },
        ]
    }

    # Run graph execution
    print("Running graph ...")

    result = graph.run(initial_state)
    print(result)

    # Save the result to the database
    print("Saving the result to the database...")
    for message in result["messages"]:
        if "id" not in message.keys():
            print("Store message to the database", message)

    print("\nDone!")


if __name__ == "__main__":
    main()
