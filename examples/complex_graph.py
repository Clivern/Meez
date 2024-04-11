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

import os, json, uuid
from datetime import datetime
from meez.core.langgraph import LangGraph, MainState
from meez.core import Langchain, Intention


def get_messages_from_database():
    """Get messages from the database"""
    try:
        with open("examples/messages.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def store_messages_to_database(messages: list):
    """Store messages to the database"""
    new_messages = []

    for message in messages:
        # Remove any internal graph messages
        if message.get("internal", False):
            continue

        if "id" not in message.keys():
            message["id"] = str(uuid.uuid4())
            message["createdAt"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        new_messages.append(message)

    # Add new messages to the database
    with open("examples/messages.json", "w") as f:
        json.dump(new_messages, f)


def get_text_intention(text: str, intentions: list[str]) -> str:
    """Get the intention of the text"""
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise Exception("OPENAI_API_KEY is not set")

    detector = Intention(
        langchain=Langchain(openai_api_key=api_key), intentions=intentions
    )

    return detector.detect(text)


def get_intent(state: MainState) -> MainState:
    """Get the intention of the text"""
    intent = get_text_intention(
        state.get("messages")[-1].get("content"),
        ["get_company_phone", "get_company_email", "unknown"],
    )

    state["messages"].append({"role": "assistant", "content": intent, "internal": True})

    return state


def decide(state: MainState) -> MainState:
    """Decide the next step"""
    return state.get("messages")[-1].get("content")


def get_company_phone(state: MainState) -> MainState:
    """Get company phone"""
    state["messages"].append(
        {"role": "assistant", "content": "The company phone is +2352553423"}
    )
    return state


def get_company_email(state: MainState) -> MainState:
    """Get company email"""
    state["messages"].append(
        {"role": "assistant", "content": "The company email is support@meez.com"}
    )
    return state


def unknown(state: MainState) -> MainState:
    """Unknown intent"""
    state["messages"].append(
        {"role": "assistant", "content": "I'm sorry, I don't know that."}
    )
    return state


def main():
    """Complex LangGraph example"""

    print("=== Complex LangGraph Example ===\n")

    # Create graph
    graph = LangGraph()

    # Add nodes
    graph.add_node("get_intent", get_intent, "Get intent")
    graph.add_node("decide", decide, "Decide the next step")
    graph.add_node("get_company_phone", get_company_phone, "Get company phone")
    graph.add_node("get_company_email", get_company_email, "Get company email")
    graph.add_node("unknown", unknown, "Unknown intent")

    # Set entry point
    graph.set_entry_point("get_intent")

    # Add edges
    graph.add_conditional_edge("get_intent", decide)

    # Add finish points
    graph.add_finish_point("get_company_phone")
    graph.add_finish_point("get_company_email")
    graph.add_finish_point("unknown")

    # Get user input
    user_input = input("Enter your message: ")

    # Initial state
    initial_state = get_messages_from_database()
    initial_state.append({"role": "user", "content": user_input})

    if not graph.validate_graph():
        raise Exception("Graph is not valid")

    # Run graph execution
    print("Running graph ...")

    result = graph.run({"messages": initial_state})

    # Save the result to the database
    print("Saving the result to the database...")
    store_messages_to_database(result.get("messages", []))

    print("\nDone!")


if __name__ == "__main__":
    main()
