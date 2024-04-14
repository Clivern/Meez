.. image:: https://images.unsplash.com/photo-1650999344693-b76317f7b9ae
  :width: 700
  :alt: Meez - AI Assistant and Copilot SDK for SaaS Applications

|

.. image:: https://img.shields.io/pypi/v/Meez.svg
    :alt: PyPI-Server
    :target: https://pypi.org/project/Meez/
.. image:: https://github.com/Clivern/Veee/actions/workflows/ci.yml/badge.svg
    :alt: Build Status
    :target: https://github.com/Clivern/Meez/actions/workflows/ci.yml
.. image:: https://img.shields.io/pypi/l/Meez.svg
    :alt: License
    :target: https://pypi.org/project/Meez/
.. image:: https://static.pepy.tech/badge/meez
    :alt: PyPI Downloads
    :target: https://pepy.tech/projects/meez

|

=====
Meez
=====

**AI Assistant and Copilot SDK for SaaS Applications**

Meez is a powerful Python library that provides AI-powered assistance capabilities for SaaS applications. It offers intention detection, contextual responses, and workflow automation using LangChain and LangGraph.

Features
--------

* **Intention Detection**: Automatically detect user intentions from natural language input
* **Contextual Responses**: Generate intelligent responses based on provided context data
* **Workflow Automation**: Create complex AI workflows using LangGraph
* **Multiple Data Sources**: Support for text, JSON, and file-based data sources
* **OpenAI Integration**: Built on top of OpenAI's GPT models for reliable AI responses
* **Flexible Architecture**: Modular design for easy integration and customization

Installation
------------

Install Meez using pip:

.. code-block:: bash

    pip install Meez

Quick Start
-----------

1. **Set up your OpenAI API key**:

.. code-block:: python

    import os
    os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"

2. **Basic intention detection**:

.. code-block:: python

    from meez.core import Langchain, Intention

    # Initialize LangChain
    langchain = Langchain(openai_api_key="your-api-key")

    # Define possible intentions
    intentions = ["greeting", "weather", "help", "goodbye", "unknown"]

    # Create intention detector
    detector = Intention(langchain=langchain, intentions=intentions)

    # Detect intention
    user_input = "Hello there!"
    intention = detector.detect(user_input)
    print(f"Detected intention: {intention}")

3. **Contextual responses**:

.. code-block:: python

    from meez.core import Langchain, Respond
    from meez.data import TextReader

    # Initialize components
    langchain = Langchain(openai_api_key="your-api-key")
    respond = Respond(langchain)

    # Create data source
    context_data = "Python is a high-level programming language..."
    data_reader = TextReader(context_data)

    # Get contextual response
    question = "What is Python?"
    response = respond.run(question=question, data=data_reader)
    print(f"Response: {response}")

Usage Examples
--------------

Intention Detection
^^^^^^^^^^^^^^^^^^^^

Detect user intentions from natural language:

.. code-block:: python

    import os
    from meez.core import Langchain, Intention

    # Setup
    api_key = os.getenv("OPENAI_API_KEY")
    langchain = Langchain(openai_api_key=api_key)

    # Define intentions
    intentions = [
        "greeting",
        "weather",
        "joke",
        "help",
        "goodbye",
        "book_appointment",
        "cancel_appointment",
        "unknown"
    ]

    # Create detector
    detector = Intention(langchain=langchain, intentions=intentions)

    # Test inputs
    test_texts = [
        "Hello there!",
        "What's the weather like?",
        "I need to book an appointment",
        "Can you help me cancel my appointment?"
    ]

    for text in test_texts:
        intention = detector.detect(text)
        print(f"'{text}' → {intention}")

Contextual Responses
^^^^^^^^^^^^^^^^^^^^^

Generate responses based on context data:

.. code-block:: python

    from meez.core import Langchain, Respond
    from meez.data import TextReader, JsonReader, FileReader

    # Initialize
    langchain = Langchain(openai_api_key=api_key)
    respond = Respond(langchain)

    # Using text data
    text_data = "Artificial Intelligence is a branch of computer science..."
    text_reader = TextReader(text_data)

    response = respond.run(
        question="What is AI?",
        data=text_reader
    )

    # Using JSON data
    json_data = {"company": {"name": "TechCorp", "employees": 250}}
    json_reader = JsonReader(json_data)

    response = respond.run(
        question="How many employees does the company have?",
        data=json_reader
    )

    # Using file data
    file_reader = FileReader("documentation.txt")
    response = respond.run(
        question="What are the main features?",
        data=file_reader
    )

Workflow Automation with LangGraph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create complex AI workflows:

.. code-block:: python

    from meez.core.langgraph import LangGraph, MainState
    from meez.core import Langchain, Intention

    # Define workflow nodes
    def get_intent(state: MainState) -> MainState:
        # Detect user intention
        detector = Intention(langchain, ["get_phone", "get_email", "unknown"])
        intent = detector.detect(state["messages"][-1]["content"])
        state["messages"].append({"role": "assistant", "content": intent, "internal": True})
        return state

    def decide(state: MainState) -> str:
        # Return the detected intent to determine next step
        return state["messages"][-1]["content"]

    def get_phone(state: MainState) -> MainState:
        state["messages"].append({"role": "assistant", "content": "Phone: +1234567890"})
        return state

    def get_email(state: MainState) -> MainState:
        state["messages"].append({"role": "assistant", "content": "Email: support@company.com"})
        return state

    def unknown(state: MainState) -> MainState:
        state["messages"].append({"role": "assistant", "content": "I'm sorry, I don't know that."})
        return state

    # Create and configure graph
    graph = LangGraph()
    graph.add_node("get_intent", get_intent)
    graph.add_node("decide", decide)
    graph.add_node("get_phone", get_phone)
    graph.add_node("get_email", get_email)
    graph.add_node("unknown", unknown)

    graph.set_entry_point("get_intent")
    graph.add_conditional_edge("get_intent", decide)
    graph.add_finish_point("get_phone")
    graph.add_finish_point("get_email")
    graph.add_finish_point("unknown")

    # Run workflow
    initial_state = {"messages": [{"role": "user", "content": "What's your phone number?"}]}
    result = graph.run(initial_state)

Examples
--------

See the ``examples/`` directory for complete working examples:

* ``intention_detection.py`` - Basic intention detection
* ``respond_to_user.py`` - Contextual responses with different data sources
* ``complex_graph.py`` - Advanced workflow automation
* ``sample_graph.py`` - Simple graph workflow
* ``sample_assistant.py`` - Complete assistant implementation

Development
-----------

Setup Development Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/clivern/meez.git
    cd meez

    # Install development dependencies
    pip install -r requirements.test.txt
    pip install -e .

Running Tests
^^^^^^^^^^^^^

.. code-block:: bash

    # Run tests
    make ci

Support
-------

* Documentation: https://github.com/clivern/meez/
* Issues: https://github.com/clivern/meez/issues
* Email: hello@clivern.com

Changelog
---------

See `CHANGELOG.rst` for a detailed history of changes.
