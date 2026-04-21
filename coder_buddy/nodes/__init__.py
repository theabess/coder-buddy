"""
LangGraph node implementations for Coder Buddy.

Each node is a plain Python function with the signature::

    def node_name(state: AgentState) -> dict:
        ...

LangGraph merges the returned partial dict back into the shared
``AgentState`` automatically.
"""
