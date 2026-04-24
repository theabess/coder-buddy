"""
Coder Buddy — a self-correcting autonomous coding agent.

Accepts a natural language prompt, generates executable Python code,
runs it in an isolated sandbox, and iteratively debugs itself until
the code passes or the retry limit is reached.
"""

from coder_buddy.agent import CoderBuddy
from coder_buddy.config import (
    AgentConfig,
    CoderBuddyError,
    ConfigurationError,
    SandboxUnavailableError,
    LLMUnavailableError,
    ParseError,
    LanguageNotSupportedError,
)
from coder_buddy.models import AgentResponse

__all__ = [
    "CoderBuddy",
    "AgentConfig",
    "AgentResponse",
    "CoderBuddyError",
    "ConfigurationError",
    "SandboxUnavailableError",
    "LLMUnavailableError",
    "ParseError",
    "LanguageNotSupportedError",
]
