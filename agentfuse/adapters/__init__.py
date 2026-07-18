"""Framework adapters — one circuit-breaker engine, three runtimes.

  * :class:`AgentKitBreaker`     — OpenAI AgentKit (first-class)
  * ``guarded_tool_loop``        — plain OpenAI SDK manual tool-use loop
  * :class:`FuseCallbackHandler` — LangGraph / LangChain callback handler
"""

from .agentkit import AgentKitBreaker
from .openai_sdk import guarded_tool_loop
from .langgraph import FuseCallbackHandler

__all__ = ["AgentKitBreaker", "guarded_tool_loop", "FuseCallbackHandler"]
