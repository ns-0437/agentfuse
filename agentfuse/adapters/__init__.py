"""Framework adapters — one circuit-breaker engine, three runtimes.

  * ``guarded_tool_loop``        — plain OpenAI SDK manual tool-use loop
  * :class:`FuseCallbackHandler` — LangGraph / LangChain callback handler
  * :class:`AgentKitBreaker`     — a generic, manually-wired bridge for an
    AgentKit hook API that does not match ``agents.RunHooks`` directly (see
    its own module docstring)

For OpenAI AgentKit specifically, ``FuseRunHooks``
(``agentfuse.adapters.agentkit_hooks``) is the one this project actually
tests against the real SDK and the one README.md recommends — a genuine
``agents.RunHooks`` subclass, not a generic bridge. It is not re-exported
here for a dependency reason, not a ranking (see the comment below).
"""

from .agentkit import AgentKitBreaker
from .openai_sdk import guarded_tool_loop
from .langgraph import FuseCallbackHandler

# `FuseRunHooks` / `BreakerInterrupt` (agentkit_hooks.py) are deliberately NOT
# re-exported here, unlike the three above. That module does `from agents import
# RunHooks` at the top level -- a hard dependency on the optional `openai-agents`
# package -- so importing it here would make `import agentfuse.adapters` require
# openai-agents even for someone only using the LangGraph or plain-SDK adapter,
# breaking the "stdlib-only core, opt-in extras" promise for two of three
# adapters to serve the third. Import it directly:
# `from agentfuse.adapters.agentkit_hooks import FuseRunHooks, BreakerInterrupt`
# (this is already how README.md and examples/real_agentkit_run.py do it).

__all__ = ["AgentKitBreaker", "guarded_tool_loop", "FuseCallbackHandler"]
