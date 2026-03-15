"""
Agent API Client — unified LLM interface for Phase 2 agents.

Supports two providers:
  - "anthropic" : Anthropic SDK (claude-opus-4-6, etc.)
  - "openai"    : OpenAI SDK or any OpenAI-compatible endpoint
                  (OpenAI, Azure, DeepSeek, Qwen, local vLLM, etc.)

All agents use the same tool spec format and the same run() interface
regardless of which provider is configured.

Configuration via phase2/config.json:
    {
        "agent_llm": {
            "provider": "anthropic",
            "model": "claude-opus-4-6",
            "api_key": "sk-ant-...",
            "base_url": null
        }
    }

Or pass parameters directly to AgentClient().
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent.parent / "config.json"


def load_agent_llm_config() -> dict:
    """
    Load agent LLM config from phase2/config.json.
    Falls back to environment variables if config file is absent.
    """
    if _CONFIG_PATH.exists():
        with _CONFIG_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("agent_llm", {})

    # Env-variable fallback
    provider = os.environ.get("AGENT_LLM_PROVIDER", "anthropic")
    if provider == "anthropic":
        return {
            "provider": "anthropic",
            "model": os.environ.get("AGENT_LLM_MODEL", "claude-opus-4-6"),
            "api_key": os.environ.get("ANTHROPIC_API_KEY"),
        }
    else:
        return {
            "provider": "openai",
            "model": os.environ.get("AGENT_LLM_MODEL", "gpt-4o"),
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "base_url": os.environ.get("AGENT_LLM_BASE_URL"),
        }


# ---------------------------------------------------------------------------
# Unified tool spec
# ---------------------------------------------------------------------------

def make_tool_spec(name: str, description: str, parameters: dict) -> dict:
    """
    Define a tool in provider-agnostic format.

    parameters follows JSON Schema:
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "..."}
        },
        "required": ["query"]
    }
    """
    return {
        "name": name,
        "description": description,
        "parameters": parameters,   # our unified key (not input_schema, not function.parameters)
    }


# ---------------------------------------------------------------------------
# Format converters
# ---------------------------------------------------------------------------

def _to_anthropic_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["parameters"],
        }
        for t in tools
    ]


def _to_openai_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in tools
    ]


# ---------------------------------------------------------------------------
# AgentClient
# ---------------------------------------------------------------------------

class AgentClient:
    """
    Unified agent LLM client with tool-use loop.

    Instantiate once per agent; call run() for each conversation.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        cfg = load_agent_llm_config()

        self.provider = (provider or cfg.get("provider", "anthropic")).lower()
        self.model = model or cfg.get("model", "claude-opus-4-6")
        self.api_key = api_key or cfg.get("api_key")
        self.base_url = base_url or cfg.get("base_url")

        self._client = self._build_client()
        logger.info(f"AgentClient: provider={self.provider}, model={self.model}")

    def _build_client(self) -> Any:
        if self.provider == "anthropic":
            import anthropic
            return anthropic.Anthropic(api_key=self.api_key)
        else:
            # openai or any compatible endpoint
            import openai
            kwargs: dict[str, Any] = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            return openai.OpenAI(**kwargs)

    # ------------------------------------------------------------------ #
    # Main interface                                                      #
    # ------------------------------------------------------------------ #

    def run(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_executor: Callable[[str, dict], str] | None = None,
        max_iterations: int = 30,
        temperature: float = 1.0,
        max_rate_limit_retries: int | None = None,
    ) -> str:
        """
        Run a conversation with optional tool use until a text response is returned.

        tools                   : list of make_tool_spec(...)
        tool_executor           : callable(tool_name, tool_input_dict) -> str
        max_rate_limit_retries  : override default retry count on 429 errors.
                                  Set to 1 to fail fast (useful when a fallback is available).
        """
        if self.provider == "anthropic":
            return self._run_anthropic(
                system, messages, tools, tool_executor, max_iterations, temperature
            )
        else:
            return self._run_openai(
                system, messages, tools, tool_executor, max_iterations, temperature,
                max_rate_limit_retries=max_rate_limit_retries,
            )

    # ------------------------------------------------------------------ #
    # Anthropic backend                                                  #
    # ------------------------------------------------------------------ #

    def _run_anthropic(self, system, messages, tools, tool_executor, max_iterations, temperature):
        import anthropic as _anthropic

        messages = list(messages)
        ant_tools = _to_anthropic_tools(tools) if tools else None

        for _ in range(max_iterations):
            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": 8096,
                "system": system,
                "messages": messages,
                "temperature": temperature,
            }
            if ant_tools:
                kwargs["tools"] = ant_tools

            response = self._call_with_retry_anthropic(kwargs)

            if response.stop_reason == "end_turn":
                return self._extract_text_anthropic(response)

            if response.stop_reason == "tool_use":
                if tool_executor is None:
                    raise ValueError("Claude requested tool use but no tool_executor provided.")

                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        logger.debug(f"[anthropic] tool call: {block.name}({block.input})")
                        try:
                            result = tool_executor(block.name, block.input)
                        except Exception as e:
                            result = f"ERROR: {e}"
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result),
                        })

                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
                continue

            logger.warning(f"[anthropic] unexpected stop_reason: {response.stop_reason}")
            return self._extract_text_anthropic(response)

        raise RuntimeError(f"AgentClient (anthropic): exceeded max_iterations={max_iterations}")

    def _call_with_retry_anthropic(self, kwargs, max_retries=3):
        import anthropic as _anthropic
        for attempt in range(max_retries):
            try:
                return self._client.messages.create(**kwargs)
            except _anthropic.RateLimitError:
                wait = 30 * (attempt + 1)
                logger.warning(f"[anthropic] rate limited, waiting {wait}s")
                time.sleep(wait)
            except _anthropic.APIStatusError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"[anthropic] API error {e.status_code}, retrying...")
                    time.sleep(5)
                else:
                    raise
        raise RuntimeError("AgentClient (anthropic): all retries exhausted")

    @staticmethod
    def _extract_text_anthropic(response) -> str:
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""

    # ------------------------------------------------------------------ #
    # OpenAI backend                                                     #
    # ------------------------------------------------------------------ #

    def _run_openai(self, system, messages, tools, tool_executor, max_iterations, temperature,
                    max_rate_limit_retries: int | None = None):
        import openai as _openai

        # OpenAI uses system as a message, not a separate param
        oai_messages = [{"role": "system", "content": system}] + list(messages)
        oai_tools = _to_openai_tools(tools) if tools else None

        for _ in range(max_iterations):
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": oai_messages,
                "temperature": temperature,
                "max_tokens": 8096,
            }
            if oai_tools:
                kwargs["tools"] = oai_tools

            response = self._call_with_retry_openai(kwargs, max_rate_limit_retries=max_rate_limit_retries)
            choice = response.choices[0]
            message = choice.message

            if choice.finish_reason == "stop":
                return message.content or ""

            if choice.finish_reason == "tool_calls":
                if tool_executor is None:
                    raise ValueError("Model requested tool calls but no tool_executor provided.")

                tool_results = []
                for tc in message.tool_calls:
                    name = tc.function.name
                    inputs = json.loads(tc.function.arguments)
                    logger.debug(f"[openai] tool call: {name}({inputs})")
                    try:
                        result = tool_executor(name, inputs)
                    except Exception as e:
                        result = f"ERROR: {e}"
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(result),
                    })

                # Append assistant message (with tool_calls) + tool results
                # Convert to dict to avoid pydantic serialization issues
                oai_messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                })
                oai_messages.extend(tool_results)
                continue

            logger.warning(f"[openai] unexpected finish_reason: {choice.finish_reason}")
            return message.content or ""

        raise RuntimeError(f"AgentClient (openai): exceeded max_iterations={max_iterations}")

    def _call_with_retry_openai(self, kwargs, max_retries=6, max_rate_limit_retries: int | None = None):
        import openai as _openai
        rl_retries = max_rate_limit_retries if max_rate_limit_retries is not None else max_retries
        rl_attempt = 0
        for attempt in range(max_retries):
            try:
                return self._client.chat.completions.create(**kwargs)
            except _openai.RateLimitError:
                rl_attempt += 1
                if rl_attempt >= rl_retries:
                    raise
                wait = 30 * (2 ** (rl_attempt - 1))
                wait = min(wait, 300)
                logger.warning(f"[openai] rate limited, waiting {wait}s (attempt {rl_attempt}/{rl_retries})")
                time.sleep(wait)
            except _openai.APIStatusError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"[openai] API error {e.status_code}, retrying...")
                    time.sleep(5)
                else:
                    raise
        raise RuntimeError("AgentClient (openai): all retries exhausted")


# ---------------------------------------------------------------------------
# Convenience: simple one-shot call (no tools)
# ---------------------------------------------------------------------------

def ask(prompt: str, system: str = "") -> str:
    """Single-turn, no-tool call. Uses config.json settings."""
    client = AgentClient()
    return client.run(
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
