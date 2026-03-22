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
            "api_keys": ["sk-ant-1", "sk-ant-2"],
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
        cfg = data.get("agent_llm", {})
        # Default request timeout keeps calls from hanging indefinitely
        # and enables deterministic retry behavior on unstable links.
        cfg.setdefault("timeout", 120)
        return cfg

    # Env-variable fallback
    provider = os.environ.get("AGENT_LLM_PROVIDER", "anthropic")
    if provider == "anthropic":
        return {
            "provider": "anthropic",
            "model": os.environ.get("AGENT_LLM_MODEL", "claude-opus-4-6"),
            "api_key": os.environ.get("ANTHROPIC_API_KEY"),
            "api_keys": _parse_env_api_keys("ANTHROPIC_API_KEYS"),
            "timeout": int(os.environ.get("AGENT_LLM_TIMEOUT", "120")),
        }
    else:
        return {
            "provider": "openai",
            "model": os.environ.get("AGENT_LLM_MODEL", "gpt-4o"),
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "api_keys": _parse_env_api_keys("OPENAI_API_KEYS"),
            "base_url": os.environ.get("AGENT_LLM_BASE_URL"),
            "timeout": int(os.environ.get("AGENT_LLM_TIMEOUT", "120")),
        }


def _parse_env_api_keys(env_name: str) -> list[str]:
    raw = os.environ.get(env_name, "")
    if not raw.strip():
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]


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
        api_keys: list[str] | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ):
        cfg = load_agent_llm_config()

        self.provider = (provider or cfg.get("provider", "anthropic")).lower()
        self.model = model or cfg.get("model", "claude-opus-4-6")
        self.base_url = base_url or cfg.get("base_url")
        self.timeout = timeout if timeout is not None else float(cfg.get("timeout", 120))

        cfg_keys = cfg.get("api_keys")
        key_pool = api_keys if api_keys is not None else cfg_keys
        self.api_keys = self._normalize_api_keys(primary=api_key or cfg.get("api_key"), pool=key_pool)
        self._clients = [self._build_client(key) for key in self.api_keys]
        self._client_idx = 0
        logger.info(
            f"AgentClient: provider={self.provider}, model={self.model}, key_pool_size={len(self.api_keys)}, timeout={self.timeout}s"
        )

    @staticmethod
    def _normalize_api_keys(primary: str | None, pool: Any) -> list[str | None]:
        keys: list[str | None] = []
        seen = set()
        if primary:
            keys.append(primary)
            seen.add(primary)
        if isinstance(pool, list):
            for k in pool:
                if isinstance(k, str) and k and k not in seen:
                    keys.append(k)
                    seen.add(k)
        if not keys:
            return [None]
        return keys

    def _build_client(self, api_key: str | None) -> Any:
        if self.provider == "anthropic":
            import anthropic
            return anthropic.Anthropic(api_key=api_key, timeout=self.timeout)
        else:
            # openai or any compatible endpoint
            import openai
            kwargs: dict[str, Any] = {}
            if api_key:
                kwargs["api_key"] = api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            kwargs["timeout"] = self.timeout
            return openai.OpenAI(**kwargs)

    @property
    def _client(self) -> Any:
        return self._clients[self._client_idx]

    def _rotate_client(self, reason: str) -> bool:
        if len(self._clients) <= 1:
            return False
        prev = self._client_idx
        self._client_idx = (self._client_idx + 1) % len(self._clients)
        logger.warning(
            f"[{self.provider}] rotating API key due to {reason}: slot {prev + 1}/{len(self._clients)} -> {self._client_idx + 1}/{len(self._clients)}"
        )
        return True

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
        rl_attempt = 0
        for attempt in range(max_retries):
            try:
                return self._client.messages.create(**kwargs)
            except (_anthropic.APIConnectionError, _anthropic.APITimeoutError) as e:
                if attempt < max_retries - 1:
                    wait = min(2 * (2 ** attempt), 20)
                    logger.warning(
                        f"[anthropic] connection/timeout error: {type(e).__name__}; retrying in {wait}s"
                    )
                    time.sleep(wait)
                    continue
                raise
            except _anthropic.RateLimitError:
                rl_attempt += 1
                if self._rotate_client("rate_limit"):
                    continue
                wait = min(10 * (2 ** (rl_attempt - 1)), 120)
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
            except (_openai.APIConnectionError, _openai.APITimeoutError) as e:
                if attempt < max_retries - 1:
                    # Connection failures are usually transient; rotate key/client
                    # and back off quickly to improve recovery in batch agent runs.
                    self._rotate_client("connection_error")
                    wait = min(2 * (2 ** attempt), 20)
                    logger.warning(
                        f"[openai] connection/timeout error: {type(e).__name__}; retrying in {wait}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait)
                    continue
                raise
            except _openai.RateLimitError:
                rl_attempt += 1
                if rl_attempt >= rl_retries:
                    raise
                if self._rotate_client("rate_limit"):
                    continue
                wait = 10 * (2 ** (rl_attempt - 1))
                wait = min(wait, 120)
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
