import os
import re

from openai import OpenAI

from ..config import LLMConfig

# qwen3-family models may emit <think>...</think> blocks through the
# OpenAI-compatible endpoint; the loop and CLI must never see them.
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        # Real endpoints (e.g. api.openai.com) need OPENAI_API_KEY from the
        # environment; Ollama ignores the key, so any placeholder works there.
        api_key = os.environ.get("OPENAI_API_KEY") or "ollama"
        self._client = OpenAI(base_url=config.base_url, api_key=api_key)
        self.last_usage: dict | None = None  # token usage of the latest call

    def chat(self, messages: list[dict], tools: list[dict]):
        """One completion turn. Returns the assistant message object
        (attributes: .content, .tool_calls)."""
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            tools=tools,
            temperature=self.config.temperature,
        )
        self.last_usage = None
        if response.usage:
            self.last_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }
        message = response.choices[0].message
        if message.content:
            message.content = _THINK_RE.sub("", message.content).strip()
        return message
