from __future__ import annotations

from ..config import Settings
from ..models import SearchHit


class LLMProvider:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def answer(
        self, prompt: str, hits: list[SearchHit] | None = None
    ) -> tuple[str, dict[str, object]]:
        if self.settings.openai_api_key:
            return self._openai(prompt)
        if self.settings.anthropic_api_key:
            return self._anthropic(prompt)
        return self._extractive(hits), {"backend": "extractive"}

    def _extractive(self, hits: list[SearchHit] | None) -> str:
        if not hits:
            return (
                "No LLM provider is configured. Retrieved evidence is available, "
                "but generation is disabled."
            )
        snippet = hits[0].chunk.text[:400]
        return f"{snippet} [E1]"

    def _openai(self, prompt: str) -> tuple[str, dict[str, object]]:
        from openai import OpenAI

        client = OpenAI(api_key=self.settings.openai_api_key)
        response = client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        usage = response.usage.model_dump() if response.usage else {}
        return response.choices[0].message.content or "", usage

    def _anthropic(self, prompt: str) -> tuple[str, dict[str, object]]:
        import anthropic

        client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        message = client.messages.create(
            model=self.settings.anthropic_model,
            max_tokens=1200,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "\n".join(getattr(block, "text", "") for block in message.content)
        usage = {
            "input_tokens": getattr(message.usage, "input_tokens", 0),
            "output_tokens": getattr(message.usage, "output_tokens", 0),
        }
        return text, usage
