from openai import OpenAI
from typing import List, Dict, Any, Generator, Optional
from dataclasses import dataclass


@dataclass
class SciBoxClient:
    api_key: str
    base_url: str = "https://llm.t1v.scibox.tech/v1"

    def __post_init__(self):
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    # ---------- 1. Список моделей ----------
    def list_models(self) -> Dict[str, Any]:
        return self._client.models.list()

    # ---------- 2. Чат-комплишн ----------
    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **params,
    ) -> Any:
        if stream:
            return self._chat_stream(model, messages, **params)

        resp = self._client.chat.completions.create(
            model=model,
            messages=messages,
            **params,
        )
        return resp

    def _chat_stream(
        self, model: str, messages: List[Dict[str, str]], **params
    ) -> Generator[str, None, None]:
        """Потоковая генерация (SSE)."""
        with self._client.chat.completions.stream(
            model=model,
            messages=messages,
            **params,
        ) as stream:
            for event in stream:
                if event.type == "message.delta" and event.delta.get("content"):
                    yield event.delta["content"]
                elif event.type == "message.completed":
                    break

    # ---------- 3. Эмбеддинги ----------
    def embeddings(self, inputs: Any, model: str = "bge-m3"):
        return self._client.embeddings.create(model=model, input=inputs)
