from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from time import perf_counter
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError

from efficient_rlm.llm.base import LLMClient, LLMResponse


@dataclass(frozen=True)
class HTTPClientConfig:
    provider: str
    endpoint: str
    model: str
    api_key: str | None = None
    temperature: float = 0.2
    max_tokens: int = 800
    timeout_seconds: float = 60.0
    max_retries: int = 1


class HTTPGenerationClient(LLMClient):
    def __init__(self, config: HTTPClientConfig) -> None:
        self.config = config

    def generate(self, prompt: str) -> str:
        return self.generate_response(prompt).text

    def generate_response(self, prompt: str) -> LLMResponse:
        payload = self._build_payload(prompt)
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                started = perf_counter()
                req = request.Request(self.config.endpoint, data=body, headers=headers, method="POST")
                with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                    request_id = response.headers.get("x-request-id")
                    data = json.loads(response.read().decode("utf-8"))
                response = self._extract_response(data, perf_counter() - started, request_id)
                return LLMResponse(**{**response.__dict__, "retries": attempt})
            except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
                last_error = exc
                if attempt < self.config.max_retries:
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"LLM request failed after retries: {last_error}")

    def _build_payload(self, prompt: str) -> dict[str, Any]:
        if self.config.provider == "ollama":
            return {
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            }
        return {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }

    def _extract_response(self, data: dict[str, Any], latency: float, request_id: str | None) -> LLMResponse:
        if self.config.provider == "ollama":
            text = str(data.get("response", "")).strip()
            prompt_tokens = data.get("prompt_eval_count")
            completion_tokens = data.get("eval_count")
            total_tokens = None
            if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
                total_tokens = prompt_tokens + completion_tokens
            return LLMResponse(
                text=text,
                model=self.config.model,
                latency_seconds=latency,
                prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
                completion_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
                total_tokens=total_tokens,
                finish_reason="done" if data.get("done") else None,
                request_id=request_id,
                provider=self.config.provider,
                raw=data,
            )
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except Exception as exc:
            raise RuntimeError(f"Could not parse model response: {data}") from exc
        usage = data.get("usage", {})
        return LLMResponse(
            text=str(content).strip(),
            model=data.get("model") or self.config.model,
            latency_seconds=latency,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            finish_reason=choice.get("finish_reason"),
            request_id=request_id or data.get("id"),
            provider=self.config.provider,
            raw=data,
        )


def build_llm_client(config) -> LLMClient:
    from efficient_rlm.llm.mock import MockLLMClient

    if config.provider == "mock":
        return MockLLMClient()
    endpoint = config.endpoint
    if not endpoint:
        if config.provider == "ollama":
            endpoint = "http://localhost:11434/api/generate"
        else:
            raise ValueError("endpoint is required for openai_compatible provider")
    api_key = os.getenv(config.api_key_env)
    if config.provider == "openai_compatible" and not api_key:
        raise ValueError(f"{config.api_key_env} is required for openai_compatible provider")
    return HTTPGenerationClient(
        HTTPClientConfig(
            provider=config.provider,
            endpoint=endpoint,
            model=config.model,
            api_key=api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
        )
    )
