"""OpenRouter chat client.

urllib rather than the openai package: one POST with one JSON body does not
justify a dependency, and the rest of this repository already avoids pulling
in libraries for single call sites.

Everything the client does is driven by ApiConfig. There is no literal URL,
timeout, retry count or status code in this file that the operator cannot
change in rag.yaml.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

from . import dotenv
from .config import ApiConfig, ModelConfig


class TeacherError(RuntimeError):
    """An API failure the caller should stop on rather than retry."""


class RateLimiter:
    """Minimum spacing between calls, so a burst cannot trip a per-minute cap."""

    def __init__(self, requests_per_minute: int) -> None:
        self._interval = 60.0 / max(1, requests_per_minute)
        self._last = 0.0

    def wait(self) -> None:
        delay = self._interval - (time.monotonic() - self._last)
        if delay > 0:
            time.sleep(delay)
        self._last = time.monotonic()


class Client:
    """One client, shared by every model role, so the rate limit is global.

    The free-tier allowance is assumed to be account-wide across models, which
    means the generator and the verifier draw from the same pool. A limiter
    per model would let two roles together exceed what either is allowed.
    """

    def __init__(self, cfg: ApiConfig, api_key: str | None = None) -> None:
        self.cfg = cfg
        self.api_key = api_key or os.environ.get(cfg.api_key_env, "")
        if not self.api_key and cfg.env_file:
            # Not yet in the environment: fall back to the nearest .env.
            found = dotenv.find_upwards(cfg.env_file)
            if found:
                dotenv.load(found)
                self.api_key = os.environ.get(cfg.api_key_env, "")
        if not self.api_key:
            raise TeacherError(
                f"no API key: set {cfg.api_key_env} in the environment or put "
                f"{cfg.api_key_env}=... in {cfg.env_file}. "
                f"Get one at https://openrouter.ai/keys"
            )
        self.limiter = RateLimiter(cfg.requests_per_minute)
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.failures = 0

    def complete(self, model: ModelConfig, prompt: str) -> str:
        """One completion. Raises TeacherError when retrying would be futile."""
        body = json.dumps({
            "model": model.id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": model.temperature,
            "max_tokens": model.max_tokens,
        }).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.cfg.referer,
            "X-Title": self.cfg.title,
        }

        last: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            self.limiter.wait()
            try:
                req = urllib.request.Request(self.cfg.base_url, data=body, headers=headers)
                with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                    payload = json.load(resp)
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code not in self.cfg.retry_status:
                    raise TeacherError(f"HTTP {exc.code}: {exc.read()[:400]!r}") from exc
                self._backoff(attempt)
                continue
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last = exc
                self._backoff(attempt)
                continue

            if "error" in payload and not payload.get("choices"):
                raise TeacherError(str(payload["error"])[:400])
            choices = payload.get("choices") or []
            if not choices:
                last = TeacherError("response carried no choices")
                self._backoff(attempt)
                continue

            usage = payload.get("usage") or {}
            self.calls += 1
            self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.completion_tokens += int(usage.get("completion_tokens") or 0)
            return choices[0]["message"]["content"] or ""

        self.failures += 1
        raise TeacherError(f"giving up after {self.cfg.max_retries} attempts: {last}")

    def _backoff(self, attempt: int) -> None:
        time.sleep(min(self.cfg.backoff_max_s, self.cfg.backoff_base_s * 2**attempt))

    def stats(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "failures": self.failures,
        }


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def extract_json(reply: str) -> dict[str, Any] | None:
    """Pull a JSON object out of a model reply, or None.

    Instruction-tuned models wrap JSON in fences and prepend a sentence of
    preamble however firmly they are told not to, so the outermost
    brace-delimited span is taken rather than the whole reply. A reply that
    still will not parse returns None: one lost page is cheap, and a
    half-parsed object would be a silent defect in the dataset.
    """
    if not reply:
        return None
    for candidate in (reply, _FENCE.sub("", reply)):
        match = _JSON_BLOCK.search(candidate)
        if not match:
            continue
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None
