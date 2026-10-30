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


class QuotaExhausted(TeacherError):
    """The daily free-model allowance is spent.

    Separate from TeacherError because the correct response is different.
    A 429 is counted against the allowance before it is rejected, so once
    this fires every further attempt - a retry, or the next model in a
    fallback chain - spends a request that cannot possibly succeed. It must
    propagate past the retry loop, past the fallback chain, and out of the
    verifier that otherwise swallows everything.
    """


# OpenRouter reports why it refused in the response body. Two 429s mean
# opposite things: the account's own daily cap is spent (stop, come back
# tomorrow) versus the shared :free pool for one model is saturated by other
# users (try a different vendor, which the fallback chain does).
_QUOTA_MARKERS = (
    "free-models-per-day",
    "free_models_per_day",
    "daily limit",
    "rate limit exceeded for free",
    "add credits",
    "requests per day",
)


def classify(code: int, body: bytes) -> tuple[str, str]:
    """(kind, human message) for an HTTP error, read from the body.

    kind is "quota", "upstream" or "other". The body is the only place the
    distinction is recorded, which is why it must be read before the
    exception is discarded.
    """
    text = (body or b"").decode("utf-8", "replace")
    detail = text
    try:
        payload = json.loads(text)
        err = payload.get("error") or {}
        if isinstance(err, dict):
            detail = str(err.get("message") or text)
            meta = err.get("metadata") or {}
            if isinstance(meta, dict) and meta.get("limit_source"):
                detail = f"{detail} [limit_source={meta['limit_source']}]"
    except (json.JSONDecodeError, AttributeError):
        pass

    low = detail.lower()
    if code == 429:
        if any(m in low for m in _QUOTA_MARKERS):
            return "quota", detail
        return "upstream", detail
    return "other", detail


def key_status(cfg: ApiConfig, api_key: str) -> dict[str, Any] | None:
    """The account's current free-model allowance, or None if unreachable.

    One cheap GET before a run beats discovering the cap is spent twenty
    counted requests into it. This endpoint is not itself a model call and
    does not draw on the allowance.
    """
    url = cfg.base_url.split("/chat/completions")[0] + "/auth/key"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return (json.load(resp) or {}).get("data") or {}
    except Exception:  # noqa: BLE001 - a preflight that fails must not stop the run
        return None


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
        self.fallback_used = 0
        self.rate_limited = 0
        self.quota_hit = False
        self.last_detail = ""

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

        # A 429 gets its own, smaller budget. Every attempt is counted
        # against the daily allowance whether or not it returns anything, so
        # four attempts into a saturated pool spends four requests to learn
        # what the first one already said.
        last: Exception | None = None
        for attempt in range(max(1, self.cfg.max_retries)):
            self.limiter.wait()
            try:
                req = urllib.request.Request(self.cfg.base_url, data=body, headers=headers)
                with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                    payload = json.load(resp)
            except urllib.error.HTTPError as exc:
                # Read the body before anything else. It is readable once,
                # and it carries the only statement of why the request was
                # refused; discarding it leaves "HTTP Error 429: Too Many
                # Requests", which does not distinguish a spent allowance
                # from a busy upstream pool.
                kind, detail = classify(exc.code, exc.read())
                self.last_detail = detail
                last = TeacherError(f"HTTP {exc.code}: {detail[:300]}")

                if kind == "quota":
                    self.quota_hit = True
                    raise QuotaExhausted(detail[:300]) from exc
                if exc.code not in self.cfg.retry_status:
                    raise TeacherError(f"HTTP {exc.code}: {detail[:300]}") from exc
                if exc.code == 429:
                    self.rate_limited += 1
                    if attempt + 1 >= max(1, self.cfg.max_retries_rate_limited):
                        break
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

    def complete_any(self, model: ModelConfig, prompt: str) -> tuple[str, str]:
        """Try model.id, then each fallback. Returns (reply, model_used).

        Raises only when every model in the chain fails, which is the only
        case where there is nothing left to try.
        """
        import dataclasses

        chain = [model.id, *model.fallbacks]
        last: Exception | None = None
        for model_id in chain:
            if not model_id:
                continue
            try:
                return self.complete(dataclasses.replace(model, id=model_id), prompt), model_id
            except QuotaExhausted:
                # Not this model's fault and not fixable by trying another:
                # the allowance is account-wide. Every further vendor in the
                # chain would spend one more counted request to be told the
                # same thing.
                raise
            except TeacherError as exc:
                last = exc
                self.fallback_used += 1
                continue
        raise TeacherError(f"all {len(chain)} model(s) failed, last: {last}")

    def _backoff(self, attempt: int) -> None:
        time.sleep(min(self.cfg.backoff_max_s, self.cfg.backoff_base_s * 2**attempt))

    def stats(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "failures": self.failures,
            "fallbacks_tried": self.fallback_used,
            "rate_limited": self.rate_limited,
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
