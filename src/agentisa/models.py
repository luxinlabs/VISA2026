"""Chat client for OpenAI, Ollama (OpenAI-compatible endpoint) and Anthropic, with retries and usage accounting.
The harness speaks OpenAI chat format; the Anthropic branch translates messages and tools at the boundary."""
import json, os, time
from concurrent.futures import ThreadPoolExecutor

_WATCHDOG = ThreadPoolExecutor(max_workers=64, thread_name_prefix="api")


class ChatClient:
    def __init__(self, provider, model, temperature=0.0):
        self.provider, self.model, self.temperature = provider, model, temperature
        if provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(timeout=120, max_retries=0)
            return
        from openai import OpenAI
        if provider == "openai":
            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=120, max_retries=0)
        elif provider == "openrouter":  # open-weights models (Addendum 6); OpenAI-compatible endpoint
            self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], timeout=120, max_retries=0)
        elif provider == "ollama":
            self.client = OpenAI(base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                                 api_key="ollama", timeout=600, max_retries=0)
        else:
            raise ValueError(provider)

    def chat(self, messages, tools=None, max_tokens=2048, seed=0, tool_choice=None):
        if self.provider == "anthropic":
            return self._chat_anthropic(messages, tools, max_tokens, tool_choice)
        kw = dict(model=self.model, messages=messages, temperature=self.temperature, seed=seed, max_tokens=max_tokens)
        if self.provider == "openrouter":  # pin one host so every call runs the same deployment (no silent fallback)
            kw["extra_body"] = {"provider": {"order": [os.environ.get("AGENTISA_OR_HOST", "DeepInfra")], "allow_fallbacks": False}}
        if self.model.startswith("gpt-5"):
            # reasoning models (Addendum 6): no sampling parameters; the token cap covers reasoning + visible output;
            # lowest reasoning effort, so the model is as close as it allows to a non-reasoning generator
            kw = dict(model=self.model, messages=messages, seed=seed, max_completion_tokens=max_tokens,
                      reasoning_effort=os.environ.get("AGENTISA_REASONING_EFFORT", "none"))
        if tools:
            kw["tools"] = tools
            if tool_choice:
                kw["tool_choice"] = tool_choice
        for attempt in range(8):
            try:
                # hard watchdog: a half-dead TCP connection can block an SSL read far beyond the client timeout
                fut = _WATCHDOG.submit(self.client.chat.completions.create, **kw)
                r = fut.result(timeout=150)
                break
            except Exception as e:  # rate limits / transient network errors
                if attempt == 7 or isinstance(e, (TypeError, ValueError, KeyError)) or type(e).__name__ in ("AuthenticationError", "PermissionDeniedError", "NotFoundError", "BadRequestError"):
                    raise
                time.sleep(min(60, 2 ** attempt + 1))
        return self._parse_openai(r)

    def _call(self, fn, **kw):
        for attempt in range(8):
            try:
                # hard watchdog: a half-dead TCP connection can block an SSL read far beyond the client timeout
                return _WATCHDOG.submit(fn, **kw).result(timeout=150)
            except Exception as e:  # rate limits / transient network errors
                if attempt == 7 or isinstance(e, (TypeError, ValueError, KeyError)) or type(e).__name__ in ("AuthenticationError", "PermissionDeniedError", "NotFoundError", "BadRequestError"):
                    raise
                time.sleep(min(60, 2 ** attempt + 1))

    def _chat_anthropic(self, messages, tools, max_tokens, tool_choice):
        # Claude Haiku 5.5: sampling parameters must stay at their defaults (temperature is not sent);
        # thinking is disabled so the model, like GPT-4.1, answers without a reasoning phase; no seed parameter.
        # Claude Haiku 4.5 (Addendum 6): temperature 0 as for GPT-4.1; no thinking parameter (off by default).
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        conv = []
        for m in messages:
            if m["role"] == "system":
                continue
            if m["role"] == "assistant":
                conv.append({"role": "assistant", "content": m.get("anthropic_content") or [{"type": "text", "text": m["content"] or "(empty)"}]})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if conv and conv[-1]["role"] == "user" and isinstance(conv[-1]["content"], list):
                    conv[-1]["content"].append(block)  # all results of one assistant turn go in a single user message
                else:
                    conv.append({"role": "user", "content": [block]})
            else:
                conv.append({"role": "user", "content": m["content"]})
        kw = dict(model=self.model, max_tokens=max_tokens, system=system, messages=conv)
        if "haiku-4-5" in self.model:
            kw["extra_body"] = {"temperature": self.temperature}  # SDK 1.x dropped the argument; the model honours it
        else:
            kw["thinking"] = {"type": "disabled"}
        if tools:
            kw["tools"] = [{"name": t["function"]["name"], "description": t["function"]["description"],
                            "input_schema": t["function"]["parameters"]} for t in tools]
            if tool_choice == "required":
                kw["tool_choice"] = {"type": "any"}
        r = self._call(self.client.messages.create, **kw)
        text = "".join(b.text for b in r.content if b.type == "text")
        tcs = [{"id": b.id, "name": b.name, "arguments": json.dumps(b.input)} for b in r.content if b.type == "tool_use"]
        am = {"role": "assistant", "content": text, "anthropic_content": [b.model_dump(exclude_none=True) for b in r.content]}
        if tcs:
            am["tool_calls"] = [{"id": t["id"], "type": "function", "function": {"name": t["name"], "arguments": t["arguments"]}} for t in tcs]
        u = r.usage
        cached = (u.cache_read_input_tokens or 0)
        return {"text": text, "tool_calls": tcs, "assistant_message": am,
                "input_tokens": u.input_tokens + cached + (u.cache_creation_input_tokens or 0), "cached_input_tokens": cached,
                "output_tokens": u.output_tokens, "finish_reason": r.stop_reason, "system_fingerprint": r.model}

    def _parse_openai(self, r):
        m = r.choices[0].message
        tcs = [{"id": t.id, "name": t.function.name, "arguments": t.function.arguments} for t in (m.tool_calls or [])]
        am = {"role": "assistant", "content": m.content or ""}
        if tcs:
            am["tool_calls"] = [{"id": t["id"], "type": "function",
                                 "function": {"name": t["name"], "arguments": t["arguments"]}} for t in tcs]
        u = r.usage
        return {"text": m.content, "tool_calls": tcs, "assistant_message": am,
                "input_tokens": getattr(u, "prompt_tokens", 0) or 0, "output_tokens": getattr(u, "completion_tokens", 0) or 0,  # includes reasoning tokens
                "cached_input_tokens": getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0,
                "finish_reason": r.choices[0].finish_reason,
                # OpenRouter: the hosting provider that served the call (logged for reproducibility)
                "system_fingerprint": getattr(r, "provider", None) or getattr(r, "system_fingerprint", None)}
