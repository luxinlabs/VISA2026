"""Chat client for OpenAI and Ollama (OpenAI-compatible endpoint), with retries and usage accounting."""
import os, time


class ChatClient:
    def __init__(self, provider, model, temperature=0.0):
        from openai import OpenAI
        self.provider, self.model, self.temperature = provider, model, temperature
        if provider == "openai":
            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=120, max_retries=0)
        elif provider == "ollama":
            self.client = OpenAI(base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                                 api_key="ollama", timeout=600, max_retries=0)
        else:
            raise ValueError(provider)

    def chat(self, messages, tools=None, max_tokens=2048, seed=0, tool_choice=None):
        kw = dict(model=self.model, messages=messages, temperature=self.temperature, seed=seed, max_tokens=max_tokens)
        if tools:
            kw["tools"] = tools
            if tool_choice:
                kw["tool_choice"] = tool_choice
        for attempt in range(8):
            try:
                r = self.client.chat.completions.create(**kw)
                break
            except Exception as e:  # rate limits / transient network errors
                if attempt == 7:
                    raise
                time.sleep(min(60, 2 ** attempt + 1))
        m = r.choices[0].message
        tcs = [{"id": t.id, "name": t.function.name, "arguments": t.function.arguments} for t in (m.tool_calls or [])]
        am = {"role": "assistant", "content": m.content or ""}
        if tcs:
            am["tool_calls"] = [{"id": t["id"], "type": "function",
                                 "function": {"name": t["name"], "arguments": t["arguments"]}} for t in tcs]
        u = r.usage
        return {"text": m.content, "tool_calls": tcs, "assistant_message": am,
                "input_tokens": getattr(u, "prompt_tokens", 0) or 0, "output_tokens": getattr(u, "completion_tokens", 0) or 0,
                "finish_reason": r.choices[0].finish_reason, "system_fingerprint": getattr(r, "system_fingerprint", None)}
