"""List prices and trial cost, shared by the runner (budget caps) and every analysis script.

PRICE maps model -> (input, output, cached input) in USD per 1M tokens. Cached input applies only to tokens the
provider reported as served from its prompt cache (logged as `cached_input_tokens` since Addendum 5); older logs
lack the field and are priced entirely at the input rate, which is how their costs were reported."""

PRICE = {
    "gpt-4.1-nano-2025-04-14": (0.10, 0.40, 0.025),
    "gpt-4.1-mini-2025-04-14": (0.40, 1.60, 0.10),
    "gpt-4.1-2025-04-14": (2.00, 8.00, 0.50),
    "gpt-5.4-mini-2026-03-17": (0.75, 4.50, 0.075),
    "claude-haiku-5-5": (0.10, 0.50, 0.10),  # no prompt caching requested for Claude
    "claude-haiku-4-5-20251001": (1.00, 5.00, 1.00),
    "qwen/qwen3-coder": (0.30, 1.00, 0.30),  # OpenRouter list price; the served price can vary by host
}


def trial_cost_usd(log, model=None):
    """List-price cost of one trial log (all model calls, cached input discounted when reported)."""
    pin, pout, pcache = PRICE[model or log["model"]]
    cached = log.get("cached_input_tokens", 0) or 0
    return ((log["input_tokens"] - cached) * pin + cached * pcache + log["output_tokens"] * pout) / 1e6
