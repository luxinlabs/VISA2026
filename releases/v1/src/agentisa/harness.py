"""Trial harness: one fresh conversation per (model, task, condition, trial).

Flow: system + user(task [+ documentation]) -> [tool calls, interactive only] -> program
      -> grade (pinned verifier/XLA + independent oracle) -> up to R repair turns with the diagnostic.
The documentation stays in the conversation during repairs, and the interactive tool stays available
(sharing one tool-call budget), so the repair endpoint is still conditioned on the treatment.
"""
import json, re, time
from dataclasses import dataclass, asdict, field
from .evaluator import run_program
from .interfaces import render, TOOL_SPEC, tool_get_facts, interactive_preamble
from .tasks import expected_outputs, compare

SYSTEM = ("You are an expert compiler engineer. You write programs in StableHLO, the MLIR dialect used by "
          "OpenXLA. Write a complete, valid StableHLO program for the task. Reply with the complete program "
          "in a single ```mlir code block.")


@dataclass
class Evaluation:
    stage: str               # executed | static_error | signature_mismatch | no_main | forbidden_op | compile_error | runtime_error | timeout | crash | no_program
    verified: bool           # parsed + verified + signature matched + compiled
    semantic_correct: bool   # executed and matched the independent oracle on all held-out cases
    oracle_cases: int        # number of oracle cases actually executed and compared
    error_type: str | None = None
    diagnostic: str = ""


def extract_program(text):
    if not text:
        return None
    blocks = re.findall(r"```(?:mlir|MLIR)?[ \t]*\n(.*?)```", text, flags=re.S)
    blocks = [b for b in blocks if "func.func" in b] or blocks
    if blocks:
        return blocks[-1].strip()
    i = text.find("func.func")
    return text[i:].strip() if i >= 0 else None


def grade(program, task, timeout=60.0) -> Evaluation:
    if not program:
        return Evaluation("no_program", False, False, 0, "no_program", "No ```mlir code block with a program was found.")
    res = run_program(program, task["signature"], task["cases"], timeout=timeout)
    st = res["stage"]
    if st != "executed":
        et = res.get("subclass", st)
        return Evaluation(st, False, False, 0, et, res.get("error", ""))
    exp = expected_outputs(task)
    for k, (a, e) in enumerate(zip(res["outputs"], exp)):
        ok, msg = compare(a, e, task)
        if not ok:
            return Evaluation("executed", True, False, k + 1, "semantic_mismatch",
                              f"The program verified and ran, but on a test input it produced {msg}.")
    return Evaluation("executed", True, True, len(exp))


def task_prompt(task, docs):
    ops = ", ".join(("element-wise ops (" + "add/subtract/multiply/divide/maximum/minimum/negate/exponential/sqrt/rsqrt/abs" + ")")
                    if o == "elementwise" else o for o in task["ops"])
    body = (f"Task: write a StableHLO program whose entry function has exactly this signature:\n"
            f"{task['header']}\n\nIt must compute: {task['spec']}\n\n"
            f"Relevant operations: {ops}.\n\nReply with the complete program in a single ```mlir code block.")
    return (f"Reference documentation:\n\n{docs}\n\n---\n\n{body}") if docs else body


def feedback(ev: Evaluation):
    return (f"Your program failed ({ev.error_type}).\n{ev.diagnostic[:1500]}\n\n"
            "Fix the program and reply with the complete corrected program in a single ```mlir code block.")


def run_trial(client, task, condition, budget, trial=0):
    """client: ChatClient. budget: dict(repair_attempts, max_tool_calls, max_model_calls, max_output_tokens)."""
    r = render(task, condition)
    docs = r["content"]
    user = task_prompt(task, docs)
    tools = None
    if condition in ("interactive", "interactive_forced"):
        user = interactive_preamble(r["available_ops"]).format(max_calls=budget["max_tool_calls"]) + "\n\n" + user
        tools = TOOL_SPEC
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    log = dict(task_id=task["task_id"], template=task["template"], family=task["family"], split=task["split"],
               condition=condition, trial=trial, model=client.model, provider=client.provider,
               available_fact_ids=r["fact_ids"], attempts=[], tool_calls=0, retrieved_fact_ids=[],
               tool_queries=[], input_tokens=0, output_tokens=0, model_calls=0, t0=time.time())
    attempts = 0
    while log["model_calls"] < budget["max_model_calls"]:
        use_tools = tools if (tools and log["tool_calls"] < budget["max_tool_calls"]) else None
        if tools and not use_tools and msgs[-1]["role"] == "tool":
            msgs.append({"role": "user", "content": "Tool budget exhausted. Write the program now."})
        force = "required" if (condition == "interactive_forced" and log["model_calls"] == 0) else None
        resp = client.chat(msgs, tools=use_tools, max_tokens=budget["max_output_tokens"], seed=trial, tool_choice=force)
        log["model_calls"] += 1
        log["input_tokens"] += resp["input_tokens"]; log["output_tokens"] += resp["output_tokens"]
        if resp["tool_calls"]:
            msgs.append(resp["assistant_message"])
            for tc in resp["tool_calls"]:
                log["tool_calls"] += 1
                if log["tool_calls"] > budget["max_tool_calls"]:
                    out, ids = {"error": "Tool budget exhausted. Write the program now."}, []
                else:
                    try:
                        a = json.loads(tc["arguments"] or "{}")
                    except json.JSONDecodeError:
                        a = {}
                    out, ids = tool_get_facts(r["available_ops"], a.get("operation", ""), a.get("category", "all"))
                    log["tool_queries"].append([a.get("operation"), a.get("category", "all")])
                    log["retrieved_fact_ids"] += [i for i in ids if i not in log["retrieved_fact_ids"]]
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(out, ensure_ascii=False)})
            continue
        text = resp["text"] or ""
        msgs.append({"role": "assistant", "content": text})
        prog = extract_program(text)
        ev = grade(prog, task)
        log["attempts"].append(dict(asdict(ev), program=prog))
        if ev.semantic_correct or attempts >= budget["repair_attempts"]:
            break
        attempts += 1
        msgs.append({"role": "user", "content": feedback(ev)})
    a = log["attempts"]
    log.update(
        first_attempt_semantic_correct=bool(a and a[0]["semantic_correct"]),
        first_attempt_verified=bool(a and a[0]["verified"]),
        first_attempt_error_type=(a[0]["error_type"] if a else "no_attempt"),
        correct_after_repair=bool(a and a[-1]["semantic_correct"]),
        repair_attempts=max(len(a) - 1, 0),
        n_retrieved_facts=len(log["retrieved_fact_ids"]),
        elapsed_seconds=round(time.time() - log.pop("t0"), 2),
    )
    return log, msgs
