"""Track B: closed-loop agent. The model drives its own loop with tools and decides when to submit.

Tools: get_facts (interactive conditions only), verify (parser/verifier/signature/compile diagnostics; no oracle),
run_example (executes on ONE visible example and compares with its expected output), submit (ends the trial).
The submitted program is graded by the independent oracle on all cases; cases 1..2 are never shown to the agent.
"""
import json, time
from .evaluator import run_program
from .harness import extract_program, grade, task_prompt, SYSTEM
from .interfaces import render, TOOL_SPEC, tool_get_facts, interactive_preamble, INTERACTIVE
from .tasks import expected_outputs, compare

AGENT_SYSTEM = ("You are an autonomous compiler agent that writes programs in StableHLO, the MLIR dialect used by OpenXLA. "
                "Use the tools to check your work: `verify` runs the StableHLO parser and verifier, `run_example` executes "
                "your program on one example input and compares it with the expected output. When you are confident, call "
                "`submit` with the complete program. Your submission is graded on hidden test inputs.")

_PROG = {"type": "object", "properties": {"program": {"type": "string", "description": "Complete StableHLO module text."}},
         "required": ["program"]}
AGENT_TOOLS = [
    {"type": "function", "function": {"name": "verify", "description": "Parse and verify a StableHLO program and check its @main signature. Returns OK or the diagnostic.", "parameters": _PROG}},
    {"type": "function", "function": {"name": "run_example", "description": "Verify, compile and run the program on one example input; reports whether the output matches the expected output (and where it differs).", "parameters": _PROG}},
    {"type": "function", "function": {"name": "submit", "description": "Submit the final program. Ends the task.", "parameters": _PROG}},
]


def _static_check(program, task):
    if not program:
        return False, "No program given."
    res = run_program(program, task["signature"], [], timeout=60)
    if res["stage"] == "executed":
        return True, "OK: the program parses, verifies, has the required @main signature and compiles."
    return False, f"{res['stage']}: {res.get('error', '')[:1500]}"


def _run_example(program, task):
    ok, msg = _static_check(program, task)
    if not ok:
        return False, msg
    res = run_program(program, task["signature"], [task["cases"][0]], timeout=60)
    if res["stage"] != "executed":
        return False, f"{res['stage']}: {res.get('error', '')[:1500]}"
    good, m = compare(res["outputs"][0], expected_outputs(task)[0], task)
    return good, ("Output matches the expected output on this example." if good else f"Mismatch on this example: {m}")


def run_agent_trial(client, task, condition, budget, trial=0):
    r = render(task, condition)
    user = task_prompt(task, r["content"]).replace(
        "Reply with the complete program in a single ```mlir code block.",
        "Use the tools to develop and check the program, then call submit with the complete program.")
    tools = list(AGENT_TOOLS)
    if condition in INTERACTIVE:
        user = interactive_preamble(r["available_ops"], condition == "interactive_core").format(
            max_calls=budget["agent_max_tool_calls"]) + "\n\n" + user
        tools = TOOL_SPEC + tools
    msgs = [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": user}]
    log = dict(track="agent", task_id=task["task_id"], template=task["template"], family=task["family"],
               source=task.get("source"), split=task["split"], condition=condition, trial=trial, model=client.model,
               provider=client.provider, available_fact_ids=r["fact_ids"], tool_calls=0, tool_counts={},
               retrieved_fact_ids=list(r.get("upfront_fact_ids", [])), tool_queries=[], input_tokens=0, output_tokens=0,
               model_calls=0, submitted=False, programs_tried=0, t0=time.time())
    last_prog, final, nudged = None, None, False
    while log["model_calls"] < budget["agent_max_model_calls"] and final is None:
        out_of_tools = log["tool_calls"] >= budget["agent_max_tool_calls"]
        if out_of_tools:
            msgs.append({"role": "user", "content": "Tool budget exhausted. Reply with your final program in a ```mlir code block."})
        resp = client.chat(msgs, tools=None if out_of_tools else tools, max_tokens=budget["max_output_tokens"], seed=trial)
        log["model_calls"] += 1
        log["input_tokens"] += resp["input_tokens"]; log["output_tokens"] += resp["output_tokens"]
        if not resp["tool_calls"]:
            msgs.append({"role": "assistant", "content": resp["text"] or ""})
            p = extract_program(resp["text"] or "")
            if p and (out_of_tools or nudged):
                final, last_prog = p, p; break
            if p:
                last_prog = p
            if out_of_tools:
                break
            nudged = True
            msgs.append({"role": "user", "content": "Please use the tools (verify / run_example) and finish by calling submit with the complete program."})
            continue
        msgs.append(resp["assistant_message"])
        for tc in resp["tool_calls"]:
            log["tool_calls"] += 1
            name = tc["name"]; log["tool_counts"][name] = log["tool_counts"].get(name, 0) + 1
            try:
                a = json.loads(tc["arguments"] or "{}")
            except json.JSONDecodeError:
                a = {}
            if log["tool_calls"] > budget["agent_max_tool_calls"]:
                content = "Tool budget exhausted."
            elif name == "get_facts" and condition in INTERACTIVE:
                out, ids = tool_get_facts(r["available_ops"], a.get("operation", ""), a.get("category", "all"))
                log["tool_queries"].append([a.get("operation"), a.get("category", "all")])
                log["retrieved_fact_ids"] += [i for i in ids if i not in log["retrieved_fact_ids"]]
                content = json.dumps(out, ensure_ascii=False)
            elif name in ("verify", "run_example", "submit"):
                prog = extract_program(a.get("program", "")) or a.get("program")
                if prog:
                    last_prog = prog; log["programs_tried"] += 1
                if name == "submit":
                    final = prog; log["submitted"] = True; content = "Submitted."
                else:
                    ok, msg = (_static_check if name == "verify" else _run_example)(prog, task)
                    content = msg
            else:
                content = f"Unknown tool {name}."
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": content})
        if final is not None:
            break
    prog = final if final is not None else last_prog
    ev = grade(prog, task)
    log.update(semantic_correct=ev.semantic_correct, verified=ev.verified, error_type=ev.error_type,
               diagnostic=ev.diagnostic[:800], program=prog, n_retrieved_facts=len(log["retrieved_fact_ids"]),
               elapsed_seconds=round(time.time() - log.pop("t0"), 2))
    return log, msgs
