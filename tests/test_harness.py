import numpy as np
import pytest
from agentisa.harness import extract_program, grade
from agentisa.tasks import all_tasks, compare, expected_outputs

jax = pytest.importorskip("jax")  # evaluator tests need the pinned toolchain (.venv)


def test_extract_program_variants():
    assert extract_program("x\n```mlir\nfunc.func @main() {}\n```\n") == "func.func @main() {}"
    assert extract_program("blah func.func @main() {}") == "func.func @main() {}"
    assert extract_program("no code") is None


def test_oracle_checks_shape_and_dtype():
    t = all_tasks("dev")[0]; e = expected_outputs(t)[0]
    assert not compare(e.astype(np.float64), e, t)[0]
    assert not compare(e.reshape(-1), e, t)[0]
    assert compare(e.copy(), e, t)[0]


def test_reference_solution_passes_and_wrong_program_fails():
    t = [t for t in all_tasks("dev") if t["template"] == "bias_add_row"][0]
    assert grade(t["sol"], t).semantic_correct
    wrong = t["sol"].replace("stablehlo.add", "stablehlo.subtract")
    ev = grade(wrong, t)
    assert ev.verified and not ev.semantic_correct and ev.error_type == "semantic_mismatch"


def test_signature_and_forbidden_ops_rejected():
    t = [t for t in all_tasks("dev") if t["template"] == "relu"][0]
    bad_sig = t["sol"].replace("-> tensor", "-> tensor", 1).replace("@main(%x: tensor<", "@main(%x: tensor<1x", 1)
    assert grade(bad_sig, t).semantic_correct is False
    sh = t["signature"].split("-> ")[1]
    cc = t["sol"].replace(f"%r = stablehlo.maximum %x, %z : {sh}",
                          f'%r = "stablehlo.custom_call"(%x) {{call_target_name = "evil"}} : ({sh}) -> {sh}')
    assert grade(cc, t).stage == "forbidden_op"
