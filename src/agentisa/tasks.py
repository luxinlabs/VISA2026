"""Parametric StableHLO task suite with independent numpy oracles.

Each template, given a seeded RNG, yields a task: an exact @main signature, a precise mathematical
specification, the list of relevant operations (identical for all treatments), held-out input cases,
a numpy reference (the oracle; it never looks at StableHLO) and a hand-written reference StableHLO
solution used only to validate the task (gate G1). Instances of one template share a cluster id.
"""
import hashlib, json
from pathlib import Path
import numpy as np

F32, I32 = "f32", "i32"
NP = {F32: np.float32, I32: np.int32, "i1": np.bool_, "i64": np.int64}
MLIR_DT = {"float32": F32, "int32": I32, "int64": "i64", "bool": "i1"}


def T(shape, dt=F32):
    return "tensor<" + "".join(f"{d}x" for d in shape) + dt + ">"


def distinct(rng, k, lo=2, hi=7):
    return [int(v) for v in rng.choice(np.arange(lo, hi + 1), size=k, replace=False)]


def fn(args, out, body):
    sig = ", ".join(f"%{n}: {T(s, d)}" for n, s, d in args)
    return f"func.func @main({sig}) -> {T(*out)} {{\n" + "\n".join("  " + l for l in body.strip().splitlines()) + \
        f"\n  return %r : {T(*out)}\n}}\n"


def red_body(op, et=F32):
    return (f'({{\n  ^bb0(%acc: tensor<{et}>, %elt: tensor<{et}>):\n    %comb = "stablehlo.{op}"(%acc, %elt) : '
            f'(tensor<{et}>, tensor<{et}>) -> tensor<{et}>\n    "stablehlo.return"(%comb) : (tensor<{et}>) -> ()\n  }})')


def bcast(dst, src, src_shape, out_shape, dims, dt=F32):
    return (f'%{dst} = "stablehlo.broadcast_in_dim"(%{src}) {{broadcast_dimensions = array<i64: '
            f'{", ".join(map(str, dims))}>}} : ({T(src_shape, dt)}) -> {T(out_shape, dt)}')


def dot(dst, a, b, sa, sb, so, lb, rb, lc, rc):
    batch = f"lhs_batching_dimensions = {lb}, rhs_batching_dimensions = {rb}, " if lb else ""
    return (f'%{dst} = "stablehlo.dot_general"(%{a}, %{b}) {{dot_dimension_numbers = #stablehlo.dot<{batch}'
            f'lhs_contracting_dimensions = {lc}, rhs_contracting_dimensions = {rc}>}} : '
            f'({T(sa)}, {T(sb)}) -> {T(so)}')


# ------------------------------------------------------------------ templates
# each returns dict(args=[(name, shape, dtype)], out=(shape, dtype), spec=str, ops=[...], ref=callable,
#                   sol=str, family=str, gen=optional callable(rng, case_idx) -> list of arrays, tol=optional)
def bias_add_row(r):
    M, N = distinct(r, 2)
    return dict(family="broadcast", args=[("x", [M, N], F32), ("b", [N], F32)], out=([M, N], F32),
                spec="out[i, j] = x[i, j] + b[j] for all i, j.", ops=["broadcast_in_dim", "elementwise"],
                ref=lambda x, b: x + b[None, :],
                sol=fn([("x", [M, N], F32), ("b", [N], F32)], ([M, N], F32),
                       bcast("bb", "b", [N], [M, N], [1]) + f"\n%r = stablehlo.add %x, %bb : {T([M, N])}"))


def bias_add_col(r):
    M, N = distinct(r, 2)
    return dict(family="broadcast", args=[("x", [M, N], F32), ("b", [M], F32)], out=([M, N], F32),
                spec="out[i, j] = x[i, j] + b[i] for all i, j.", ops=["broadcast_in_dim", "elementwise"],
                ref=lambda x, b: x + b[:, None],
                sol=fn([("x", [M, N], F32), ("b", [M], F32)], ([M, N], F32),
                       bcast("bb", "b", [M], [M, N], [0]) + f"\n%r = stablehlo.add %x, %bb : {T([M, N])}"))


def outer_product(r):
    M, N = distinct(r, 2)
    return dict(family="broadcast", args=[("u", [M], F32), ("v", [N], F32)], out=([M, N], F32),
                spec="out[i, j] = u[i] * v[j] for all i, j (outer product).", ops=["broadcast_in_dim", "elementwise"],
                ref=lambda u, v: np.outer(u, v),
                sol=fn([("u", [M], F32), ("v", [N], F32)], ([M, N], F32),
                       bcast("uu", "u", [M], [M, N], [0]) + "\n" + bcast("vv", "v", [N], [M, N], [1]) +
                       f"\n%r = stablehlo.multiply %uu, %vv : {T([M, N])}"))


def broadcast_size1(r):
    M, N = distinct(r, 2)
    return dict(family="broadcast", args=[("y", [M, N], F32), ("s", [1, N], F32)], out=([M, N], F32),
                spec="out[i, j] = y[i, j] * s[0, j] for all i, j.", ops=["broadcast_in_dim", "elementwise"],
                ref=lambda y, s: y * s,
                sol=fn([("y", [M, N], F32), ("s", [1, N], F32)], ([M, N], F32),
                       bcast("ss", "s", [1, N], [M, N], [0, 1]) + f"\n%r = stablehlo.multiply %y, %ss : {T([M, N])}"))


def rank_expand_mid(r):
    A, N, B = distinct(r, 3)
    return dict(family="broadcast", args=[("x", [A, N, B], F32), ("v", [N], F32)], out=([A, N, B], F32),
                spec="out[a, n, b] = x[a, n, b] - v[n] for all a, n, b.", ops=["broadcast_in_dim", "elementwise"],
                ref=lambda x, v: x - v[None, :, None],
                sol=fn([("x", [A, N, B], F32), ("v", [N], F32)], ([A, N, B], F32),
                       bcast("vv", "v", [N], [A, N, B], [1]) + f"\n%r = stablehlo.subtract %x, %vv : {T([A, N, B])}"))


def scale_outer_dims(r):
    A, B, C = distinct(r, 3)
    return dict(family="broadcast", args=[("x", [A, B, C], F32), ("s", [A, C], F32)], out=([A, B, C], F32),
                spec="out[a, b, c] = x[a, b, c] * s[a, c] for all a, b, c.", ops=["broadcast_in_dim", "elementwise"],
                ref=lambda x, s: x * s[:, None, :],
                sol=fn([("x", [A, B, C], F32), ("s", [A, C], F32)], ([A, B, C], F32),
                       bcast("ss", "s", [A, C], [A, B, C], [0, 2]) + f"\n%r = stablehlo.multiply %x, %ss : {T([A, B, C])}"))


def sum_axis(r):
    A, B, C = distinct(r, 3); k = int(r.integers(0, 3)); sh = [A, B, C]; out = [d for i, d in enumerate(sh) if i != k]
    return dict(family="reduce", args=[("x", sh, F32)], out=(out, F32),
                spec=f"out = the sum of x over dimension {k} (that dimension is removed from the result).",
                ops=["reduce", "constant", "elementwise"], ref=lambda x: x.sum(axis=k),
                sol=fn([("x", sh, F32)], (out, F32),
                       f"%z = stablehlo.constant dense<0.0> : tensor<f32>\n"
                       f'%r = "stablehlo.reduce"(%x, %z) {red_body("add")} {{dimensions = array<i64: {k}>}} : ({T(sh)}, tensor<f32>) -> {T(out)}'))


def max_axis(r):
    M, N = distinct(r, 2); k = int(r.integers(0, 2)); sh = [M, N]; out = [sh[1 - k]]
    return dict(family="reduce", args=[("x", sh, F32)], out=(out, F32),
                spec=f"out = the maximum of x over dimension {k} (that dimension is removed). Inputs may be all negative.",
                ops=["reduce", "constant", "elementwise"], ref=lambda x: x.max(axis=k),
                gen=lambda rng, c: [(rng.standard_normal(sh) - (6.0 if c == 2 else 0.0)).astype(np.float32)],
                sol=fn([("x", sh, F32)], (out, F32),
                       f"%z = stablehlo.constant dense<0xFF800000> : tensor<f32>\n"
                       f'%r = "stablehlo.reduce"(%x, %z) {red_body("maximum")} {{dimensions = array<i64: {k}>}} : ({T(sh)}, tensor<f32>) -> {T(out)}'))


def mean_axis(r):
    M, N = distinct(r, 2, 3, 8); k = int(r.integers(0, 2)); sh = [M, N]; out = [sh[1 - k]]
    return dict(family="reduce", args=[("x", sh, F32)], out=(out, F32),
                spec=f"out = the arithmetic mean of x over dimension {k} (that dimension is removed).",
                ops=["reduce", "constant", "elementwise"], ref=lambda x: x.mean(axis=k),
                sol=fn([("x", sh, F32)], (out, F32),
                       f"%z = stablehlo.constant dense<0.0> : tensor<f32>\n"
                       f'%s = "stablehlo.reduce"(%x, %z) {red_body("add")} {{dimensions = array<i64: {k}>}} : ({T(sh)}, tensor<f32>) -> {T(out)}\n'
                       f"%n = stablehlo.constant dense<{float(sh[k])}> : {T(out)}\n"
                       f"%r = stablehlo.divide %s, %n : {T(out)}"))


def sum_two_axes(r):
    A, B, C = distinct(r, 3)
    return dict(family="reduce", args=[("x", [A, B, C], F32)], out=([B], F32),
                spec="out[b] = sum over a and c of x[a, b, c].", ops=["reduce", "constant", "elementwise"],
                ref=lambda x: x.sum(axis=(0, 2)),
                sol=fn([("x", [A, B, C], F32)], ([B], F32),
                       f"%z = stablehlo.constant dense<0.0> : tensor<f32>\n"
                       f'%r = "stablehlo.reduce"(%x, %z) {red_body("add")} {{dimensions = array<i64: 0, 2>}} : ({T([A, B, C])}, tensor<f32>) -> {T([B])}'))


def softmax_rows(r):
    M, N = distinct(r, 2, 3, 8)

    def ref(x):
        e = np.exp(x - x.max(axis=1, keepdims=True)); return e / e.sum(axis=1, keepdims=True)
    return dict(family="composite", args=[("x", [M, N], F32)], out=([M, N], F32),
                spec="out[i, j] = exp(x[i, j]) / sum_k exp(x[i, k]) (softmax over dimension 1).",
                ops=["reduce", "broadcast_in_dim", "elementwise", "constant"], ref=ref,
                sol=fn([("x", [M, N], F32)], ([M, N], F32),
                       f"%ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n"
                       f'%mx = "stablehlo.reduce"(%x, %ninf) {red_body("maximum")} {{dimensions = array<i64: 1>}} : ({T([M, N])}, tensor<f32>) -> {T([M])}\n'
                       + bcast("mb", "mx", [M], [M, N], [0]) + "\n"
                       f"%d = stablehlo.subtract %x, %mb : {T([M, N])}\n%e = stablehlo.exponential %d : {T([M, N])}\n"
                       f"%z = stablehlo.constant dense<0.0> : tensor<f32>\n"
                       f'%s = "stablehlo.reduce"(%e, %z) {red_body("add")} {{dimensions = array<i64: 1>}} : ({T([M, N])}, tensor<f32>) -> {T([M])}\n'
                       + bcast("sb", "s", [M], [M, N], [0]) + f"\n%r = stablehlo.divide %e, %sb : {T([M, N])}"))


def l2_normalize_rows(r):
    M, N = distinct(r, 2, 3, 8)
    return dict(family="composite", args=[("x", [M, N], F32)], out=([M, N], F32),
                spec="out[i, j] = x[i, j] / sqrt(sum_k x[i, k]^2) (each row scaled to unit L2 norm).",
                ops=["reduce", "broadcast_in_dim", "elementwise", "constant"],
                ref=lambda x: x / np.sqrt((x * x).sum(axis=1, keepdims=True)),
                sol=fn([("x", [M, N], F32)], ([M, N], F32),
                       f"%sq = stablehlo.multiply %x, %x : {T([M, N])}\n%z = stablehlo.constant dense<0.0> : tensor<f32>\n"
                       f'%s = "stablehlo.reduce"(%sq, %z) {red_body("add")} {{dimensions = array<i64: 1>}} : ({T([M, N])}, tensor<f32>) -> {T([M])}\n'
                       f"%n = stablehlo.sqrt %s : {T([M])}\n" + bcast("nb", "n", [M], [M, N], [0]) +
                       f"\n%r = stablehlo.divide %x, %nb : {T([M, N])}"))


def matmul(r):
    M, K, N = distinct(r, 3)
    return dict(family="dot", args=[("a", [M, K], F32), ("b", [K, N], F32)], out=([M, N], F32),
                spec="out[i, j] = sum_k a[i, k] * b[k, j] (matrix product).", ops=["dot_general"],
                ref=lambda a, b: a @ b,
                sol=fn([("a", [M, K], F32), ("b", [K, N], F32)], ([M, N], F32),
                       dot("r", "a", "b", [M, K], [K, N], [M, N], None, None, [1], [0])))


def matvec(r):
    M, K = distinct(r, 2)
    return dict(family="dot", args=[("a", [M, K], F32), ("x", [K], F32)], out=([M], F32),
                spec="out[i] = sum_k a[i, k] * x[k] (matrix-vector product).", ops=["dot_general"],
                ref=lambda a, x: a @ x,
                sol=fn([("a", [M, K], F32), ("x", [K], F32)], ([M], F32),
                       dot("r", "a", "x", [M, K], [K], [M], None, None, [1], [0])))


def batched_matmul(r):
    B, M, K, N = distinct(r, 4, 2, 7)
    return dict(family="dot", args=[("a", [B, M, K], F32), ("b", [B, K, N], F32)], out=([B, M, N], F32),
                spec="out[n, i, j] = sum_k a[n, i, k] * b[n, k, j] (batched matrix product).", ops=["dot_general"],
                ref=lambda a, b: a @ b,
                sol=fn([("a", [B, M, K], F32), ("b", [B, K, N], F32)], ([B, M, N], F32),
                       dot("r", "a", "b", [B, M, K], [B, K, N], [B, M, N], [0], [0], [2], [1])))


def at_b(r):
    K, M, N = distinct(r, 3)
    return dict(family="dot", args=[("a", [K, M], F32), ("b", [K, N], F32)], out=([M, N], F32),
                spec="out[i, j] = sum_k a[k, i] * b[k, j] (that is, transpose(a) times b).", ops=["dot_general"],
                ref=lambda a, b: a.T @ b,
                sol=fn([("a", [K, M], F32), ("b", [K, N], F32)], ([M, N], F32),
                       dot("r", "a", "b", [K, M], [K, N], [M, N], None, None, [0], [0])))


def a_bt(r):
    M, K, N = distinct(r, 3)
    return dict(family="dot", args=[("a", [M, K], F32), ("b", [N, K], F32)], out=([M, N], F32),
                spec="out[i, j] = sum_k a[i, k] * b[j, k] (that is, a times transpose(b)).", ops=["dot_general"],
                ref=lambda a, b: a @ b.T,
                sol=fn([("a", [M, K], F32), ("b", [N, K], F32)], ([M, N], F32),
                       dot("r", "a", "b", [M, K], [N, K], [M, N], None, None, [1], [1])))


def batched_dot(r):
    B, K = distinct(r, 2)
    return dict(family="dot", args=[("x", [B, K], F32), ("y", [B, K], F32)], out=([B], F32),
                spec="out[n] = sum_k x[n, k] * y[n, k] (row-wise dot products).", ops=["dot_general"],
                ref=lambda x, y: (x * y).sum(axis=1),
                sol=fn([("x", [B, K], F32), ("y", [B, K], F32)], ([B], F32),
                       dot("r", "x", "y", [B, K], [B, K], [B], [0], [0], [1], [1])))


def transpose2d(r):
    M, N = distinct(r, 2)
    return dict(family="layout", args=[("x", [M, N], F32)], out=([N, M], F32),
                spec="out[j, i] = x[i, j] (matrix transpose).", ops=["transpose"], ref=lambda x: x.T.copy(),
                sol=fn([("x", [M, N], F32)], ([N, M], F32),
                       f'%r = "stablehlo.transpose"(%x) {{permutation = array<i64: 1, 0>}} : ({T([M, N])}) -> {T([N, M])}'))


def transpose3d(r):
    A, B, C = distinct(r, 3)
    return dict(family="layout", args=[("x", [A, B, C], F32)], out=([C, A, B], F32),
                spec="out[c, a, b] = x[a, b, c] for all a, b, c.", ops=["transpose"],
                ref=lambda x: np.transpose(x, (2, 0, 1)).copy(),
                sol=fn([("x", [A, B, C], F32)], ([C, A, B], F32),
                       f'%r = "stablehlo.transpose"(%x) {{permutation = array<i64: 2, 0, 1>}} : ({T([A, B, C])}) -> {T([C, A, B])}'))


def flatten_last2(r):
    A, B, C = distinct(r, 3)
    return dict(family="layout", args=[("x", [A, B, C], F32)], out=([A, B * C], F32),
                spec=f"out[a, b * {C} + c] = x[a, b, c] (merge the last two dimensions in row-major order).",
                ops=["reshape"], ref=lambda x: x.reshape(A, B * C),
                sol=fn([("x", [A, B, C], F32)], ([A, B * C], F32), f"%r = stablehlo.reshape %x : ({T([A, B, C])}) -> {T([A, B * C])}"))


def split_heads(r):
    S, H, D = distinct(r, 3)
    return dict(family="layout", args=[("x", [S, H * D], F32)], out=([H, S, D], F32),
                spec=f"out[h, s, d] = x[s, h * {D} + d] for all h < {H}, s < {S}, d < {D}.",
                ops=["reshape", "transpose"], ref=lambda x: x.reshape(S, H, D).transpose(1, 0, 2).copy(),
                sol=fn([("x", [S, H * D], F32)], ([H, S, D], F32),
                       f"%t = stablehlo.reshape %x : ({T([S, H * D])}) -> {T([S, H, D])}\n"
                       f'%r = "stablehlo.transpose"(%t) {{permutation = array<i64: 1, 0, 2>}} : ({T([S, H, D])}) -> {T([H, S, D])}'))


def strided_slice(r):
    N = int(r.integers(10, 16)); s = int(r.integers(1, 4)); k = int(r.integers(2, 4)); e = N - int(r.integers(0, 3))
    n = -(-(e - s) // k)
    return dict(family="slice_pad", args=[("x", [N], F32)], out=([n], F32),
                spec=f"out[i] = x[{s} + {k} * i] for i = 0 .. {n - 1} (Python x[{s}:{e}:{k}]).", ops=["slice"],
                ref=lambda x: x[s:e:k].copy(),
                sol=fn([("x", [N], F32)], ([n], F32),
                       f'%r = "stablehlo.slice"(%x) {{start_indices = array<i64: {s}>, limit_indices = array<i64: {e}>, strides = array<i64: {k}>}} : ({T([N])}) -> {T([n])}'))


def slice2d(r):
    M, N = [int(v) for v in r.integers(6, 10, size=2)]
    r0, c0 = int(r.integers(1, 3)), int(r.integers(0, 3)); r1, c1 = M - int(r.integers(0, 3)), N - int(r.integers(1, 3))
    return dict(family="slice_pad", args=[("x", [M, N], F32)], out=([r1 - r0, c1 - c0], F32),
                spec=f"out = x[{r0}:{r1}, {c0}:{c1}] in Python notation (rows {r0}..{r1 - 1}, columns {c0}..{c1 - 1}).",
                ops=["slice"], ref=lambda x: x[r0:r1, c0:c1].copy(),
                sol=fn([("x", [M, N], F32)], ([r1 - r0, c1 - c0], F32),
                       f'%r = "stablehlo.slice"(%x) {{start_indices = array<i64: {r0}, {c0}>, limit_indices = array<i64: {r1}, {c1}>, strides = array<i64: 1, 1>}} : ({T([M, N])}) -> {T([r1 - r0, c1 - c0])}'))


def pad_edges(r):
    M, N = distinct(r, 2); lo = [int(v) for v in r.integers(0, 3, 2)]; hi = [int(v) for v in r.integers(1, 3, 2)]
    val = float(r.choice([0.0, -1.0, 2.5])); out = [M + lo[0] + hi[0], N + lo[1] + hi[1]]

    def ref(x):
        y = np.full(out, val, np.float32); y[lo[0]:lo[0] + M, lo[1]:lo[1] + N] = x; return y
    return dict(family="slice_pad", args=[("x", [M, N], F32)], out=(out, F32),
                spec=(f"out has shape {out[0]}x{out[1]}: x is placed so that out[i + {lo[0]}, j + {lo[1]}] = x[i, j], "
                      f"with {lo[0]} rows above, {hi[0]} rows below, {lo[1]} columns left and {hi[1]} columns right "
                      f"filled with the value {val}."),
                ops=["pad", "constant"], ref=ref,
                sol=fn([("x", [M, N], F32)], (out, F32),
                       f"%pv = stablehlo.constant dense<{val}> : tensor<f32>\n"
                       f'%r = "stablehlo.pad"(%x, %pv) {{edge_padding_low = array<i64: {lo[0]}, {lo[1]}>, edge_padding_high = array<i64: {hi[0]}, {hi[1]}>, interior_padding = array<i64: 0, 0>}} : ({T([M, N])}, tensor<f32>) -> {T(out)}'))


def upsample_interior(r):
    N = int(r.integers(4, 9)); z = int(r.integers(1, 3)); L = N + (N - 1) * z

    def ref(x):
        y = np.zeros(L, np.float32); y[::z + 1] = x; return y
    return dict(family="slice_pad", args=[("x", [N], F32)], out=([L], F32),
                spec=f"out has length {L}: out[{z + 1} * i] = x[i] and every other element is 0.0 ({z} zero(s) inserted between consecutive elements, none at the ends).",
                ops=["pad", "constant"], ref=ref,
                sol=fn([("x", [N], F32)], ([L], F32),
                       f"%pv = stablehlo.constant dense<0.0> : tensor<f32>\n"
                       f'%r = "stablehlo.pad"(%x, %pv) {{edge_padding_low = array<i64: 0>, edge_padding_high = array<i64: 0>, interior_padding = array<i64: {z}>}} : ({T([N])}, tensor<f32>) -> {T([L])}'))


def concat(r):
    M, N1, N2 = distinct(r, 3)
    return dict(family="slice_pad", args=[("x", [M, N1], F32), ("y", [M, N2], F32)], out=([M, N1 + N2], F32),
                spec=f"out[i, j] = x[i, j] for j < {N1} and out[i, {N1} + j] = y[i, j] (join along dimension 1).",
                ops=["concatenate"], ref=lambda x, y: np.concatenate([x, y], axis=1),
                sol=fn([("x", [M, N1], F32), ("y", [M, N2], F32)], ([M, N1 + N2], F32),
                       f'%r = "stablehlo.concatenate"(%x, %y) {{dimension = 1 : i64}} : ({T([M, N1])}, {T([M, N2])}) -> {T([M, N1 + N2])}'))


def concat3_axis0(r):
    A, B, C, N = distinct(r, 4, 1, 6)
    return dict(family="slice_pad", args=[("x", [A, N], F32), ("y", [B, N], F32), ("z", [C, N], F32)], out=([A + B + C, N], F32),
                spec="out = the rows of x, followed by the rows of y, followed by the rows of z (join along dimension 0).",
                ops=["concatenate"], ref=lambda x, y, z: np.concatenate([x, y, z], axis=0),
                sol=fn([("x", [A, N], F32), ("y", [B, N], F32), ("z", [C, N], F32)], ([A + B + C, N], F32),
                       f'%r = "stablehlo.concatenate"(%x, %y, %z) {{dimension = 0 : i64}} : ({T([A, N])}, {T([B, N])}, {T([C, N])}) -> {T([A + B + C, N])}'))


def causal_mask(r):
    N = int(r.integers(3, 8))
    return dict(family="mask_index", args=[("x", [N, N], F32)], out=([N, N], F32),
                spec="out[i, j] = x[i, j] if j <= i, otherwise 0.0 (keep the lower triangle including the diagonal).",
                ops=["iota", "compare", "select", "constant"], ref=lambda x: np.tril(x),
                sol=fn([("x", [N, N], F32)], ([N, N], F32),
                       f'%ri = "stablehlo.iota"() {{iota_dimension = 0 : i64}} : () -> {T([N, N], I32)}\n'
                       f'%ci = "stablehlo.iota"() {{iota_dimension = 1 : i64}} : () -> {T([N, N], I32)}\n'
                       f'%m = "stablehlo.compare"(%ci, %ri) {{comparison_direction = #stablehlo<comparison_direction LE>}} : ({T([N, N], I32)}, {T([N, N], I32)}) -> {T([N, N], "i1")}\n'
                       f"%z = stablehlo.constant dense<0.0> : {T([N, N])}\n"
                       f'%r = "stablehlo.select"(%m, %x, %z) : ({T([N, N], "i1")}, {T([N, N])}, {T([N, N])}) -> {T([N, N])}'))


def one_hot(r):
    N, C = distinct(r, 2, 3, 8)
    return dict(family="mask_index", args=[("idx", [N], I32)], out=([N, C], F32),
                spec=f"out[n, c] = 1.0 if idx[n] == c else 0.0, for c < {C}; a row whose idx is outside 0..{C - 1} is all zeros.",
                ops=["iota", "broadcast_in_dim", "compare", "select", "constant"],
                ref=lambda idx: (idx[:, None] == np.arange(C)[None, :]).astype(np.float32),
                gen=lambda rng, c: [rng.integers(-1 if c == 2 else 0, C + (1 if c == 2 else 0), N).astype(np.int32)],
                sol=fn([("idx", [N], I32)], ([N, C], F32),
                       f'%io = "stablehlo.iota"() {{iota_dimension = 1 : i64}} : () -> {T([N, C], I32)}\n'
                       + bcast("ib", "idx", [N], [N, C], [0], I32) + "\n"
                       f'%m = "stablehlo.compare"(%ib, %io) {{comparison_direction = #stablehlo<comparison_direction EQ>}} : ({T([N, C], I32)}, {T([N, C], I32)}) -> {T([N, C], "i1")}\n'
                       f"%one = stablehlo.constant dense<1.0> : {T([N, C])}\n%zero = stablehlo.constant dense<0.0> : {T([N, C])}\n"
                       f'%r = "stablehlo.select"(%m, %one, %zero) : ({T([N, C], "i1")}, {T([N, C])}, {T([N, C])}) -> {T([N, C])}'))


def relu(r):
    M, N = distinct(r, 2)
    return dict(family="elementwise", args=[("x", [M, N], F32)], out=([M, N], F32),
                spec="out[i, j] = max(x[i, j], 0.0).", ops=["elementwise", "constant"], ref=lambda x: np.maximum(x, 0),
                sol=fn([("x", [M, N], F32)], ([M, N], F32),
                       f"%z = stablehlo.constant dense<0.0> : {T([M, N])}\n%r = stablehlo.maximum %x, %z : {T([M, N])}"))


def leaky_relu(r):
    M, N = distinct(r, 2); al = float(r.choice([0.1, 0.2, 0.01]))
    return dict(family="mask_index", args=[("x", [M, N], F32)], out=([M, N], F32),
                spec=f"out[i, j] = x[i, j] if x[i, j] > 0, otherwise {al} * x[i, j].",
                ops=["compare", "select", "elementwise", "constant"], ref=lambda x: np.where(x > 0, x, np.float32(al) * x),
                sol=fn([("x", [M, N], F32)], ([M, N], F32),
                       f"%z = stablehlo.constant dense<0.0> : {T([M, N])}\n%a = stablehlo.constant dense<{al}> : {T([M, N])}\n"
                       f'%m = "stablehlo.compare"(%x, %z) {{comparison_direction = #stablehlo<comparison_direction GT>}} : ({T([M, N])}, {T([M, N])}) -> {T([M, N], "i1")}\n'
                       f"%s = stablehlo.multiply %x, %a : {T([M, N])}\n"
                       f'%r = "stablehlo.select"(%m, %x, %s) : ({T([M, N], "i1")}, {T([M, N])}, {T([M, N])}) -> {T([M, N])}'))


def clamp_scalar(r):
    N = int(r.integers(5, 12)); lo, hi = sorted(float(v) for v in r.choice([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0], 2, replace=False))
    return dict(family="elementwise", args=[("x", [N], F32)], out=([N], F32),
                spec=f"out[i] = min(max(x[i], {lo}), {hi}).", ops=["clamp", "constant"],
                ref=lambda x: np.clip(x, lo, hi),
                sol=fn([("x", [N], F32)], ([N], F32),
                       f"%lo = stablehlo.constant dense<{lo}> : tensor<f32>\n%hi = stablehlo.constant dense<{hi}> : tensor<f32>\n"
                       f'%r = "stablehlo.clamp"(%lo, %x, %hi) : (tensor<f32>, {T([N])}, tensor<f32>) -> {T([N])}'))


def trunc_to_int(r):
    N = int(r.integers(5, 12))
    return dict(family="convert", args=[("x", [N], F32)], out=([N], I32),
                spec="out[i] = x[i] converted to a 32-bit integer by rounding toward zero (e.g. -2.7 -> -2, 2.7 -> 2).",
                ops=["convert"], ref=lambda x: np.trunc(x).astype(np.int32),
                gen=lambda rng, c: [(rng.standard_normal(N) * 5).astype(np.float32)],
                sol=fn([("x", [N], F32)], ([N], I32), f"%r = stablehlo.convert %x : ({T([N])}) -> {T([N], I32)}"))


def count_positive(r):
    M, N = distinct(r, 2, 3, 8)

    def gen(rng, c):
        x = rng.standard_normal((M, N)).astype(np.float32); x[rng.random((M, N)) < 0.25] = 0.0; return [x]
    return dict(family="convert", args=[("x", [M, N], F32)], out=([M], I32),
                spec="out[i] = the number of j with x[i, j] > 0 (strictly greater; zeros do not count), as i32.",
                ops=["compare", "convert", "reduce", "constant", "elementwise"], ref=lambda x: (x > 0).sum(axis=1).astype(np.int32),
                gen=gen,
                sol=fn([("x", [M, N], F32)], ([M], I32),
                       f"%z = stablehlo.constant dense<0.0> : {T([M, N])}\n"
                       f'%m = "stablehlo.compare"(%x, %z) {{comparison_direction = #stablehlo<comparison_direction GT>}} : ({T([M, N])}, {T([M, N])}) -> {T([M, N], "i1")}\n'
                       f"%c = stablehlo.convert %m : ({T([M, N], 'i1')}) -> {T([M, N], I32)}\n"
                       f"%zi = stablehlo.constant dense<0> : tensor<i32>\n"
                       f'%r = "stablehlo.reduce"(%c, %zi) {red_body("add", I32)} {{dimensions = array<i64: 1>}} : ({T([M, N], I32)}, tensor<i32>) -> {T([M], I32)}'))


def reverse_axis(r):
    M, N = distinct(r, 2); k = int(r.integers(0, 2))
    idx = "out[i, j] = x[" + (f"{M - 1} - i, j]" if k == 0 else f"i, {N - 1} - j]")
    return dict(family="layout", args=[("x", [M, N], F32)], out=([M, N], F32),
                spec=f"{idx} (reverse dimension {k}).", ops=["reverse"], ref=lambda x: np.flip(x, axis=k).copy(),
                sol=fn([("x", [M, N], F32)], ([M, N], F32),
                       f'%r = "stablehlo.reverse"(%x) {{dimensions = array<i64: {k}>}} : ({T([M, N])}) -> {T([M, N])}'))


def dynamic_rows(r):
    R, C = distinct(r, 2, 5, 9); K = int(r.integers(2, 4))

    def ref(x, i):
        s = int(np.clip(int(i), 0, R - K)); return x[s:s + K].copy()

    def gen(rng, c):
        i = [int(rng.integers(0, R - K + 1)), -2, R][c]
        return [rng.standard_normal((R, C)).astype(np.float32), np.array(i, np.int32)]
    return dict(family="mask_index", args=[("x", [R, C], F32), ("i", [], I32)], out=([K, C], F32),
                spec=f"out = rows s .. s+{K - 1} of x, where s = min(max(i, 0), {R - K}) (the start row is clamped so the window stays inside x).",
                ops=["dynamic_slice", "constant"], ref=ref, gen=gen,
                sol=fn([("x", [R, C], F32), ("i", [], I32)], ([K, C], F32),
                       f"%z = stablehlo.constant dense<0> : tensor<i32>\n"
                       f'%r = "stablehlo.dynamic_slice"(%x, %i, %z) {{slice_sizes = array<i64: {K}, {C}>}} : ({T([R, C])}, tensor<i32>, tensor<i32>) -> {T([K, C])}'))


def pairwise_sqdist(r):
    M, N, D = distinct(r, 3)
    return dict(family="composite", args=[("x", [M, D], F32), ("y", [N, D], F32)], out=([M, N], F32),
                spec="out[i, j] = sum_d (x[i, d] - y[j, d])^2 (squared Euclidean distance between rows).",
                ops=["broadcast_in_dim", "elementwise", "reduce", "constant"],
                ref=lambda x, y: ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=2),
                sol=fn([("x", [M, D], F32), ("y", [N, D], F32)], ([M, N], F32),
                       bcast("xb", "x", [M, D], [M, N, D], [0, 2]) + "\n" + bcast("yb", "y", [N, D], [M, N, D], [1, 2]) +
                       f"\n%d = stablehlo.subtract %xb, %yb : {T([M, N, D])}\n%s = stablehlo.multiply %d, %d : {T([M, N, D])}\n"
                       f"%z = stablehlo.constant dense<0.0> : tensor<f32>\n"
                       f'%r = "stablehlo.reduce"(%s, %z) {red_body("add")} {{dimensions = array<i64: 2>}} : ({T([M, N, D])}, tensor<f32>) -> {T([M, N])}'))


TEMPLATES = [bias_add_row, bias_add_col, outer_product, broadcast_size1, rank_expand_mid, scale_outer_dims,
             sum_axis, max_axis, mean_axis, sum_two_axes, softmax_rows, l2_normalize_rows,
             matmul, matvec, batched_matmul, at_b, a_bt, batched_dot,
             transpose2d, transpose3d, flatten_last2, split_heads, reverse_axis,
             strided_slice, slice2d, pad_edges, upsample_interior, concat, concat3_axis0,
             causal_mask, one_hot, leaky_relu, dynamic_rows, relu, clamp_scalar, trunc_to_int, count_positive,
             pairwise_sqdist]

N_INSTANCES = 4  # instance 0 = development split, 1..3 = frozen evaluation split
N_CASES = 3


def _default_gen(args):
    def gen(rng, c):
        out = []
        for _, shape, dt in args:
            if dt == I32:
                out.append(rng.integers(-5, 6, size=shape).astype(np.int32))
            else:
                x = rng.standard_normal(shape).astype(np.float32)
                if c == 2 and x.size > 1:
                    x.flat[0] = 0.0
                out.append(x)
        return out
    return gen


def signature(task):
    return "(" + ", ".join(T(s, d) for _, s, d in task["args"]) + ") -> " + T(*task["out"])


def header(task):
    return "func.func @main(" + ", ".join(f"%{n}: {T(s, d)}" for n, s, d in task["args"]) + ") -> " + T(*task["out"])


def build_task(template, instance, base_seed=20261013):
    seed = int(hashlib.sha256(f"{base_seed}:{template.__name__}:{instance}".encode()).hexdigest()[:8], 16)
    t = template(np.random.default_rng(seed))
    t.update(template=template.__name__, instance=instance, task_id=f"{template.__name__}__{instance}",
             split="dev" if instance == 0 else "eval", seed=seed)
    t["signature"] = signature(t)
    t["header"] = header(t)
    gen = t.get("gen") or _default_gen(t["args"])
    crng = np.random.default_rng(seed + 1)
    t["cases"] = [gen(crng, c) for c in range(N_CASES)]
    return t


# ------------------------------------------------------------------ KernelBench-derived tasks
KB_DIR = Path(__file__).resolve().parents[2] / "data/processed/kernelbench"
# reference ops -> documented op groups; torchax lowering artifacts (while, dynamic_update_slice, gather) are not listed
OP_GROUP = {**{o: "elementwise" for o in ["add", "subtract", "multiply", "divide", "maximum", "minimum", "negate",
                                          "exponential", "sqrt", "rsqrt", "abs", "square"]},
            **{o: "math" for o in ["tanh", "log", "log_plus_one", "exponential_minus_one", "logistic", "power", "floor", "sign"]},
            **{o: "logical" for o in ["and", "or", "not"]}, "erf": "erf", "erfc": "erf",
            **{o: o for o in ["constant", "broadcast_in_dim", "reduce", "dot_general", "transpose", "reshape", "slice",
                              "pad", "concatenate", "iota", "compare", "select", "convert", "clamp", "reverse",
                              "dynamic_slice", "convolution", "reduce_window"]}}
GROUP_ORDER = ["constant", "elementwise", "math", "logical", "erf", "broadcast_in_dim", "reshape", "transpose",
               "convolution", "dot_general", "reduce", "reduce_window", "compare", "select", "convert", "clamp", "iota",
               "slice", "pad", "concatenate", "reverse", "dynamic_slice"]


def _kb_task(r):
    z = np.load(KB_DIR / f"{r['task_id']}.npz")
    n = len(r["arg_shapes"])
    names = [f"arg{i}" for i in range(n)]
    args = [(names[i], r["arg_shapes"][i], MLIR_DT[r["arg_dtypes"][i]]) for i in range(n)]
    out = (r["out_shape"], MLIR_DT[r["out_dtype"]])
    groups = {OP_GROUP[o.split(".")[1]] for o in r["reference_ops"] if o.split(".")[1] in OP_GROUP}
    mapping = "; ".join(f"%{names[i]} = " + (f"parameter `{r['arg_names'][i]}`" if i < r["n_params"]
                                              else f"forward input {i - r['n_params']}") for i in range(n))
    spec = (f"the forward output of the PyTorch module below (the module is used as constructed, i.e. in training "
            f"mode, so normalization layers use batch statistics; its parameters are passed as arguments). "
            f"Argument mapping: {mapping}.\n```python\n{r['pytorch_source'].strip()}\n```")
    t = dict(task_id=r["task_id"], template=r["task_id"], instance=0, seed=0, source="kernelbench",
             upstream_file=r["source_file"], family=f"kb_level{r['level']}", args=args, out=out, spec=spec,
             ops=[g for g in GROUP_ORDER if g in groups], sol=r["sol"], tol=(1e-3, 1e-4),
             split="dev" if int(hashlib.sha256(r["task_id"].encode()).hexdigest(), 16) % 10 == 0 else "eval")
    t["cases"] = [[z[f"c{c}_a{i}"] for i in range(n)] for c in range(N_CASES)]
    t["expected"] = [z[f"y{c}"] for c in range(N_CASES)]
    t["signature"], t["header"] = signature(t), header(t)
    return t


def kb_tasks():
    rs = [json.loads(p.read_text()) for p in sorted((KB_DIR / "results").glob("*.json"))]
    return [_kb_task(r) for r in rs if r["status"] == "converted"]


def all_tasks(split=None, suite="synthetic"):
    ts = []
    if suite in ("synthetic", "all"):
        ts += [dict(build_task(tp, i), source="synthetic") for tp in TEMPLATES for i in range(N_INSTANCES)]
    if suite in ("kernelbench", "all"):
        ts += kb_tasks()
    return [t for t in ts if split is None or t["split"] == split]


def expected_outputs(task):
    if "expected" in task:
        return [np.asarray(e) for e in task["expected"]]
    return [np.asarray(task["ref"](*case)).astype(NP[task["out"][1]]) for case in task["cases"]]


def compare(actual, expected, task):
    """Independent oracle comparison: exact shape and dtype, values exact (int/bool) or allclose (f32)."""
    actual = np.asarray(actual)
    if actual.shape != expected.shape:
        return False, f"result shape {list(actual.shape)} but expected {list(expected.shape)}"
    if actual.dtype != expected.dtype:
        return False, f"result dtype {actual.dtype} but expected {expected.dtype}"
    if expected.dtype == np.float32:
        rtol, atol = task.get("tol", (1e-4, 1e-5))
        ok = np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False)
    else:
        ok = np.array_equal(actual, expected)
    if ok:
        return True, ""
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    k = np.unravel_index(int(np.nanargmax(np.where(np.isnan(diff), np.inf, diff))), diff.shape)
    return False, (f"wrong values: at index {list(map(int, k))} expected {expected[k]!r} but got {actual[k]!r} "
                   f"({int((diff > 1e-4).sum())} of {diff.size} elements differ)")


def manifest_record(task):
    return {k: task.get(k) for k in ("task_id", "template", "instance", "split", "family", "ops", "spec",
                                     "signature", "header", "seed", "source")}


def manifest_hash(tasks):
    return hashlib.sha256(json.dumps([manifest_record(t) for t in tasks], sort_keys=True).encode()).hexdigest()
