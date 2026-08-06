"""Deterministic worked-trace construction for variable-chain instances.
Torch-free so it can be unit tested and shared across probe/patch harnesses.
"""


def build_trace(inst, write_target=True):
    """Return (text, step_end_char_positions, target_op_index).

    The trace shows each step as 'name = prev op arg = value.'. The target
    step (op index depth//2) either shows its result value or elides it
    (keeping the operation), so writing of that one value becomes an
    exogenous variable. step_end_char_positions[i] is the char offset at the
    end of step i's line (after ops[i], producing intermediate i+1).
    """
    ops = inst.meta["ops"]
    names = [n for n, _ in inst.intermediates]
    vals = [int(v) for _, v in inst.intermediates]
    tgt = len(ops) // 2
    text = f"{names[0]} = {vals[0]}."
    ends = []
    for i, (op, arg) in enumerate(ops):
        if i == tgt and not write_target:
            line = f" {names[i+1]} = {names[i]} {op} {arg}."
        else:
            line = f" {names[i+1]} = {names[i]} {op} {arg} = {vals[i+1]}."
        text += line
        ends.append(len(text))
    return text, ends, tgt
