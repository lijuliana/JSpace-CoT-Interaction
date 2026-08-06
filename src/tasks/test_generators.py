"""Self-tests for the task generators and the corruption math.

The load-bearing check: for every instance, replaying the chain from any
intermediate's clean value must reproduce the instance answer. This ties
gate B's forward_from to the generators; if either changes shape, this
fails before any GPU time is spent.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tasks.generators import (mod_arithmetic, variable_chain,  # noqa: E402
                              entity_tracking, dag_reachability)
from harness.gate_b_corruption import forward_from  # noqa: E402


def test_forward_from_consistency():
    for d in [2, 4, 8, 16]:
        for seed in range(25):
            for gen in (mod_arithmetic, variable_chain):
                inst = gen(d, seed)
                for idx, (name, value) in enumerate(inst.intermediates):
                    got = forward_from(inst, idx, int(value))
                    assert got == inst.answer, (
                        f"{inst.family} d={d} seed={seed} idx={idx}: "
                        f"replay gave {got}, answer {inst.answer}")


def test_mod_arithmetic_steps_match_prompt():
    inst = mod_arithmetic(6, 3)
    assert "Step 6" in inst.prompt and "Step 7" not in inst.prompt
    assert len(inst.intermediates) == 6
    assert inst.intermediates[-1][1] == inst.answer


def test_variable_chain_range():
    inst = variable_chain(8, 1)
    first = int(inst.intermediates[0][1])
    assert 100 <= first <= 999
    assert inst.intermediates[-1][1] == inst.answer


def test_entity_tracking_values_are_objects():
    inst = entity_tracking(5, 6, 2)
    for name, value in inst.intermediates:
        assert value.isalpha(), value
    assert inst.answer.isalpha()


def test_dag_has_dead_ends_but_unique_full_path():
    for seed in range(30):
        inst = dag_reachability(5, seed)
        edges = set()
        for pair in inst.prompt.split(": ")[1].split(". ")[0].split(", "):
            a, b = pair.split("->")
            edges.add((a, b))
        # exactly one full path: walk all routes of length hops from source
        path = inst.meta["path"]
        frontier = [path[0]]
        for _ in range(5):
            frontier = [b for a, b in edges for x in frontier if a == x]
        assert frontier.count(path[-1]) == 1
        assert len(set(frontier)) == 1
        # at least some instances give path nodes an off-path choice
    branching = 0
    for seed in range(30):
        inst = dag_reachability(5, seed)
        edges = set()
        for pair in inst.prompt.split(": ")[1].split(". ")[0].split(", "):
            a, b = pair.split("->")
            edges.add((a, b))
        outdeg = {}
        for a, b in edges:
            outdeg[a] = outdeg.get(a, 0) + 1
        if any(outdeg.get(n, 0) > 1 for n in inst.meta["path"][:-1]):
            branching += 1
    assert branching > 10, f"only {branching}/30 instances have branching"


if __name__ == "__main__":
    for fn in [test_forward_from_consistency,
               test_mod_arithmetic_steps_match_prompt,
               test_variable_chain_range,
               test_entity_tracking_values_are_objects,
               test_dag_has_dead_ends_but_unique_full_path]:
        fn()
        print(f"{fn.__name__} ok")
