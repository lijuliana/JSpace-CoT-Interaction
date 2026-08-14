"""Natural-language state task: a story character's running count changes
through narrative events. Same dependency structure as arithmetic chains
(exact ground truth for every intermediate), different surface: varied
entities, verbs, and sentence forms instead of one fixed template.

Reviewer-motivated: tests whether the memory-slot results are a fact about
one arithmetic scratchpad format or about written state generally.
"""

import random

NAMES = ["Maya", "Ravi", "Lena", "Tomas", "Aisha", "Piotr", "Nadia", "Kofi"]
ITEMS = ["marbles", "stamps", "coins", "beads", "shells", "tickets"]
GAIN = ["finds {n} more", "wins {n} in a game", "is given {n} by a friend",
        "buys {n} at the market", "picks up {n} from the floor"]
LOSE = ["gives {n} to {other}", "loses {n} through a hole in the bag",
        "trades {n} away", "donates {n} to the school", "drops {n} in the river"]
TRACE_FORMS = ["After event {i}, {name} has {v} {item}.",
               "Following event {i}, {name} is left with {v} {item}.",
               "Event {i} leaves {name} with {v} {item}."]


def make_item(rng, d, j):
    name = rng.choice(NAMES)
    other = rng.choice([n for n in NAMES if n != name])
    item = rng.choice(ITEMS)
    v0 = rng.randint(100, 999)
    deltas, vals, events = [], [v0], []
    for _ in range(d):
        c = rng.choice([x for x in range(-99, 100) if abs(x) >= 10])
        # keep the count positive so the story stays coherent
        if vals[-1] + c < 10:
            c = abs(c)
        deltas.append(c)
        vals.append(vals[-1] + c)
        tmpl = rng.choice(GAIN if c >= 0 else LOSE)
        events.append(f"{name} {tmpl.format(n=abs(c), other=other)}.")
    prompt = (f"{name} starts with {v0} {item}. "
              + " ".join(f"Event {i+1}: {e}" for i, e in enumerate(events))
              + f" Track how many {item} {name} has after each event, one "
              "line per event. After the last event, give the final count "
              "on its own line as '#### <number>'.")
    lines = [rng.choice(TRACE_FORMS).format(i=i + 1, name=name,
                                            v=vals[i + 1], item=item)
             for i in range(d)]
    return {"v0": v0, "deltas": deltas, "vals": vals, "j": j,
            "answer": vals[-1], "deep_prompt": prompt, "lines": lines,
            "name": name, "item": item}


def forward_from(item, j, vj_new):
    val = vj_new
    for c in item["deltas"][j:]:
        val = val + c
    return val


def prefix(item, vj_shown):
    lines = list(item["lines"][:item["j"]])
    # rewrite the j-th line with the shown value, keeping its surface form
    lines[-1] = lines[-1].replace(str(item["vals"][item["j"]]),
                                  str(vj_shown), 1)
    return "\n".join(lines) + "\n"
