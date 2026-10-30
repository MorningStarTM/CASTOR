"""Simulator-grounded rows. The execution check, and the only decisive one.

Answers here come from running grid2op rather than from reading it, so they
are correct by construction and no teacher is involved. This is the plan's
rule applied literally: the simulator supplies the facts, the language model
supplies the prose, never the other way round.

Only facts that hold for every episode of an environment are asked about.
Nothing here touches a chronic's values, which change with the scenario and
would produce rows that are true of one episode and false of the next.

grid2op is imported lazily so the rest of the pipeline runs on a machine that
has an API key but not a simulator.
"""

from __future__ import annotations

from typing import Any, Iterator

from .schema import EXEC, EXECUTED, PASS, QARow

_COUNTS = [
    ("power lines", "n_line"),
    ("generators", "n_gen"),
    ("loads", "n_load"),
    ("substations", "n_sub"),
]

_ATTRS = [
    ("rho", "the ratio of current flow to thermal limit on each power line"),
    ("topo_vect", "the bus each grid element is connected to, 1 or 2, or -1 if disconnected"),
    ("line_status", "whether each power line is currently connected"),
    ("gen_p", "active power production of each generator, in MW"),
    ("load_p", "active power consumption of each load, in MW"),
    ("p_or", "active power flowing into the origin end of each line, in MW"),
    ("time_before_cooldown_line", "steps remaining before each line may be acted on again"),
]


def _row(number: int, question: str, answer: str, page: str,
         reasoning: str = "", depth: int = 0) -> QARow:
    checks = QARow.new_checks()
    checks["execution"] = True
    return QARow(
        id=f"g2o-{number:06d}",
        question=question,
        reasoning=reasoning,
        answer=answer,
        support="",
        grounding=EXECUTED,
        reasoning_depth=depth,
        verdict=PASS,
        checks=checks,
        reason="",
        page=page,
        refs=[],
        source=EXEC,
        model="",
        verifier="",
    )


def _facts(env: Any, obs: Any, name: str) -> Iterator[tuple[str, str, str, str, int]]:
    """(question, answer, page, reasoning, depth) for one environment."""
    for word, attr in _COUNTS:
        value = getattr(env, attr)
        yield (f"How many {word} does the grid2op environment {name} have?",
               f"{value}. In code this is env.{attr}.",
               f"{name}.env.{attr}", "", 0)
        yield (f"What does env.{attr} return for {name}?",
               f"It returns {value}, the number of {word} in the grid.",
               f"{name}.env.{attr}", "", 0)

    dim_topo = int(env.action_space.dim_topo)
    yield (f"What is dim_topo for {name}, and what does it count?",
           f"dim_topo is {dim_topo} for {name}. It counts every grid element that can "
           f"be assigned to a busbar: each line origin, each line extremity, each "
           f"generator and each load. It is therefore the length of obs.topo_vect.",
           f"{name}.env.dim_topo",
           f"Each of the {env.n_line} lines contributes two entries, one for its origin "
           f"and one for its extremity, and each of the {env.n_gen} generators and "
           f"{env.n_load} loads contributes one. "
           f"{env.n_line} x 2 + {env.n_gen} + {env.n_load} = {dim_topo}.", 2)

    for attr, meaning in _ATTRS:
        value = getattr(obs, attr, None)
        if value is None or not hasattr(value, "shape"):
            continue
        yield (f"What is the shape of obs.{attr} in {name}, and what does it hold?",
               f"obs.{attr} has shape {tuple(value.shape)} in {name}. It holds {meaning}.",
               f"{name}.obs.{attr}", "", 0)

    size = int(obs.to_vect().shape[0])
    yield (f"How many values does the flattened observation vector contain in {name}?",
           f"obs.to_vect() returns {size:,} values for {name}.",
           f"{name}.obs.to_vect", "", 0)

    yield ("What does a value of rho greater than 1.0 mean in grid2op?",
           "The line is carrying more current than its thermal limit allows. It is not "
           "disconnected immediately: an overflow timer starts, and the line trips only "
           "if the overload persists.",
           f"{name}.obs.rho",
           "rho is defined as flow divided by thermal limit, so a value above 1.0 means "
           "the numerator exceeds the denominator and the line is beyond its rating.", 1)

    yield ("In grid2op, what does the value -1 mean in obs.topo_vect?",
           "That element is disconnected. A connected element carries 1 or 2, naming "
           "which busbar at its substation it is attached to.",
           f"{name}.obs.topo_vect", "", 0)


def build(environment: str, start_number: int = 0) -> list[QARow]:
    """Every executed row for one environment, numbered from start_number."""
    import warnings

    import grid2op

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from lightsim2grid import LightSimBackend

            env = grid2op.make(environment, backend=LightSimBackend())
        except ImportError:
            # PandaPower is slower but yields identical counts and shapes,
            # which is all this module reads.
            env = grid2op.make(environment)
        obs = env.reset()

    return [
        _row(start_number + i, question, answer, page, reasoning, depth)
        for i, (question, answer, page, reasoning, depth) in enumerate(_facts(env, obs, environment))
    ]
