"""Sequential acceptance limits and candidate consumption."""

from datetime import datetime, timedelta

from epydemix.calibration._scheduler import SequentialScheduler


def test_sequential_cutoffs_do_not_consume_extra_candidates():
    """The serial loop stays lazy, with no work after a cutoff.

    Local callbacks need no pickling. Progress is emitted after each evaluation;
    exhausted inputs and physical budgets return the accepted prefix unchanged.
    """
    consumed, progress = [], []

    def arguments():
        for index in range(4):
            consumed.append(index)
            yield (index,)

    def evaluate(index):
        return {"accepted": index % 2 == 0, "params": [index]}

    for options in (
        {"max_simulations": 0},
        {"deadline": datetime.now() - timedelta(seconds=1)},
    ):
        result = SequentialScheduler().run_until_n_accepted(
            evaluate, arguments(), 3, **options
        )
        assert result == {"accepted_results": [], "n_simulations": 0}
        assert consumed == []
    result = SequentialScheduler().run_until_n_accepted(
        evaluate,
        arguments(),
        3,
        max_simulations=3,
        progress=lambda completed, accepted: progress.append((completed, accepted)),
    )
    assert consumed == [0, 1, 2]
    assert progress == [(1, 1), (2, 1), (3, 2)]
    assert result == {
        "accepted_results": [evaluate(0), evaluate(2)],
        "n_simulations": 3,
    }
    exhausted = SequentialScheduler().run_until_n_accepted(evaluate, arguments(), 3)
    assert exhausted == {**result, "n_simulations": 4}
