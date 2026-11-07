from __future__ import annotations

import math

from hypothesis import given
from hypothesis import strategies as st

from bonetwin.bounds.conditional import dkw_union_gap, gap_within_bound
from bonetwin.bounds.synergy import compositional_interaction, synergy_lower_bound_holds


def test_theorem2_matches_closed_form() -> None:
    gap = dkw_union_gap(n_strata=4, n_min=483, delta=0.05)
    expected = math.sqrt(math.log(2.0 * 4 / 0.05) / (2.0 * 483))
    assert math.isclose(gap, expected)
    assert gap_within_bound(gap, 4, 483, 0.05)


@given(
    n_strata=st.integers(min_value=2, max_value=20),
    n_min=st.integers(min_value=100, max_value=5000),
    delta=st.sampled_from([0.01, 0.05, 0.10]),
)
def test_theorem2_bound_is_decreasing_in_nmin(n_strata: int, n_min: int, delta: float) -> None:
    coarse = dkw_union_gap(n_strata, n_min, delta)
    finer = dkw_union_gap(n_strata, n_min * 2, delta)
    assert finer <= coarse
    assert gap_within_bound(finer, n_strata, n_min * 2, delta)


def test_theorem3_synergy_value_and_sign() -> None:
    interaction = compositional_interaction(0.859, 0.728, [0.025, 0.035, 0.007])
    assert interaction > 0.0
    assert synergy_lower_bound_holds(0.859, 0.728, [0.025, 0.035, 0.007])


@given(
    base=st.floats(min_value=0.5, max_value=0.7),
    d1=st.floats(min_value=0.0, max_value=0.1),
    d2=st.floats(min_value=0.0, max_value=0.1),
)
def test_theorem3_lower_bound_holds_when_full_exceeds_parts(
    base: float, d1: float, d2: float
) -> None:
    full = base + d1 + d2 + 0.01
    assert synergy_lower_bound_holds(full, base, [d1, d2])
