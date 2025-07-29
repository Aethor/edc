from hypothesis import assume, example, given, strategies as st
from string import ascii_lowercase
from datetime import datetime, timedelta
from evaluate.evaluation_script import evaluaterefcand


@st.composite
def st_timestamps(draw, start: datetime, end: datetime) -> str:
    delta = end - start
    out_date = start + timedelta(
        days=draw(st.integers(min_value=0, max_value=delta.days))
    )
    return out_date.strftime("%Y-%m-%d")


@st.composite
def st_quads(draw, **kwargs) -> str:
    sub = draw(st.text(alphabet=ascii_lowercase, **kwargs))
    pred = draw(st.text(alphabet=ascii_lowercase, **kwargs))
    obj = draw(st.text(alphabet=ascii_lowercase, **kwargs))
    ts = draw(st_timestamps(datetime(1900, 1, 1), datetime(2050, 1, 1)))
    return f"{sub} | {pred} | {obj} | {ts}"


@given(st_quads(min_size=1))
def test_perfect_cand_gives_perfect_score(ref: str):
    results, _ = evaluaterefcand(ref, ref)
    for metric_mode in ["strict", "exact", "partial", "ent_type"]:
        assert results[metric_mode]["precision"] == 1.0
        assert results[metric_mode]["recall"] == 1.0
        assert results[metric_mode]["f1"] == 1.0


def swap_attrs(quad: str, i: int, j: int) -> str:
    swapped = quad.split(" | ")
    tmp = swapped[i]
    swapped[i] = swapped[j]
    swapped[j] = tmp
    swapped = " | ".join(swapped)
    return swapped


@given(
    st_quads(min_size=1),
    st.integers(min_value=0, max_value=3),
    st.integers(min_value=0, max_value=3),
)
def test_swap_perfect_cand_attrs_gives_perfect_exact_and_partial_score(
    ref: str, i: int, j: int
):
    assume(i != j)
    cand = swap_attrs(ref, i, j)
    results, _ = evaluaterefcand(ref, cand)
    assert results["exact"]["f1"] == 1.0
    assert results["partial"]["f1"] == 1.0


@given(st_quads(min_size=1))
def test_unaligned_elements_gives_null_strict_and_type_score(ref: str):
    assume(len(set(ref.split(" | "))) == 4)
    cand = swap_attrs(ref, 0, 2)
    cand = swap_attrs(cand, 1, 3)
    results, _ = evaluaterefcand(ref, cand)
    assert results["ent_type"]["f1"] == 0.0
    assert results["strict"]["f1"] == 0.0


@given(st_quads(min_size=1), st_quads(min_size=1))
def test_score_ordering_is_correct(ref: str, cand: str):
    results, _ = evaluaterefcand(ref, cand)
    assert results["ent_type"]["f1"] >= results["strict"]["f1"]
    assert results["exact"]["f1"] >= results["strict"]["f1"]
