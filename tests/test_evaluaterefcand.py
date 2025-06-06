import os, contextlib
from hypothesis import given, strategies as st
from string import ascii_letters
from evaluate.archive import evaluaterefcand as evaluaterefcand_old
from evaluate.evaluation_script import evaluaterefcand


@st.composite
def st_quad(draw, **kwargs) -> str:
    sub = draw(st.text(alphabet=ascii_letters, **kwargs))
    pred = draw(st.text(alphabet=ascii_letters, **kwargs))
    obj = draw(st.text(alphabet=ascii_letters, **kwargs))
    ts = draw(st.text(alphabet=ascii_letters, **kwargs))
    return f"{sub} | {pred} | {obj} | {ts}"


@st.composite
def st_triple(draw, **kwargs) -> str:
    sub = draw(st.text(alphabet=ascii_letters, **kwargs))
    pred = draw(st.text(alphabet=ascii_letters, **kwargs))
    obj = draw(st.text(alphabet=ascii_letters, **kwargs))
    return f"{sub} | {pred} | {obj}"


@given(st_triple(min_size=1), st_triple(min_size=1))
def test_retrocompatible(ref: str, cand: str):
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            old_out = evaluaterefcand_old(ref, cand)
            new_out = evaluaterefcand(ref, cand)
    assert old_out == new_out


# @given(st_quad())
# def test_identical_output_has_perfect_score(quad: str):
#     results, result_per_tag = evaluaterefcand(quad, quad)
#     for metric_variant in ["ent_type", "partial", "strict", "exact"]:
#         assert results[metric_variant]["f1"] == 1.0


# @given(st_quad(), st_quad())
# def test_quad(ref: str, cand: str):
#     assert evaluaterefcand(ref, cand)
