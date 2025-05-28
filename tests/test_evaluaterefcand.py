import os, contextlib
from hypothesis import given, strategies as st
from string import ascii_letters
from evaluate.archive import evaluaterefcand as evaluaterefcand_old
from evaluate.evaluation_script import evaluaterefcand


@given(
    st.text(alphabet=ascii_letters, min_size=1),
    st.text(alphabet=ascii_letters, min_size=1),
    st.text(alphabet=ascii_letters, min_size=1),
    st.text(alphabet=ascii_letters, min_size=1),
    st.text(alphabet=ascii_letters, min_size=1),
    st.text(alphabet=ascii_letters, min_size=1),
)
def test_triplet_same_output(
    refsub: str, refpred: str, refobj: str, predsub: str, predpred: str, predobj: str
):
    ref = f"{refsub} | {refpred} | {refobj}"
    pred = f"{predsub} | {predpred} | {predobj}"
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            old_out = evaluaterefcand_old(ref, pred)
            new_out = evaluaterefcand(ref, pred)
    assert old_out == new_out
