from hypothesis import given, strategies as st
from string import ascii_letters
from evaluate.scratch import evaluaterefcand_2
from evaluate.evaluation_script import evaluaterefcand


@given(
    st.text(alphabet=ascii_letters),
    st.text(alphabet=ascii_letters),
    st.text(alphabet=ascii_letters),
    st.text(alphabet=ascii_letters),
    st.text(alphabet=ascii_letters),
    st.text(alphabet=ascii_letters),
)
def test_triplet_same_output(
    refsub: str, refpred: str, refobj: str, predsub: str, predpred: str, predobj: str
):
    ref = f"{refsub} {refpred} {refobj}"
    pred = f"{predsub} {predpred} {predobj}"
    assert evaluaterefcand_2(ref, pred) == evaluaterefcand(ref, pred)
