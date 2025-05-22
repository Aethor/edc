from hypothesis import given, strategies as st
from string import ascii_letters
from edc.utils.llm_utils import parse_raw_quadruples, parse_raw_entities


@given(st.lists(st.text(alphabet=ascii_letters, min_size=1), min_size=4, max_size=4))
def test_parse_trivial_quadruples(quad: list[str]):
    assert len(quad) == 4
    parsed = parse_raw_quadruples(str(quad))
    assert parsed.__class__ == list
    assert len(parsed) == 1  # only 1 quadruple
    assert len(parsed[0]) == 4  # the quadruple has 4 elements


@given(st.lists(st.text(alphabet=ascii_letters)))
def test_parse_raw_entities(raw_entities_lst: list[str]):
    parsed = parse_raw_entities(str(raw_entities_lst))
    assert parsed.__class__ == list
    assert len(parsed) == len(raw_entities_lst)
