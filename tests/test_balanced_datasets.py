from hypothesis import given, assume, strategies as st
from datetime import datetime
from balanced_datasets import balance

Fact = list[str]


@st.composite
def st_YYYY_MM_DD(draw):
    dt = draw(
        st.datetimes(min_value=datetime(1000, 1, 1), max_value=datetime(9999, 12, 31))
    )
    return dt.strftime("%Y-%m-%d")


def st_fact():
    return st.lists(st.tuples(st.text(), st.text(), st.text(), st_YYYY_MM_DD()))  # type: ignore


@given(st.lists(st.text()))
def test_balance_remove_unique(other_rels: list[str]):
    assume(all(rel != "endMemberOf" for rel in other_rels))

    fact_descs1 = ["test1"]
    ref_list1 = [[["Linus", "endMemberOf", "Linux Foundation", "2026-01-01"]]]

    fact_descs2 = ["test1"] + ["test2"] * (len(other_rels))
    ref_list2 = [
        [["Linus", "endMemberOf", "Linux Foundation", "2026-01-01"]],
    ]
    ref_list2 += [[["subj", rel, "obj", "2026-01-01"]] for rel in other_rels]

    fact_descs1, ref_list1, fact_descs2, ref_list2 = balance(
        fact_descs1, ref_list1, fact_descs2, ref_list2
    )

    assert fact_descs1 == ["test1"]
    assert ref_list1 == [[["Linus", "endMemberOf", "Linux Foundation", "2026-01-01"]]]
    assert fact_descs2 == ["test1"]
    assert ref_list2 == [[["Linus", "endMemberOf", "Linux Foundation", "2026-01-01"]]]


@given(
    st.lists(st.tuples(st.text(), st_fact())), st.lists(st.tuples(st.text(), st_fact()))
)
def test_balanced_is_smaller(
    desc_and_ref_1: tuple[list[str], list[list[Fact]]],
    desc_and_ref_2: tuple[list[str], list[list[Fact]]],
):
    fact_descs1: list[str] = [desc for desc, _ in desc_and_ref_1]  # type: ignore
    ref_list1: list[list[Fact]] = [ref for _, ref in desc_and_ref_1]  # type: ignore
    fact_descs2: list[str] = [desc for desc, _ in desc_and_ref_2]  # type: ignore
    ref_list2: list[list[Fact]] = [ref for _, ref in desc_and_ref_2]  # type: ignore

    (
        balanced_fact_descs1,
        balanced_ref_list1,
        balanced_fact_descs2,
        balanced_ref_list2,
    ) = balance(fact_descs1, ref_list1, fact_descs2, ref_list2)

    assert len(balanced_fact_descs1) <= len(fact_descs1)
    assert len(balanced_ref_list1) <= len(ref_list1)
    assert len(balanced_fact_descs2) <= len(fact_descs2)
    assert len(balanced_ref_list2) <= len(ref_list2)


@given(
    st.lists(st.tuples(st.text(), st_fact())), st.lists(st.tuples(st.text(), st_fact()))
)
def test_balanced_keeps_same_len(
    desc_and_ref_1: tuple[list[str], list[list[Fact]]],
    desc_and_ref_2: tuple[list[str], list[list[Fact]]],
):
    fact_descs1: list[str] = [desc for desc, _ in desc_and_ref_1]  # type: ignore
    ref_list1: list[list[Fact]] = [ref for _, ref in desc_and_ref_1]  # type: ignore
    fact_descs2: list[str] = [desc for desc, _ in desc_and_ref_2]  # type: ignore
    ref_list2: list[list[Fact]] = [ref for _, ref in desc_and_ref_2]  # type: ignore

    fact_descs1, ref_list1, fact_descs2, ref_list2 = balance(
        fact_descs1, ref_list1, fact_descs2, ref_list2
    )

    assert len(fact_descs1) == len(ref_list1)
    assert len(fact_descs2) == len(ref_list2)


@given(st.lists(st.tuples(st.text(), st_fact())))
def test_balanced_does_not_change_same(
    desc_and_ref: tuple[list[str], list[list[Fact]]],
):
    fact_descs: list[str] = [desc for desc, _ in desc_and_ref]  # type: ignore
    ref_list: list[list[Fact]] = [ref for _, ref in desc_and_ref]  # type: ignore
    fact_descs1, ref_list1, fact_descs2, ref_list2 = balance(
        fact_descs, ref_list, fact_descs, ref_list
    )
    assert fact_descs1 == fact_descs2
    assert ref_list1 == ref_list2
