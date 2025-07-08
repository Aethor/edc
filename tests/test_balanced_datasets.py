from hypothesis import given, assume, strategies as st
from balanced_datasets import balance

Facts = list[str]


def test_balance_trivial():
    fact_descs1 = ["test1"]
    ref_list1 = [[["Linus", "endMemberOf", "Linux Foundation", "2026-01-01"]]]
    # the second fact should be eliminated since startMemberOf does
    # not appear in ref_list1
    fact_descs2 = ["test1", "test2"]
    ref_list2 = [
        [["Linus", "endMemberOf", "Linux Foundation", "2026-01-01"]],
        [["Bill", "startMemberOf", "Linux Foundation", "2026-01-01"]],
    ]
    fact_descs1, ref_list1, fact_descs2, ref_list2 = balance(
        fact_descs1, ref_list1, fact_descs2, ref_list2
    )
    assert fact_descs1 == ["test1"]
    assert ref_list1 == [[["Linus", "endMemberOf", "Linux Foundation", "2026-01-01"]]]
    assert fact_descs2 == ["test1"]
    assert ref_list2 == [[["Linus", "endMemberOf", "Linux Foundation", "2026-01-01"]]]


@given(
    st.lists(
        st.tuples(
            st.text(), st.lists(st.tuples(st.text(), st.text(), st.text(), st.text()))
        )
    ),
    st.lists(
        st.tuples(
            st.text(), st.lists(st.tuples(st.text(), st.text(), st.text(), st.text()))
        ),
    ),
)
def test_balanced_is_smaller(
    desc_and_ref_1: tuple[list[str], list[list[Facts]]],
    desc_and_ref_2: tuple[list[str], list[list[Facts]]],
):
    fact_descs1 = [desc for desc, _ in desc_and_ref_1]
    ref_list1 = [ref for _, ref in desc_and_ref_1]
    fact_descs2 = [desc for desc, _ in desc_and_ref_2]
    ref_list2 = [ref for _, ref in desc_and_ref_2]
    (
        balanced_fact_descs1,
        balanced_ref_list1,
        balanced_fact_descs2,
        balanced_ref_list2,
    ) = balance(fact_descs1, ref_list1, fact_descs2, ref_list2)  # type: ignore
    assert len(balanced_fact_descs1) <= len(fact_descs1)
    assert len(balanced_ref_list1) <= len(ref_list1)
    assert len(balanced_fact_descs2) <= len(fact_descs2)
    assert len(balanced_ref_list2) <= len(ref_list2)
