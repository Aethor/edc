import argparse, ast, copy
import pathlib as pl
import itertools as it
from collections import defaultdict
from statistics import mean
from datetime import date
from collections import Counter
from balanced_datasets import downsample_rel
from significance_tests import XP, load_xp

Fact = list[str]


def get_year(ts: str) -> int:
    return int(date.fromisoformat(ts).strftime("%Y"))


def filter_xp_year(xp: XP, year: int) -> XP:
    new_xp = copy.deepcopy(xp)
    new_xp.refs = [facts for facts in xp.refs if get_year(facts[0][3]) == year]
    new_xp.preds = [
        preds
        for preds, facts in zip(xp.preds, xp.refs)
        if get_year(facts[0][3]) == year
    ]
    new_xp.texts = [
        text for text, facts in zip(xp.texts, xp.refs) if get_year(facts[0][3]) == year
    ]
    return new_xp


def downsample_xp_rel(xp: XP, rel_limit: dict[str, int]) -> XP:
    rel_counter = defaultdict(int)
    downsampled_texts = []
    downsampled_refs = []
    downsampled_preds = []

    for text, ref, pred in zip(xp.texts, xp.refs, xp.preds):
        fact_rel_counter = Counter([rel for _, rel, _, _ in ref])
        if any(
            rel_counter[rel] + count > rel_limit[rel]
            for rel, count in fact_rel_counter.items()
        ):
            continue
        downsampled_texts.append(text)
        downsampled_refs.append(ref)
        downsampled_preds.append(pred)
        for rel, count in fact_rel_counter.items():
            rel_counter[rel] += count

    return XP(downsampled_texts, downsampled_refs, downsampled_preds)


def balance_xp(xp1: XP, xp2: XP) -> tuple[XP, XP]:
    rel_counter_1 = Counter([rel for quads in xp1.refs for _, rel, _, _ in quads])
    rel_counter_2 = Counter([rel for quads in xp2.refs for _, rel, _, _ in quads])
    all_rels = set(rel_counter_1.keys()).union(set(rel_counter_2.keys()))
    rel_limit = {
        rel: min(rel_counter_1.get(rel, 0), rel_counter_2.get(rel, 0))
        for rel in all_rels
    }

    xp1 = downsample_xp_rel(xp1, rel_limit)
    xp2 = downsample_xp_rel(xp2, rel_limit)

    return (xp1, xp2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n1", "--xp1-name", type=str)
    parser.add_argument("-s1", "--xp1-system", type=str)
    parser.add_argument("-m1", "--xp1-model", type=str)
    parser.add_argument("-n2", "--xp2-name", type=str)
    parser.add_argument("-s2", "--xp2-system", type=str)
    parser.add_argument("-m2", "--xp2-model", type=str)
    args = parser.parse_args()

    xp1 = load_xp(args.xp1_name, args.xp1_system, args.xp1_model)
    xp2 = load_xp(args.xp2_name, args.xp2_system, args.xp2_model)

    year_list1 = sorted({get_year(ts) for ref in xp1.refs for _, _, _, ts in ref})
    year_list2 = sorted({get_year(ts) for ref in xp2.refs for _, _, _, ts in ref})
    for year1, year2 in it.product(year_list1, year_list2):
        year_xp1 = filter_xp_year(xp1, year1)
        year_xp2 = filter_xp_year(xp2, year2)
        year_xp1, year_xp2 = balance_xp(year_xp1, year_xp2)
        print(f"{args.xp1_name} {args.xp1_system} {args.xp1_model} {year1}")
        print({k: mean(v) for k, v in year_xp1.scores().items()})
        print(f"{args.xp2_name} {args.xp2_system} {args.xp2_model} {year2}")
        print({k: mean(v) for k, v in year_xp2.scores().items()})
