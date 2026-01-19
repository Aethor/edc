from typing import Optional, Literal, TypeVar, Generator, cast
import re, sys, contextlib
import itertools as it
from dataclasses import dataclass
from scipy.stats import permutation_test
from scipy.stats._resampling import PermutationTestResult
import numpy as np
from tqdm import tqdm
from evaluate.evaluation_script import (
    evaluaterefcand,
    calculateAllScores,
    calculateSystemScore,
)
from unidecode import unidecode

Fact = list[Optional[str]]
MetricMode = Literal["exact", "strict", "ent_type", "partial"]


def _cleanup_evaluaterefcand_quad(quad: str) -> str:
    newquad = re.sub(r"([a-z])([A-Z])", r"\g<1> \g<2>", quad).lower()
    newquad = re.sub(r"_", " ", newquad).lower()
    newquad = re.sub(r"\s+", " ", newquad).lower()
    newquad = unidecode(newquad)
    adjusttriple = newquad.split(" | ")
    manualmodified = re.search(r"^(.*?)(\s\((.*?)\))$", adjusttriple[-1])
    if manualmodified:
        adjusttriple[-1] = manualmodified.group(1)
        newquad = " | ".join(adjusttriple)
    return newquad


@dataclass
class XP:
    texts: list[str]
    refs: list[list[Fact]]
    preds: list[list[Fact]]

    def scores(self) -> list[dict[MetricMode, float]]:
        # the WebNLG eval script takes quadruples with each element
        # separated with pipes
        refs = [
            [_cleanup_evaluaterefcand_quad(" | ".join(quad)) for quad in ref]
            for ref in self.refs
        ]
        preds = [
            [_cleanup_evaluaterefcand_quad(" | ".join(quad)) for quad in pred]
            for pred in self.preds
        ]
        totalsemevallist, totalsemevallistpertag = calculateAllScores(refs, preds)
        with contextlib.redirect_stdout(None):
            score_dicts, *_ = calculateSystemScore(
                totalsemevallist, totalsemevallistpertag, refs, preds
            )
        return [
            {
                "exact": (d["exact"]["f1"]),
                "strict": d["strict"]["f1"],
                "partial": d["partial"]["f1"],
                "ent_type": d["ent_type"]["f1"],
            }
            for d in score_dicts
        ]  # type: ignore


def load_xp(name: str, system: str, model: str) -> XP:
    texts = []
    refs = []
    preds = []
    with open(f"./dsets/{name}.txt") as f:
        for line in f:
            texts.append(line.strip("\n"))
    with open(f"./evaluate/references/{name}.txt") as f:
        for line in f:
            refs.append(eval(line))
    pred_path = f"./output/{system}/{model}/{name}_target_alignment/iter0/canon_kg.txt"
    with open(pred_path) as f:
        for line in f:
            try:
                preds.append(eval(line))
            except TypeError:
                print(f"{pred_path=} error while loading line : {line=}")
                continue
    return XP(texts, refs, preds)


def mean_diff(arr1: np.ndarray, arr2: np.ndarray, axis: int) -> float:
    return np.mean(arr1, axis=axis) - np.mean(arr2, axis=axis)


def test_greater(
    scores1: list[dict[MetricMode, float]], scores2: list[dict[MetricMode, float]]
) -> dict[MetricMode, float]:
    res_dict = {}
    for mode in ["exact", "strict", "ent_type", "partial"]:
        mode = cast(MetricMode, mode)
        res = permutation_test(
            [
                np.array([s[mode] for s in scores1]),
                np.array([s[mode] for s in scores2]),
            ],
            statistic=mean_diff,
            alternative="greater",
        )
        res_dict[mode] = res.pvalue
    return res_dict


if __name__ == "__main__":
    for system, model in [
        ("edc", "mistralai:Mistral-7B-Instruct-v0.2"),
        ("baseline", "mistralai:Mistral-7B-Instruct-v0.2"),
        ("baseline", "meta-llama:Llama-3.1-8B"),
    ]:
        print(f"==={system=} {model=}===")
        xp2022 = load_xp("yago2022:balanced-yago2026", system, model)
        xp2026 = load_xp("yago2026:balanced-yago2022", system, model)

        xp2022_scores = xp2022.scores()
        xp2026_scores = xp2026.scores()
        print("2022 > 2026 ?")
        print(test_greater(xp2022_scores, xp2026_scores))

        xp2022_multi = load_xp("yago2022_multi:balanced-yago2026_multi", system, model)
        xp2026_multi = load_xp("yago2026_multi:balanced-yago2022_multi", system, model)
        xp2022_multi_scores = xp2022_multi.scores()
        xp2026_multi_scores = xp2026_multi.scores()
        print("2022_multi > 2026_multi ?")
        print(test_greater(xp2022_multi_scores, xp2026_multi_scores))

        xp2022_2026 = load_xp(
            "yago2022:balanced-yago2026:retimestamped-2026", system, model
        )
        xp2022_2026_scores = xp2022_2026.scores()
        print("2022 > 2022->2026 ?")
        print(test_greater(xp2022_scores, xp2022_2026_scores))

        xp2026_2022 = load_xp(
            "yago2026:balanced-yago2022:retimestamped-2022", system, model
        )
        xp2026_2022_scores = xp2026_2022.scores()
        print("2026->2022 > 2026 ?")
        print(test_greater(xp2026_2022_scores, xp2026_scores))

        xp2022_multi = load_xp("yago2022_multi:balanced-yago2026_multi", system, model)
        xp2022_multi_2026 = load_xp(
            "yago2022_multi:balanced-yago2026_multi:retimestamped-2026", system, model
        )
        xp2022_multi_scores = xp2022_multi.scores()
        xp2022_multi_2026_scores = xp2022_multi_2026.scores()
        print("2022_multi->2026 > 2022_multi ?")
        print(test_greater(xp2022_multi_2026_scores, xp2022_multi_scores))

        xp2026_multi = load_xp("yago2026_multi:balanced-yago2022_multi", system, model)
        xp2026_multi_2022 = load_xp(
            "yago2026_multi:balanced-yago2022_multi:retimestamped-2022", system, model
        )
        xp2026_multi_scores = xp2026_multi.scores()
        xp2026_multi_2022_scores = xp2026_multi_2022.scores()
        print("2026_multi->2022 > 2026_multi ?")
        print(test_greater(xp2026_multi_2022_scores, xp2026_multi_scores))
