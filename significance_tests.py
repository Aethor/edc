from typing import Optional, Literal, TypeVar, Generator, cast
import re
import itertools as it
from dataclasses import dataclass
from scipy.stats import permutation_test
from scipy.stats._resampling import PermutationTestResult
import numpy as np
from tqdm import tqdm
from evaluate.evaluation_script import evaluaterefcand
from unidecode import unidecode

Fact = list[Optional[str]]
MetricMode = Literal["exact", "strict", "partial", "ent_type"]

T = TypeVar("T")
S = TypeVar("S")


def mappings(lst1: list[T], lst2: list[S]) -> Generator[tuple[tuple[T, S]], None, None]:
    for perm in it.permutations(lst2, len(lst1)):
        yield tuple(zip(lst1, perm))  # type: ignore


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
class XPExample:
    text: str
    ref: list[Fact]
    pred: list[Fact]

    def score(self) -> list[dict[MetricMode, float]]:
        # equalize the length of ref/pred. This is done so that it it
        # possible to map each ref quadruple to each pred quadruple
        # one-to-one.
        ref = self.ref + [""] * (len(self.pred) - len(self.ref))
        pred = self.pred + [""] * (len(self.ref) - len(self.pred))

        # pred has possible 'None' in its quadruples. Transform them
        # to "" to match the original implementation
        pred = [[elt if not elt is None else "" for elt in fact] for fact in pred]

        best_scores = []
        best_mean = 0.0
        cache = {}
        for mapping in mappings(ref, pred):
            mapping_scores = []
            for ref_fact, pred_fact in mapping:
                cache_key = (tuple(ref_fact), tuple(pred_fact))
                if cache_key in cache:
                    results = cache[cache_key]
                else:
                    try:
                        results, _ = evaluaterefcand(
                            _cleanup_evaluaterefcand_quad(" | ".join(ref_fact)),
                            _cleanup_evaluaterefcand_quad(" | ".join(pred_fact)),
                        )
                        cache[cache_key] = results
                    except TypeError:
                        continue
                mapping_scores.append(
                    {
                        "exact": results["exact"]["f1"],
                        "strict": results["strict"]["f1"],
                        "partial": results["partial"]["f1"],
                        "ent_type": results["ent_type"]["f1"],
                    }
                )

            mean = sum(sum(score.values()) for score in mapping_scores)
            if mean >= best_mean:
                best_mean = mean
                best_scores = mapping_scores

        return best_scores


@dataclass
class XP:
    texts: list[str]
    refs: list[list[Fact]]
    preds: list[list[Fact]]

    def examples(self) -> Generator[XPExample, None, None]:
        for text, ref, pred in zip(self.texts, self.refs, self.preds):
            yield XPExample(text, ref, pred)

    def __len__(self) -> int:
        assert len({len(self.texts), len(self.refs), len(self.preds)}) == 1
        return len(self.texts)

    def flattened_scores(self) -> Generator[dict[MetricMode, float], None, None]:
        for ex in self.examples():
            for score_dict in ex.score():
                yield score_dict


def load_xp(name: str) -> XP:
    texts = []
    refs = []
    preds = []
    with open(f"./dsets/{name}.txt") as f:
        for line in f:
            texts.append(line.strip("\n"))
    with open(f"./evaluate/references/{name}.txt") as f:
        for line in f:
            refs.append(eval(line))
    with open(f"./output/{name}_target_alignment/iter0/canon_kg.txt") as f:
        for line in f:
            preds.append(eval(line))
    return XP(texts, refs, preds)


# %%


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


xp2022 = load_xp("yago2022:balanced-yago2026")
xp2026 = load_xp("yago2026:balanced-yago2022")

xp2022_scores = list(tqdm(list(xp2022.flattened_scores()), ascii=True))
xp2026_scores = list(tqdm(list(xp2026.flattened_scores()), ascii=True))
print("2022 > 2026 ?")
print(test_greater(xp2022_scores, xp2026_scores))

xp2022_multi = load_xp("yago2022_multi:balanced-yago2026_multi")
xp2026_multi = load_xp("yago2026_multi:balanced-yago2022_multi")
xp2022_multi_scores = list(tqdm(list(xp2022_multi.flattened_scores()), ascii=True))
xp2026_multi_scores = list(tqdm(list(xp2026_multi.flattened_scores()), ascii=True))
print("2022_multi > 2026_multi ?")
print(test_greater(xp2022_multi_scores, xp2026_multi_scores))

xp2022_2026 = load_xp("yago2022:balanced-yago2026:retimestamped-2026")
xp2022_2026_scores = list(tqdm(list(xp2022_2026.flattened_scores()), ascii=True))
print("2022 > 2022->2026 ?")
print(test_greater(xp2022_scores, xp2022_2026_scores))

xp2026_2022 = load_xp("yago2026:balanced-yago2022:retimestamped-2022")
xp2026_2022_scores = list(tqdm(list(xp2026_2022.flattened_scores()), ascii=True))
print("2026->2022 > 2026 ?")
print(test_greater(xp2026_2022_scores, xp2026_scores))

# %%
xp2022_multi = load_xp("yago2022_multi:balanced-yago2026_multi")
xp2022_multi_2026 = load_xp("yago2022_multi:balanced-yago2026_multi:retimestamped-2026")
xp2022_multi_scores = list(tqdm(list(xp2022_multi.flattened_scores()), ascii=True))
xp2022_multi_2026_scores = list(
    tqdm(list(xp2022_multi_2026.flattened_scores()), ascii=True)
)
print("2022_multi->2026 > 2022_multi ?")
print(test_greater(xp2022_multi_2026_scores, xp2022_multi_scores))

xp2026_multi = load_xp("yago2026_multi:balanced-yago2022_multi")
xp2026_multi_2022 = load_xp("yago2026_multi:balanced-yago2022_multi:retimestamped-2022")
xp2026_multi_scores = list(tqdm(list(xp2026_multi.flattened_scores()), ascii=True))
xp2026_multi_2022_scores = list(
    tqdm(list(xp2026_multi_2022.flattened_scores()), ascii=True)
)
print("2026_multi->2022 > 2026_multi ?")
print(test_greater(xp2026_multi_2022_scores, xp2026_multi_scores))
