from typing import Optional, Literal, TypeVar, Generator, cast
import re, sys, contextlib
from statistics import mean
import itertools as it
from dataclasses import dataclass
from joblib import Parallel, delayed
import pandas as pd
from scipy.stats import permutation_test
from scipy.stats._resampling import PermutationTestResult
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from more_itertools import flatten
from evaluate.evaluation_script import (
    evaluaterefcand,
    calculateAllScores,
    calculateSystemScore,
)
from unidecode import unidecode

Fact = list[str]
MetricMode = Literal["exact", "strict", "ent_type", "partial"]


def _cleanup_evaluaterefcand_quad(quad: list[str]) -> str:
    quad = [str(elt) if not elt is None else "" for elt in quad]
    newquad = " | ".join(quad)  # type: ignore
    newquad = re.sub(r"([a-z])([A-Z])", r"\g<1> \g<2>", newquad).lower()
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

    def scores(
        self, silent: bool = False, n_jobs: int = 1
    ) -> dict[MetricMode, list[float]]:
        if len(self.refs) == 0:
            return {"strict": [], "exact": [], "ent_type": [], "partial": []}

        # the WebNLG eval script takes quadruples with each element
        # separated with pipes
        refs = [
            [_cleanup_evaluaterefcand_quad(quad) for quad in ref] for ref in self.refs
        ]
        preds = [
            [_cleanup_evaluaterefcand_quad(quad) for quad in pred]
            for pred in self.preds
        ]

        chunk_size = len(refs) // n_jobs
        chunk_size = len(refs) if chunk_size == 0 else chunk_size
        with Parallel(n_jobs=n_jobs) as parallel:
            # [(totalsemevallist, totalsemevallistpertag, refs, preds), ...]
            #  |                                                     |
            #  <              one such tuple per chunk               >
            scores = parallel(
                delayed(calculateAllScores)(
                    refs[i : i + chunk_size], preds[i : i + chunk_size], silent
                )
                for i in range(0, len(refs), chunk_size)
            )
            totalsemevallist, totalsemevallistpertag, refs, preds = [], [], [], []
            for (
                totalsemevallist_chunk,
                totalsemevallistpertag_chunk,
                ref_chunk,
                pred_chunk,
            ) in scores:
                totalsemevallist += totalsemevallist_chunk
                totalsemevallistpertag += totalsemevallistpertag_chunk
                refs += ref_chunk
                preds += pred_chunk

        with contextlib.redirect_stdout(None):
            score_dicts, *_ = calculateSystemScore(
                totalsemevallist, totalsemevallistpertag, refs, preds
            )

        return {
            mode: [d[mode]["f1"] for d in score_dicts]
            for mode in ["strict", "exact", "ent_type", "partial"]
        }  # type: ignore


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


def mean_round(scores: list[float]) -> str:
    return str(round(mean(scores) * 100, 2))


def test_greater(scores1: list[float], scores2: list[float], **kwargs) -> float:
    res = permutation_test(
        [np.array(scores1), np.array(scores2)],
        statistic=mean_diff,
        alternative="greater",
        **kwargs,
    )
    return res.pvalue


def sigstars(
    scores1: list[float], scores2: list[float], permutation_type: str = "independent"
) -> str:
    pvalue = test_greater(scores1, scores2, permutation_type=permutation_type)
    if pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return ""


if __name__ == "__main__":
    for system, model in [
        ("edc", "mistralai:Mistral-7B-Instruct-v0.2"),
        ("baseline", "mistralai:Mistral-7B-Instruct-v0.2"),
        ("baseline", "meta-llama:Llama-3.1-8B-Instruct"),
    ]:
        print(f"==={system=} {model=}===")
        xp2022 = load_xp("yago2022:balanced-yago2026", system, model)
        xp2026 = load_xp("yago2026:balanced-yago2022", system, model)
        xp2022_2026 = load_xp(
            "yago2022:balanced-yago2026:retimestamped-2026", system, model
        )
        xp2026_2022 = load_xp(
            "yago2026:balanced-yago2022:retimestamped-2022", system, model
        )
        xp2022_multi = load_xp("yago2022_multi:balanced-yago2026_multi", system, model)
        xp2026_multi = load_xp("yago2026_multi:balanced-yago2022_multi", system, model)
        xp2022_multi_2026 = load_xp(
            "yago2022_multi:balanced-yago2026_multi:retimestamped-2026", system, model
        )
        xp2026_multi_2022 = load_xp(
            "yago2026_multi:balanced-yago2022_multi:retimestamped-2022", system, model
        )

        (
            xp2022_scores,
            xp2026_scores,
            xp2022_2026_scores,
            xp2026_2022_scores,
            xp2022_multi_scores,
            xp2026_multi_scores,
            xp2022_multi_2026_scores,
            xp2026_multi_2022_scores,
        ) = tuple(
            Parallel(n_jobs=8, return_as="generator")(
                delayed(lambda xp: xp.scores())(xp)
                for xp in [
                    xp2022,
                    xp2026,
                    xp2022_2026,
                    xp2026_2022,
                    xp2022_multi,
                    xp2026_multi,
                    xp2022_multi_2026,
                    xp2026_multi_2022,
                ]
            )
        )

        df_main = pd.DataFrame(
            {
                "dataset": [
                    "YAGO 2022",
                    "YAGO 2026",
                    "YAGO 2022 multi",
                    "YAGO 2026 multi",
                ],
                **{
                    mode: [
                        # 2022
                        (mean_round(xp2022_scores[mode]))
                        + sigstars(xp2022_scores[mode], xp2026_scores[mode]),
                        # 2026
                        mean_round(xp2026_scores[mode])
                        + sigstars(xp2026_scores[mode], xp2022_scores[mode]),
                        # 2022 multi
                        mean_round(xp2022_multi_scores[mode])
                        + sigstars(
                            xp2022_multi_scores[mode], xp2026_multi_scores[mode]
                        ),
                        # 2026 multi
                        mean_round(xp2026_multi_scores[mode])
                        + sigstars(
                            xp2026_multi_scores[mode], xp2022_multi_scores[mode]
                        ),
                    ]
                    for mode in ["strict", "exact", "ent_type", "partial"]
                },
            }
        )
        print(df_main)

        df_ts = pd.DataFrame(
            {
                "dataset": [
                    "YAGO 2022",
                    "YAGO 2022 => 2026",
                    "YAGO 2026",
                    "YAGO 2026 => 2022",
                ],
                **{
                    mode: [
                        # 2022
                        (mean_round(xp2022_scores[mode]))
                        + sigstars(xp2022_scores[mode], xp2022_2026_scores[mode]),
                        # 2022 => 2026
                        mean_round(xp2022_2026_scores[mode])
                        + sigstars(xp2022_2026_scores[mode], xp2022_scores[mode]),
                        # 2026
                        mean_round(xp2026_scores[mode])
                        + sigstars(xp2026_scores[mode], xp2026_2022_scores[mode]),
                        # 2026 => 2022
                        mean_round(xp2026_2022_scores[mode])
                        + sigstars(xp2026_2022_scores[mode], xp2026_scores[mode]),
                    ]
                    for mode in ["strict", "exact", "ent_type", "partial"]
                },
            }
        )
        print(df_ts)

        df_ts_multi = pd.DataFrame(
            {
                "dataset": [
                    "YAGO 2022 multi",
                    "YAGO 2022 multi => 2026",
                    "YAGO 2026 multi",
                    "YAGO 2026 multi => 2022",
                ],
                **{
                    mode: [
                        # 2022 multi
                        mean_round(xp2022_multi_scores[mode])
                        + sigstars(
                            xp2022_multi_scores[mode], xp2022_multi_2026_scores[mode]
                        ),
                        # 2022 multi => 2026
                        mean_round(xp2022_multi_2026_scores[mode])
                        + sigstars(
                            xp2022_multi_2026_scores[mode], xp2022_multi_scores[mode]
                        ),
                        # 2026 multi
                        mean_round(xp2026_multi_scores[mode])
                        + sigstars(
                            xp2026_multi_scores[mode], xp2026_multi_2022_scores[mode]
                        ),
                        # 2026 multi => 2022
                        mean_round(xp2026_multi_2022_scores[mode])
                        + sigstars(
                            xp2026_multi_2022_scores[mode], xp2026_multi_scores[mode]
                        ),
                    ]
                    for mode in ["strict", "exact", "ent_type", "partial"]
                },
            }
        )
        print(df_ts_multi)
