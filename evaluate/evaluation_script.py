# import warnings filter
from warnings import simplefilter
from sklearn.exceptions import UndefinedMetricWarning

# ignore all UndefinedMetricWarning warnings
simplefilter(action="ignore", category=UndefinedMetricWarning)
from typing import Literal, Tuple, List, TypedDict
import itertools as it
from bs4 import BeautifulSoup
import os
import regex as re
import functools as ft
from operator import add
import itertools
import statistics
from nervaluate import Evaluator
import nltk
from nltk.util import ngrams
import string
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import preprocessing
from unidecode import unidecode
from argparse import ArgumentParser
import os
import ast
import xml.etree.ElementTree as ET
from dataclasses import dataclass


AttrType = Literal["SUB", "PRED", "OBJ"]
yn = Literal["y", "n"]


def convert_to_xml(result_path: str, gold_path: str, max_length_diff=None):
    output_dir = result_path.split(os.sep)[-2]
    os.makedirs(f"./result_xmls/{output_dir}", exist_ok=True)
    pred_xml_path = os.path.join(f"./result_xmls/{output_dir}", f"pred.xml")
    ref_xml_path = os.path.join(f"./result_xmls/{output_dir}", f"ref.xml")

    pred_triplets = [l.strip() for l in open(result_path, "r").readlines()]
    gold_triplets = [l.strip() for l in open(gold_path, "r").readlines()]

    collected_pred_triplets = []
    collected_gold_triplets = []

    for idx, triplets in enumerate(pred_triplets):
        try:
            evaled_triplets = ast.literal_eval(triplets)
            for triplet in evaled_triplets:
                if len(triplet) < 3 or len(triplet) > 4:
                    raise Exception
                for element in triplet:
                    if not isinstance(element, str):
                        raise Exception

            collected_pred_triplets.append(evaled_triplets)
            collected_gold_triplets.append(ast.literal_eval(gold_triplets[idx]))
        except Exception:
            pass

    assert len(collected_pred_triplets) == len(collected_gold_triplets)

    pred_root_node = ET.Element("benchmark")
    pred_entries_node = ET.SubElement(pred_root_node, "entries")

    gold_root_node = ET.Element("benchmark")
    gold_entries_node = ET.SubElement(gold_root_node, "entries")

    # Iterate over the elements
    skipped = 0
    collected = 0

    for idx in range(len(collected_gold_triplets)):
        length_diff = abs(
            len(collected_gold_triplets[idx]) - len(collected_pred_triplets[idx])
        )

        if max_length_diff is not None:
            if length_diff > int(max_length_diff):
                skipped += 1
                continue

        pred_entry_node = ET.SubElement(pred_entries_node, "entry")
        pred_generated_tripleset = ET.SubElement(pred_entry_node, "generatedtripleset")
        for triplet in collected_pred_triplets[idx]:
            gtriplet_node = ET.SubElement(pred_generated_tripleset, "gtriple")
            gtriplet_node.text = " | ".join(triplet)

        gold_entry_node = ET.SubElement(gold_entries_node, "entry")
        gold_reference_tripleset = ET.SubElement(gold_entry_node, "modifiedtripleset")
        for triplet in collected_gold_triplets[idx]:
            rtriplet_node = ET.SubElement(gold_reference_tripleset, "mtriple")
            rtriplet_node.text = " | ".join(triplet)

        collected += 1

    pred_tree = ET.ElementTree(pred_root_node)
    ET.indent(pred_tree, space="\t", level=0)
    pred_tree.write(pred_xml_path)

    gold_tree = ET.ElementTree(gold_root_node)
    ET.indent(gold_tree, space="\t", level=0)
    gold_tree.write(ref_xml_path)

    return pred_xml_path, ref_xml_path


currentpath = os.getcwd()


def getText(filepath):
    with open(filepath, encoding="utf-8") as fp:
        refssoup = BeautifulSoup(fp, "lxml")

    entries = refssoup.find("benchmark").find("entries").find_all("entry")
    texts = []
    for entry in entries:
        texts.append(entry.find("text").text)
    return texts


def getRefs(filepath) -> Tuple[List[List[str]], List[List[str]]]:
    with open(filepath, encoding="utf-8") as fp:
        refssoup = BeautifulSoup(fp, "lxml")

    refsentries = refssoup.find("benchmark").find("entries").find_all("entry")

    allreftriples = []

    for entry in refsentries:
        entryreftriples = []
        modtriplesref = entry.find("modifiedtripleset").find_all("mtriple")
        for modtriple in modtriplesref:
            entryreftriples.append(modtriple.text)
        allreftriples.append(entryreftriples)

    newreflist = []

    for entry in allreftriples:
        newtriples = []
        for triple in entry:
            newtriple = re.sub(r"([a-z])([A-Z])", r"\g<1> \g<2>", triple).lower()
            newtriple = re.sub(r"_", " ", newtriple).lower()
            newtriple = re.sub(r"\s+", " ", newtriple).lower()

            newtriple = unidecode(newtriple)
            adjusttriple = newtriple.split(" | ")
            manualmodified = re.search(r"^(.*?)(\s\((.*?)\))$", adjusttriple[-1])
            if manualmodified:
                adjusttriple[-1] = manualmodified.group(1)
                newtriple = " | ".join(adjusttriple)
            newtriples.append(newtriple)
        newreflist.append(newtriples)

    return allreftriples, newreflist


def getCands(filepath) -> Tuple[List[List[str]], List[List[str]]]:
    with open(filepath, encoding="utf-8") as fp:
        candssoup = BeautifulSoup(fp, "lxml")

    candssentries = candssoup.find("benchmark").find("entries").find_all("entry")

    allcandtriples = []

    for entry in candssentries:
        entrycandtriples = []
        modtriplescand = entry.find("generatedtripleset").find_all("gtriple")
        for modtriple in modtriplescand:
            entrycandtriples.append(modtriple.text)
        allcandtriples.append(entrycandtriples)

    newcandlist = []

    for entry in allcandtriples:
        newtriples = []
        for triple in entry:
            newtriple = re.sub(r"([a-z])([A-Z])", r"\g<1> \g<2>", triple).lower()
            newtriple = re.sub(r"_", " ", newtriple).lower()
            newtriple = re.sub(r"\s+", " ", newtriple).lower()
            newtriple = unidecode(newtriple)
            adjusttriple = newtriple.split(" | ")
            manualmodified = re.search(r"^(.*?)(\s\((.*?)\))$", adjusttriple[-1])
            if manualmodified:
                adjusttriple[-1] = manualmodified.group(1)
                newtriple = " | ".join(adjusttriple)
            newtriples.append(newtriple)
        newcandlist.append(newtriples)

    return allcandtriples, newcandlist


def find_sub_list(sl, l):
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind : ind + sll] == sl:
            return ind, ind + sll - 1


def nonrefwords(
    newreflist: list[str], newcandlist: list[str], foundnum: int, ngramlength: int
) -> tuple[list[str], list[str]]:
    """We are going to try to find matches with the reference,
    starting with the highest chunk possible (all the words in the
    reference).  If we don't find that, we are going to search for all
    n-grams -1 the number of words in the reference; than -2; than -3;
    etc.

    >>> nonrefwords("this is a test".split(), "this is a".split(), 1, 2)
    (['FOUNDREF-1-0', 'FOUNDREF-1-1', 'FOUNDREF-2-2', 'test'], ['FOUNDCAND-1-0', 'FOUNDCAND-1-1', 'FOUNDCAND-2-2'])

    :param newreflist: a list of tokens
    :param newcandlist: a list of tokens
    :param foundnum: recursive argument, starts at 1
    :param ngramlength:

    :return: two list of strings:
        - one for NEWREFLIST. Each token is either one of the original tokens
          (not matched) or of the form FOUNDREF-X-Y where X is index of the
          matched ngram and Y the index of the token in that ngram.
        - one for NEWCANDLIST. Similarly as above, but with matched tokens of
          the form FOUNDCAND-X-Y
    """
    while ngramlength > 0:
        # Get a list of all the ngrams of that size
        ngramlist = list(ngrams(newcandlist, ngramlength))
        for ngram in ngramlist:
            # If we find this ngram (in the same order) in the reference
            if find_sub_list(list(ngram), newreflist) is not None:
                # We're getting the start and end index of the ngram in the reference
                findnewref = find_sub_list(list(ngram), newreflist)
                # And all the numbers in between
                newrefindex = list(range(findnewref[0], findnewref[1] + 1))
                # Change the matched words to FOUNDREF-[FOUNDNUMBER]-[FOUNDINDEX]
                for idx in newrefindex:
                    newreflist[idx] = "FOUNDREF-" + str(foundnum) + "-" + str(idx)

                # Now find the start and end index of the ngram in the candidate as well
                findnewcand = find_sub_list(list(ngram), newcandlist)
                # And all the indices in between
                newcandindex = list(range(findnewcand[0], findnewcand[1] + 1))
                # Change the matched words to FOUNDCAND-[FOUNDNUMBER]-[REFERENCE-FOUNDINDEX]
                for idx, val in enumerate(newcandindex):
                    newcandlist[val] = (
                        "FOUNDCAND-" + str(foundnum) + "-" + str(newrefindex[idx])
                    )
                foundnum += 1
                # And try to find new matches again
                nonrefwords(newreflist, newcandlist, foundnum, ngramlength)
        # If no match is found, try to find matches for ngrams 1 smaller
        ngramlength -= 1
    # Return the new lists if all possible ngrams have been searched
    return newreflist, newcandlist


def getrefdict(
    newreflist: List[str],
    newcandlist: List[str],
    tripletyperef: AttrType,
    tripletypecand: AttrType,
    baseidx: int,
) -> Tuple[Literal["y", "n"], List[dict], List[dict], List[str]]:
    """
    :param newreflist: a list of token, where each token can be either
        a specific token or of the form "FOUNDREF-X-Y" where X is the
        index of the ngram where the token was matched and Y the
        corresponding match in NEWCANDLIST.
    :param newcandlist: a list of token.  Same as above except
        "FOUNDCAND" replaces "FOUNDREF"

    >>> getrefdict(['FOUNDREF-1-0', 'FOUNDREF-1-1', 'FOUNDREF-2-2', 'test'], ['FOUNDCAND-1-0', 'FOUNDCAND-1-1', 'FOUNDCAND-2-2'], "SUB", "OBJ", 0)
    ('y', [{'label': 'SUB', 'start': 0, 'end': 3}], [{'label': 'OBJ', 'start': 0, 'end': 1}, {'label': 'OBJ', 'start': 2, 'end': 2}], ['FOUNDREF-1-0', 'FOUNDREF-1-1', 'FOUNDREF-2-2', 'test'])

    >>> getrefdict(['FOUNDREF-1-0', 'FOUNDREF-1-1', 'FOUNDREF-1-2', 'test'], ['FOUNDCAND-1-0', 'FOUNDCAND-1-1', 'FOUNDCAND-1-2'], "SUB", "OBJ", 0)
    ('y', [{'label': 'SUB', 'start': 0, 'end': 3}], [{'label': 'OBJ', 'start': 0, 'end': 2}], ['FOUNDREF-1-0', 'FOUNDREF-1-1', 'FOUNDREF-1-2', 'test'])

    >>> getrefdict(["test"], ["quetzalcoatl", "quetzalcoatl"], "SUB", "OBJ", 0)
    ('n', [{'label': 'SUB', 'start': 0, 'end': 1}], [{'label': 'OBJ', 'start': 2, 'end': 3}], ['test', 'goddammit', 'quetzalcoatl', 'quetzalcoatl'])

    :return: a quadruple:
        - whether or not there was at least a partial match between ref and cand tokens
        -
        -
        -
    """
    try:
        # If some match is found with the reference
        firstfoundidx = newcandlist.index(
            [i for i in newcandlist if re.findall(r"^FOUNDCAND", i)][0]
        )
        candidatefound = "y"
    except IndexError:
        candidatefound = "n"

    if candidatefound == "y":
        unlinkedlist = []
        beforelist = []
        afterlist = []

        # If the first found candidate match is also the first word in the reference
        if newcandlist[firstfoundidx].endswith("-0"):
            # Flag that some words can appear before the first match, and they are linked with the first candidate match
            beforelinked = "y"
            firstcand = re.search(
                r"^(FOUNDCAND-\d+)-", newcandlist[firstfoundidx]
            ).group(1)
        else:
            beforelinked = "n"

        lastfoundidx = None
        afterlinked = None
        # If there's more words after the last reference, link those to the last reference as well
        # If the last reference word is linked, but the last candidate word is not, one criterion of linking the last words is met
        if (newreflist[-1].startswith("FOUNDREF")) and (
            not newcandlist[-1].startswith("FOUNDCAND")
        ):
            # If the last linked reference word is the last linked candidate word, the other criterion is also met.
            lastfound = [i for i in newcandlist if re.findall(r"^FOUNDCAND", i)][-1]
            candversion = newreflist[-1].replace("FOUNDREF", "FOUNDCAND")
            if lastfound == candversion:
                lastfoundidx = newcandlist.index(
                    [i for i in newcandlist if re.findall(r"^FOUNDCAND", i)][-1]
                )
                afterlinked = "y"
                lastcand = re.search(r"^(FOUNDCAND-\d+)-", lastfound).group(1)

        # Ensure that all the not-found blocks are separated by giving them different unlinknumbers
        unlinknumber = 1
        for idx, can in enumerate(newcandlist):
            if not can.startswith("FOUNDCAND"):
                if (idx < firstfoundidx) and (beforelinked == "y"):
                    newcandlist[idx] = firstcand + "-LINKED"
                    beforelist.append(firstcand + "-LINKED")
                elif (
                    (lastfoundidx != None)
                    and (afterlinked != None)
                    and (idx > lastfoundidx)
                    and (afterlinked == "y")
                ):
                    newcandlist[idx] = lastcand + "-LINKED"
                    afterlist.append(lastcand + "-LINKED")
                else:
                    unlinkedlist.append("NOTFOUND-" + str(unlinknumber))
            else:
                unlinknumber += 1

        totallist = beforelist + newreflist + afterlist + unlinkedlist

        refstart = len(beforelist)
        refend = (len(beforelist) + len(newreflist)) - 1

        refdictlist = [
            {
                "label": tripletyperef,
                "start": baseidx + refstart,
                "end": baseidx + refend,
            }
        ]

        totallist2 = [x.replace("FOUNDREF", "FOUNDCAND") for x in totallist]

        canddictlist = []
        currentcandidate = ""
        beginidx = ""
        endidx = ""
        collecting = "n"
        for idx, candidate in enumerate(totallist2):
            if (candidate.startswith("FOUNDCAND")) or (
                candidate.startswith("NOTFOUND")
            ):
                collecting = "y"
                curcan = re.search(r"^((.*?)-\d+)", candidate).group(1)
                if curcan != currentcandidate:
                    if currentcandidate != "":
                        endidx = idx - 1
                        canddictlist.append(
                            {
                                "label": tripletypecand,
                                "start": baseidx + beginidx,
                                "end": baseidx + endidx,
                            }
                        )
                    currentcandidate = curcan
                    beginidx = idx

                if idx == len(totallist2) - 1:
                    endidx = idx
                    canddictlist.append(
                        {
                            "label": tripletypecand,
                            "start": baseidx + beginidx,
                            "end": baseidx + endidx,
                        }
                    )
            else:
                if collecting == "y":
                    endidx = idx - 1
                    canddictlist.append(
                        {
                            "label": tripletypecand,
                            "start": baseidx + beginidx,
                            "end": baseidx + endidx,
                        }
                    )

    else:
        if len(newreflist) == 0:
            refdictlist = []
            canddictlist = [
                {
                    "label": tripletypecand,
                    "start": baseidx,
                    "end": baseidx + (len(newcandlist) - 1),
                }
            ]
            totallist = newcandlist
        elif len(newcandlist) == 0:
            canddictlist = []
            refdictlist = [
                {
                    "label": tripletyperef,
                    "start": baseidx,
                    "end": baseidx + (len(newreflist) - 1),
                }
            ]
            totallist = refdictlist
        else:
            totallist = newreflist + newcandlist
            refdictlist = [
                {
                    "label": tripletyperef,
                    "start": baseidx,
                    "end": baseidx + (len(newreflist) - 1),
                }
            ]
            canddictlist = [
                {
                    "label": tripletypecand,
                    "start": baseidx + len(newreflist),
                    "end": baseidx + (len(totallist) - 1),
                }
            ]

    return candidatefound, refdictlist, canddictlist, totallist


def parse_triple(triple: str) -> list[str]:
    split = triple.split(" | ")
    if len(split) < 1:
        return ["", "", ""]
    return split


def cleanup_tokens(tokens: list[str]) -> list[str]:
    return [
        x.lower()
        for x in tokens
        if re.search(r"^[" + re.escape(string.punctuation) + r"]+$", x) == None
    ]


class NERSpan(TypedDict):
    label: AttrType
    start: int
    end: int


@dataclass
class NERSpansMatch:
    found: yn
    ref_dicts: list[NERSpan]  # in practice, the length is always 1?
    cand_dicts: list[NERSpan]  # same


def _swapped_ner_spans(
    ref: str, cand: str, attr_type1: AttrType, attr_type2: AttrType, offset: int
) -> NERSpansMatch:
    reflist = cleanup_tokens(nltk.word_tokenize(ref))
    candlist = cleanup_tokens(nltk.word_tokenize(cand))

    reflist, candlist = nonrefwords(reflist, candlist, 1, len(candlist))
    candfound, refdicts, canddicts, _ = getrefdict(
        reflist, candlist, attr_type1, attr_type2, offset
    )

    return NERSpansMatch(candfound, refdicts, canddicts)


def evaluaterefcand(reference: str, candidate: str) -> tuple[dict, dict]:
    ref = parse_triple(reference)
    cand = parse_triple(candidate)
    assert len(ref) == len(cand)
    if len(ref) == 3:
        attr_types = ["SUB", "PRED", "OBJ"]
    elif len(ref) == 4:
        attr_types = ["SUB", "PRED", "OBJ", "TS"]
    else:
        raise ValueError(f"invalid n-tuple length: {len(ref)}")

    best_scores = (
        {"exact": {"f1": 0.0}, "partial": {"f1": 0.0}},
        {attr_type: {} for attr_type in attr_types},
    )

    for cand_permut, cand_attr_types_permut in zip(
        it.permutations(cand), it.permutations(attr_types)
    ):
        offset = 0
        ref_dict = {}
        cand_dict = {}
        for ref_attr, cand_attr, ref_attr_type, cand_attr_type in zip(
            ref, cand_permut, attr_types, cand_attr_types_permut
        ):
            match_ = _swapped_ner_spans(
                ref_attr, cand_attr, ref_attr_type, cand_attr_type, offset
            )
            ref_dict[ref_attr_type] = match_.ref_dicts
            cand_dict[cand_attr_type] = match_.cand_dicts
            offset = (
                max(
                    max(d["end"] for d in match_.ref_dicts),
                    max(d["end"] for d in match_.cand_dicts),
                )
                + 1
            )

        ref_list = list(ft.reduce(add, [ref_dict[attr] for attr in attr_types]))
        cand_list = list(ft.reduce(add, [cand_dict[attr] for attr in attr_types]))
        scores = Evaluator([ref_list], [cand_list], tags=attr_types).evaluate()

        # This is the default alignment: we use it to obtain "strict"
        # and "ent_type" scores since these depends on the candidate
        # alignment.
        if cand_attr_types_permut == tuple(attr_types):
            best_scores[0]["strict"] = scores[0]["strict"]
            best_scores[0]["ent_type"] = scores[0]["ent_type"]
            for attr_type in attr_types:
                best_scores[1][attr_type]["strict"] = scores[1][attr_type]["strict"]
                best_scores[1][attr_type]["ent_type"] = scores[1][attr_type]["ent_type"]

        # For "exact" and "partial" scores, we are allowed to search
        # for the best alignment. If this alignment is the best so
        # far, we update these scores. We prioritize "exact" F1 and
        # break ties with "partial" F1.
        if scores[0]["exact"]["f1"] > best_scores[0]["exact"]["f1"] or (
            scores[0]["exact"]["f1"] == best_scores[0]["exact"]["f1"]
            and scores[0]["partial"]["f1"] > best_scores[0]["partial"]["f1"]
        ):
            best_scores[0]["exact"] = scores[0]["exact"]
            best_scores[0]["partial"] = scores[0]["partial"]
            for attr_type in attr_types:
                best_scores[1][attr_type]["exact"] = scores[1][attr_type]["exact"]
                best_scores[1][attr_type]["partial"] = scores[1][attr_type]["partial"]

    return best_scores


def calculateAllScores(newreflist: list[list[str]], newcandlist: list[list[str]]):
    totalsemevallist = []
    totalsemevallistpertag = []

    for idx, candidate in enumerate(newcandlist):
        if len(newcandlist[idx]) != len(newreflist[idx]):
            differencebetween = abs(len(newcandlist[idx]) - len(newreflist[idx]))
            differencelist = [""] * differencebetween
            if len(newcandlist[idx]) < len(newreflist[idx]):
                newcandlist[idx] = newcandlist[idx] + differencelist
            else:
                newreflist[idx] = newreflist[idx] + differencelist
            assert len(newreflist[idx]) == len(newcandlist[idx])

    for idx, candidate in enumerate(newcandlist):
        candidatesemeval = []
        candidatesemevalpertag = []
        for triple in candidate:
            triplesemeval: List[dict] = []
            triplesemevalpertag = []
            for reference in newreflist[idx]:
                results, results_per_tag = evaluaterefcand(reference, triple)
                triplesemeval.append(results)
                triplesemevalpertag.append(results_per_tag)

            candidatesemeval.append(triplesemeval)
            candidatesemevalpertag.append(triplesemevalpertag)

        totalsemevallist.append(candidatesemeval)
        totalsemevallistpertag.append(candidatesemevalpertag)

    return totalsemevallist, totalsemevallistpertag


def calculateSystemScore(
    totalsemevallist, totalsemevallistpertag, newreflist, newcandlist
):
    selectedsemevallist = []
    selectedsemevallistpertag = []
    selectedalignment = []
    selectedscores = []
    alldicts = []

    # Get all the permutations of the number of scores given per candidate, so if there's 4 candidates, but 3 references, this part ensures that one of
    # The four will not be scored
    for idx, candidate in enumerate(newcandlist):
        if len(newcandlist[idx]) > len(newreflist[idx]):
            # Get all permutations
            choosecands = list(
                itertools.permutations(
                    [x[0] for x in enumerate(totalsemevallist[idx])],
                    len(totalsemevallist[idx][0]),
                )
            )
            # The permutations in different orders are not necessary: we only need one order without the number of candidates we're looking at
            choosecands = set(
                [tuple(sorted(i)) for i in choosecands]
            )  # Sort inner list and then use set
            choosecands = list(map(list, choosecands))  # Converting back to list
        else:
            # Otherwise, we're just going to score all candidates
            choosecands = [list(range(len(newcandlist[idx])))]

        # Get all permutations in which the scores can be combined
        if len(newcandlist[idx]) > len(newreflist[idx]):
            choosescore = list(
                itertools.permutations(
                    [x[0] for x in enumerate(totalsemevallist[idx][0])],
                    len(newreflist[idx]),
                )
            )
            choosescore = [list(x) for x in choosescore]
        else:
            choosescore = list(
                itertools.permutations(
                    [x[0] for x in enumerate(totalsemevallist[idx][0])],
                    len(newcandlist[idx]),
                )
            )
            choosescore = [list(x) for x in choosescore]

        # Get all possible combinations between the candidates and the scores
        combilist = list(itertools.product(choosecands, choosescore))

        totaldict = {"totalscore": 0}

        for combination in combilist:
            combiscore = 0
            # Take the combination between the candidate and the score
            zipcombi = list(zip(combination[0], combination[1]))
            collectedsemeval = []
            collectedsemevalpertag = []

            for zc_idx, zc in enumerate(zipcombi):
                collectedscores = totalsemevallist[idx][zc[0]][zc[1]]
                f1score = statistics.mean(
                    [
                        collectedscores["ent_type"]["f1"],
                        collectedscores["partial"]["f1"],
                        collectedscores["strict"]["f1"],
                        collectedscores["exact"]["f1"],
                    ]
                )
                combiscore += f1score

                collectedsemeval.append(collectedscores)

                assert (
                    combination[0][zc_idx] == zc[0] and combination[1][zc_idx] == zc[1]
                )

                collectedsemevalpertag.append(totalsemevallistpertag[idx][zc[0]][zc[1]])

            # If the combination is the highest score thus far, or the first score, make it the totaldict
            if (combiscore > totaldict["totalscore"]) or (len(totaldict) == 1):
                totaldict = {
                    "totalscore": combiscore,
                    "combination": combination,
                    "semevallist": collectedsemeval,
                    "semevalpertaglist": collectedsemevalpertag,
                }

        selectedsemevallist = selectedsemevallist + totaldict["semevallist"]
        selectedsemevallistpertag = (
            selectedsemevallistpertag + totaldict["semevalpertaglist"]
        )
        selectedalignment.append(totaldict["combination"])
        selectedscores.append(totaldict["totalscore"] / len(candidate))

    print("-----------------------------------------------------------------")
    print("Total scores")
    print("-----------------------------------------------------------------")
    print("Ent_type")
    enttypecorrect = sum([x["ent_type"]["correct"] for x in selectedsemevallist])
    enttypeincorrect = sum([x["ent_type"]["incorrect"] for x in selectedsemevallist])
    enttypepartial = sum([x["ent_type"]["partial"] for x in selectedsemevallist])
    enttypemissed = sum([x["ent_type"]["missed"] for x in selectedsemevallist])
    enttypespurious = sum([x["ent_type"]["spurious"] for x in selectedsemevallist])
    enttypepossible = sum([x["ent_type"]["possible"] for x in selectedsemevallist])
    enttypeactual = sum([x["ent_type"]["actual"] for x in selectedsemevallist])
    enttypeprecision = statistics.mean(
        [x["ent_type"]["precision"] for x in selectedsemevallist]
    )
    enttyperecall = statistics.mean(
        [x["ent_type"]["recall"] for x in selectedsemevallist]
    )
    enttypef1 = statistics.mean([x["ent_type"]["f1"] for x in selectedsemevallist])
    print(
        "Correct: "
        + str(enttypecorrect)
        + " Incorrect: "
        + str(enttypeincorrect)
        + " Partial: "
        + str(enttypepartial)
        + " Missed: "
        + str(enttypemissed)
        + "\nSpurious: "
        + str(enttypespurious)
        + " Possible: "
        + str(enttypepossible)
        + " Actual: "
        + str(enttypeactual)
        + "\nPrecision: "
        + str(enttypeprecision)
        + " Recall: "
        + str(enttyperecall)
        + "\nF1: "
        + str(enttypef1)
    )
    print("-----------------------------------------------------------------")
    print("Partial")
    partialcorrect = sum([x["partial"]["correct"] for x in selectedsemevallist])
    partialincorrect = sum([x["partial"]["incorrect"] for x in selectedsemevallist])
    partialpartial = sum([x["partial"]["partial"] for x in selectedsemevallist])
    partialmissed = sum([x["partial"]["missed"] for x in selectedsemevallist])
    partialspurious = sum([x["partial"]["spurious"] for x in selectedsemevallist])
    partialpossible = sum([x["partial"]["possible"] for x in selectedsemevallist])
    partialactual = sum([x["partial"]["actual"] for x in selectedsemevallist])
    partialprecision = statistics.mean(
        [x["partial"]["precision"] for x in selectedsemevallist]
    )
    partialrecall = statistics.mean(
        [x["partial"]["recall"] for x in selectedsemevallist]
    )
    partialf1 = statistics.mean([x["partial"]["f1"] for x in selectedsemevallist])
    print(
        "Correct: "
        + str(partialcorrect)
        + " Incorrect: "
        + str(partialincorrect)
        + " Partial: "
        + str(partialpartial)
        + " Missed: "
        + str(partialmissed)
        + "\nSpurious: "
        + str(partialspurious)
        + " Possible: "
        + str(partialpossible)
        + " Actual: "
        + str(partialactual)
        + "\nPrecision: "
        + str(partialprecision)
        + " Recall: "
        + str(partialrecall)
        + "\nF1: "
        + str(partialf1)
    )
    print("-----------------------------------------------------------------")
    print("Strict")
    strictcorrect = sum([x["strict"]["correct"] for x in selectedsemevallist])
    strictincorrect = sum([x["strict"]["incorrect"] for x in selectedsemevallist])
    strictpartial = sum([x["strict"]["partial"] for x in selectedsemevallist])
    strictmissed = sum([x["strict"]["missed"] for x in selectedsemevallist])
    strictspurious = sum([x["strict"]["spurious"] for x in selectedsemevallist])
    strictpossible = sum([x["strict"]["possible"] for x in selectedsemevallist])
    strictactual = sum([x["strict"]["actual"] for x in selectedsemevallist])
    strictprecision = statistics.mean(
        [x["strict"]["precision"] for x in selectedsemevallist]
    )
    strictrecall = statistics.mean([x["strict"]["recall"] for x in selectedsemevallist])
    strictf1 = statistics.mean([x["strict"]["f1"] for x in selectedsemevallist])
    print(
        "Correct: "
        + str(strictcorrect)
        + " Incorrect: "
        + str(strictincorrect)
        + " Partial: "
        + str(strictpartial)
        + " Missed: "
        + str(strictmissed)
        + "\nSpurious: "
        + str(strictspurious)
        + " Possible: "
        + str(strictpossible)
        + " Actual: "
        + str(strictactual)
        + "\nPrecision: "
        + str(strictprecision)
        + " Recall: "
        + str(strictrecall)
        + "\nF1: "
        + str(strictf1)
    )
    print("-----------------------------------------------------------------")
    print("Exact")
    exactcorrect = sum([x["exact"]["correct"] for x in selectedsemevallist])
    exactincorrect = sum([x["exact"]["incorrect"] for x in selectedsemevallist])
    exactpartial = sum([x["exact"]["partial"] for x in selectedsemevallist])
    exactmissed = sum([x["exact"]["missed"] for x in selectedsemevallist])
    exactspurious = sum([x["exact"]["spurious"] for x in selectedsemevallist])
    exactpossible = sum([x["exact"]["possible"] for x in selectedsemevallist])
    exactactual = sum([x["exact"]["actual"] for x in selectedsemevallist])
    exactprecision = statistics.mean(
        [x["exact"]["precision"] for x in selectedsemevallist]
    )
    exactrecall = statistics.mean([x["exact"]["recall"] for x in selectedsemevallist])
    exactf1 = statistics.mean([x["exact"]["f1"] for x in selectedsemevallist])
    print(
        "Correct: "
        + str(exactcorrect)
        + " Incorrect: "
        + str(exactincorrect)
        + " Partial: "
        + str(exactpartial)
        + " Missed: "
        + str(exactmissed)
        + "\nSpurious: "
        + str(exactspurious)
        + " Possible: "
        + str(exactpossible)
        + " Actual: "
        + str(exactactual)
        + "\nPrecision: "
        + str(exactprecision)
        + " Recall: "
        + str(exactrecall)
        + "\nF1: "
        + str(exactf1)
    )
    print("-----------------------------------------------------------------")
    print("Scores per tag")
    print("-----------------------------------------------------------------")
    print("Subjects")
    print("-----------------------------------------------------------------")
    print("Ent_type")
    subenttypecorrect = sum(
        [x["SUB"]["ent_type"]["correct"] for x in selectedsemevallistpertag]
    )
    subenttypeincorrect = sum(
        [x["SUB"]["ent_type"]["incorrect"] for x in selectedsemevallistpertag]
    )
    subenttypepartial = sum(
        [x["SUB"]["ent_type"]["partial"] for x in selectedsemevallistpertag]
    )
    subenttypemissed = sum(
        [x["SUB"]["ent_type"]["missed"] for x in selectedsemevallistpertag]
    )
    subenttypespurious = sum(
        [x["SUB"]["ent_type"]["spurious"] for x in selectedsemevallistpertag]
    )
    subenttypepossible = sum(
        [x["SUB"]["ent_type"]["possible"] for x in selectedsemevallistpertag]
    )
    subenttypeactual = sum(
        [x["SUB"]["ent_type"]["actual"] for x in selectedsemevallistpertag]
    )
    subenttypeprecision = statistics.mean(
        [x["SUB"]["ent_type"]["precision"] for x in selectedsemevallistpertag]
    )
    subenttyperecall = statistics.mean(
        [x["SUB"]["ent_type"]["recall"] for x in selectedsemevallistpertag]
    )
    subenttypef1 = statistics.mean(
        [x["SUB"]["ent_type"]["f1"] for x in selectedsemevallistpertag]
    )
    print(
        "Correct: "
        + str(subenttypecorrect)
        + " Incorrect: "
        + str(subenttypeincorrect)
        + " Partial: "
        + str(subenttypepartial)
        + " Missed: "
        + str(subenttypemissed)
        + "\nSpurious: "
        + str(subenttypespurious)
        + " Possible: "
        + str(subenttypepossible)
        + " Actual: "
        + str(subenttypeactual)
        + "\nPrecision: "
        + str(subenttypeprecision)
        + " Recall: "
        + str(subenttyperecall)
        + "\nF1: "
        + str(subenttypef1)
    )
    print("-----------------------------------------------------------------")
    print("Partial")
    subpartialcorrect = sum(
        [x["SUB"]["partial"]["correct"] for x in selectedsemevallistpertag]
    )
    subpartialincorrect = sum(
        [x["SUB"]["partial"]["incorrect"] for x in selectedsemevallistpertag]
    )
    subpartialpartial = sum(
        [x["SUB"]["partial"]["partial"] for x in selectedsemevallistpertag]
    )
    subpartialmissed = sum(
        [x["SUB"]["partial"]["missed"] for x in selectedsemevallistpertag]
    )
    subpartialspurious = sum(
        [x["SUB"]["partial"]["spurious"] for x in selectedsemevallistpertag]
    )
    subpartialpossible = sum(
        [x["SUB"]["partial"]["possible"] for x in selectedsemevallistpertag]
    )
    subpartialactual = sum(
        [x["SUB"]["partial"]["actual"] for x in selectedsemevallistpertag]
    )
    subpartialprecision = statistics.mean(
        [x["SUB"]["partial"]["precision"] for x in selectedsemevallistpertag]
    )
    subpartialrecall = statistics.mean(
        [x["SUB"]["partial"]["recall"] for x in selectedsemevallistpertag]
    )
    subpartialf1 = statistics.mean(
        [x["SUB"]["partial"]["f1"] for x in selectedsemevallistpertag]
    )
    print(
        "Correct: "
        + str(subpartialcorrect)
        + " Incorrect: "
        + str(subpartialincorrect)
        + " Partial: "
        + str(subpartialpartial)
        + " Missed: "
        + str(subpartialmissed)
        + "\nSpurious: "
        + str(subpartialspurious)
        + " Possible: "
        + str(subpartialpossible)
        + " Actual: "
        + str(subpartialactual)
        + "\nPrecision: "
        + str(subpartialprecision)
        + " Recall: "
        + str(subpartialrecall)
        + "\nF1: "
        + str(subpartialf1)
    )
    print("-----------------------------------------------------------------")
    print("Strict")
    substrictcorrect = sum(
        [x["SUB"]["strict"]["correct"] for x in selectedsemevallistpertag]
    )
    substrictincorrect = sum(
        [x["SUB"]["strict"]["incorrect"] for x in selectedsemevallistpertag]
    )
    substrictpartial = sum(
        [x["SUB"]["strict"]["partial"] for x in selectedsemevallistpertag]
    )
    substrictmissed = sum(
        [x["SUB"]["strict"]["missed"] for x in selectedsemevallistpertag]
    )
    substrictspurious = sum(
        [x["SUB"]["strict"]["spurious"] for x in selectedsemevallistpertag]
    )
    substrictpossible = sum(
        [x["SUB"]["strict"]["possible"] for x in selectedsemevallistpertag]
    )
    substrictactual = sum(
        [x["SUB"]["strict"]["actual"] for x in selectedsemevallistpertag]
    )
    substrictprecision = statistics.mean(
        [x["SUB"]["strict"]["precision"] for x in selectedsemevallistpertag]
    )
    substrictrecall = statistics.mean(
        [x["SUB"]["strict"]["recall"] for x in selectedsemevallistpertag]
    )
    substrictf1 = statistics.mean(
        [x["SUB"]["strict"]["f1"] for x in selectedsemevallistpertag]
    )
    print(
        "Correct: "
        + str(substrictcorrect)
        + " Incorrect: "
        + str(substrictincorrect)
        + " Partial: "
        + str(substrictpartial)
        + " Missed: "
        + str(substrictmissed)
        + "\nSpurious: "
        + str(substrictspurious)
        + " Possible: "
        + str(substrictpossible)
        + " Actual: "
        + str(substrictactual)
        + "\nPrecision: "
        + str(substrictprecision)
        + " Recall: "
        + str(substrictrecall)
        + "\nF1: "
        + str(substrictf1)
    )
    print("-----------------------------------------------------------------")
    print("Exact")
    subexactcorrect = sum(
        [x["SUB"]["exact"]["correct"] for x in selectedsemevallistpertag]
    )
    subexactincorrect = sum(
        [x["SUB"]["exact"]["incorrect"] for x in selectedsemevallistpertag]
    )
    subexactpartial = sum(
        [x["SUB"]["exact"]["partial"] for x in selectedsemevallistpertag]
    )
    subexactmissed = sum(
        [x["SUB"]["exact"]["missed"] for x in selectedsemevallistpertag]
    )
    subexactspurious = sum(
        [x["SUB"]["exact"]["spurious"] for x in selectedsemevallistpertag]
    )
    subexactpossible = sum(
        [x["SUB"]["exact"]["possible"] for x in selectedsemevallistpertag]
    )
    subexactactual = sum(
        [x["SUB"]["exact"]["actual"] for x in selectedsemevallistpertag]
    )
    subexactprecision = statistics.mean(
        [x["SUB"]["exact"]["precision"] for x in selectedsemevallistpertag]
    )
    subexactrecall = statistics.mean(
        [x["SUB"]["exact"]["recall"] for x in selectedsemevallistpertag]
    )
    subexactf1 = statistics.mean(
        [x["SUB"]["exact"]["f1"] for x in selectedsemevallistpertag]
    )
    print(
        "Correct: "
        + str(subexactcorrect)
        + " Incorrect: "
        + str(subexactincorrect)
        + " Partial: "
        + str(subexactpartial)
        + " Missed: "
        + str(subexactmissed)
        + "\nSpurious: "
        + str(subexactspurious)
        + " Possible: "
        + str(subexactpossible)
        + " Actual: "
        + str(subexactactual)
        + "\nPrecision: "
        + str(subexactprecision)
        + " Recall: "
        + str(subexactrecall)
        + "\nF1: "
        + str(subexactf1)
    )
    print("-----------------------------------------------------------------")
    print("Predicates")
    print("-----------------------------------------------------------------")
    print("Ent_type")
    predenttypecorrect = sum(
        [x["PRED"]["ent_type"]["correct"] for x in selectedsemevallistpertag]
    )
    predenttypeincorrect = sum(
        [x["PRED"]["ent_type"]["incorrect"] for x in selectedsemevallistpertag]
    )
    predenttypepartial = sum(
        [x["PRED"]["ent_type"]["partial"] for x in selectedsemevallistpertag]
    )
    predenttypemissed = sum(
        [x["PRED"]["ent_type"]["missed"] for x in selectedsemevallistpertag]
    )
    predenttypespurious = sum(
        [x["PRED"]["ent_type"]["spurious"] for x in selectedsemevallistpertag]
    )
    predenttypepossible = sum(
        [x["PRED"]["ent_type"]["possible"] for x in selectedsemevallistpertag]
    )
    predenttypeactual = sum(
        [x["PRED"]["ent_type"]["actual"] for x in selectedsemevallistpertag]
    )
    predenttypeprecision = statistics.mean(
        [x["PRED"]["ent_type"]["precision"] for x in selectedsemevallistpertag]
    )
    predenttyperecall = statistics.mean(
        [x["PRED"]["ent_type"]["recall"] for x in selectedsemevallistpertag]
    )
    predenttypef1 = statistics.mean(
        [x["PRED"]["ent_type"]["f1"] for x in selectedsemevallistpertag]
    )
    print(
        "Correct: "
        + str(predenttypecorrect)
        + " Incorrect: "
        + str(predenttypeincorrect)
        + " Partial: "
        + str(predenttypepartial)
        + " Missed: "
        + str(predenttypemissed)
        + "\nSpurious: "
        + str(predenttypespurious)
        + " Possible: "
        + str(predenttypepossible)
        + " Actual: "
        + str(predenttypeactual)
        + "\nPrecision: "
        + str(predenttypeprecision)
        + " Recall: "
        + str(predenttyperecall)
        + "\nF1: "
        + str(predenttypef1)
    )
    print("-----------------------------------------------------------------")
    print("Partial")
    predpartialcorrect = sum(
        [x["PRED"]["partial"]["correct"] for x in selectedsemevallistpertag]
    )
    predpartialincorrect = sum(
        [x["PRED"]["partial"]["incorrect"] for x in selectedsemevallistpertag]
    )
    predpartialpartial = sum(
        [x["PRED"]["partial"]["partial"] for x in selectedsemevallistpertag]
    )
    predpartialmissed = sum(
        [x["PRED"]["partial"]["missed"] for x in selectedsemevallistpertag]
    )
    predpartialspurious = sum(
        [x["PRED"]["partial"]["spurious"] for x in selectedsemevallistpertag]
    )
    predpartialpossible = sum(
        [x["PRED"]["partial"]["possible"] for x in selectedsemevallistpertag]
    )
    predpartialactual = sum(
        [x["PRED"]["partial"]["actual"] for x in selectedsemevallistpertag]
    )
    predpartialprecision = statistics.mean(
        [x["PRED"]["partial"]["precision"] for x in selectedsemevallistpertag]
    )
    predpartialrecall = statistics.mean(
        [x["PRED"]["partial"]["recall"] for x in selectedsemevallistpertag]
    )
    predpartialf1 = statistics.mean(
        [x["PRED"]["partial"]["f1"] for x in selectedsemevallistpertag]
    )
    print(
        "Correct: "
        + str(predpartialcorrect)
        + " Incorrect: "
        + str(predpartialincorrect)
        + " Partial: "
        + str(predpartialpartial)
        + " Missed: "
        + str(predpartialmissed)
        + "\nSpurious: "
        + str(predpartialspurious)
        + " Possible: "
        + str(predpartialpossible)
        + " Actual: "
        + str(predpartialactual)
        + "\nPrecision: "
        + str(predpartialprecision)
        + " Recall: "
        + str(predpartialrecall)
        + "\nF1: "
        + str(predpartialf1)
    )
    print("-----------------------------------------------------------------")
    print("Strict")
    predstrictcorrect = sum(
        [x["PRED"]["strict"]["correct"] for x in selectedsemevallistpertag]
    )
    predstrictincorrect = sum(
        [x["PRED"]["strict"]["incorrect"] for x in selectedsemevallistpertag]
    )
    predstrictpartial = sum(
        [x["PRED"]["strict"]["partial"] for x in selectedsemevallistpertag]
    )
    predstrictmissed = sum(
        [x["PRED"]["strict"]["missed"] for x in selectedsemevallistpertag]
    )
    predstrictspurious = sum(
        [x["PRED"]["strict"]["spurious"] for x in selectedsemevallistpertag]
    )
    predstrictpossible = sum(
        [x["PRED"]["strict"]["possible"] for x in selectedsemevallistpertag]
    )
    predstrictactual = sum(
        [x["PRED"]["strict"]["actual"] for x in selectedsemevallistpertag]
    )
    predstrictprecision = statistics.mean(
        [x["PRED"]["strict"]["precision"] for x in selectedsemevallistpertag]
    )
    predstrictrecall = statistics.mean(
        [x["PRED"]["strict"]["recall"] for x in selectedsemevallistpertag]
    )
    predstrictf1 = statistics.mean(
        [x["PRED"]["strict"]["f1"] for x in selectedsemevallistpertag]
    )
    print(
        "Correct: "
        + str(predstrictcorrect)
        + " Incorrect: "
        + str(predstrictincorrect)
        + " Partial: "
        + str(predstrictpartial)
        + " Missed: "
        + str(predstrictmissed)
        + "\nSpurious: "
        + str(predstrictspurious)
        + " Possible: "
        + str(predstrictpossible)
        + " Actual: "
        + str(predstrictactual)
        + "\nPrecision: "
        + str(predstrictprecision)
        + " Recall: "
        + str(predstrictrecall)
        + "\nF1: "
        + str(predstrictf1)
    )
    print("-----------------------------------------------------------------")
    print("Exact")
    predexactcorrect = sum(
        [x["PRED"]["exact"]["correct"] for x in selectedsemevallistpertag]
    )
    predexactincorrect = sum(
        [x["PRED"]["exact"]["incorrect"] for x in selectedsemevallistpertag]
    )
    predexactpartial = sum(
        [x["PRED"]["exact"]["partial"] for x in selectedsemevallistpertag]
    )
    predexactmissed = sum(
        [x["PRED"]["exact"]["missed"] for x in selectedsemevallistpertag]
    )
    predexactspurious = sum(
        [x["PRED"]["exact"]["spurious"] for x in selectedsemevallistpertag]
    )
    predexactpossible = sum(
        [x["PRED"]["exact"]["possible"] for x in selectedsemevallistpertag]
    )
    predexactactual = sum(
        [x["PRED"]["exact"]["actual"] for x in selectedsemevallistpertag]
    )
    predexactprecision = statistics.mean(
        [x["PRED"]["exact"]["precision"] for x in selectedsemevallistpertag]
    )
    predexactrecall = statistics.mean(
        [x["PRED"]["exact"]["recall"] for x in selectedsemevallistpertag]
    )
    predexactf1 = statistics.mean(
        [x["PRED"]["exact"]["f1"] for x in selectedsemevallistpertag]
    )
    print(
        "Correct: "
        + str(predexactcorrect)
        + " Incorrect: "
        + str(predexactincorrect)
        + " Partial: "
        + str(predexactpartial)
        + " Missed: "
        + str(predexactmissed)
        + "\nSpurious: "
        + str(predexactspurious)
        + " Possible: "
        + str(predexactpossible)
        + " Actual: "
        + str(predexactactual)
        + "\nPrecision: "
        + str(predexactprecision)
        + " Recall: "
        + str(predexactrecall)
        + "\nF1: "
        + str(predexactf1)
    )
    print("-----------------------------------------------------------------")
    print("Objects")
    print("-----------------------------------------------------------------")
    print("Ent_type")
    objenttypecorrect = sum(
        [x["OBJ"]["ent_type"]["correct"] for x in selectedsemevallistpertag]
    )
    objenttypeincorrect = sum(
        [x["OBJ"]["ent_type"]["incorrect"] for x in selectedsemevallistpertag]
    )
    objenttypepartial = sum(
        [x["OBJ"]["ent_type"]["partial"] for x in selectedsemevallistpertag]
    )
    objenttypemissed = sum(
        [x["OBJ"]["ent_type"]["missed"] for x in selectedsemevallistpertag]
    )
    objenttypespurious = sum(
        [x["OBJ"]["ent_type"]["spurious"] for x in selectedsemevallistpertag]
    )
    objenttypepossible = sum(
        [x["OBJ"]["ent_type"]["possible"] for x in selectedsemevallistpertag]
    )
    objenttypeactual = sum(
        [x["OBJ"]["ent_type"]["actual"] for x in selectedsemevallistpertag]
    )
    objenttypeprecision = statistics.mean(
        [x["OBJ"]["ent_type"]["precision"] for x in selectedsemevallistpertag]
    )
    objenttyperecall = statistics.mean(
        [x["OBJ"]["ent_type"]["recall"] for x in selectedsemevallistpertag]
    )
    objenttypef1 = statistics.mean(
        [x["OBJ"]["ent_type"]["f1"] for x in selectedsemevallistpertag]
    )
    print(
        "Correct: "
        + str(objenttypecorrect)
        + " Incorrect: "
        + str(objenttypeincorrect)
        + " Partial: "
        + str(objenttypepartial)
        + " Missed: "
        + str(objenttypemissed)
        + "\nSpurious: "
        + str(objenttypespurious)
        + " Possible: "
        + str(objenttypepossible)
        + " Actual: "
        + str(objenttypeactual)
        + "\nPrecision: "
        + str(objenttypeprecision)
        + " Recall: "
        + str(objenttyperecall)
        + "\nF1: "
        + str(objenttypef1)
    )
    print("-----------------------------------------------------------------")
    print("Partial")
    objpartialcorrect = sum(
        [x["OBJ"]["partial"]["correct"] for x in selectedsemevallistpertag]
    )
    objpartialincorrect = sum(
        [x["OBJ"]["partial"]["incorrect"] for x in selectedsemevallistpertag]
    )
    objpartialpartial = sum(
        [x["OBJ"]["partial"]["partial"] for x in selectedsemevallistpertag]
    )
    objpartialmissed = sum(
        [x["OBJ"]["partial"]["missed"] for x in selectedsemevallistpertag]
    )
    objpartialspurious = sum(
        [x["OBJ"]["partial"]["spurious"] for x in selectedsemevallistpertag]
    )
    objpartialpossible = sum(
        [x["OBJ"]["partial"]["possible"] for x in selectedsemevallistpertag]
    )
    objpartialactual = sum(
        [x["OBJ"]["partial"]["actual"] for x in selectedsemevallistpertag]
    )
    objpartialprecision = statistics.mean(
        [x["OBJ"]["partial"]["precision"] for x in selectedsemevallistpertag]
    )
    objpartialrecall = statistics.mean(
        [x["OBJ"]["partial"]["recall"] for x in selectedsemevallistpertag]
    )
    objpartialf1 = statistics.mean(
        [x["OBJ"]["partial"]["f1"] for x in selectedsemevallistpertag]
    )
    print(
        "Correct: "
        + str(objpartialcorrect)
        + " Incorrect: "
        + str(objpartialincorrect)
        + " Partial: "
        + str(objpartialpartial)
        + " Missed: "
        + str(objpartialmissed)
        + "\nSpurious: "
        + str(objpartialspurious)
        + " Possible: "
        + str(objpartialpossible)
        + " Actual: "
        + str(objpartialactual)
        + "\nPrecision: "
        + str(objpartialprecision)
        + " Recall: "
        + str(objpartialrecall)
        + "\nF1: "
        + str(objpartialf1)
    )
    print("-----------------------------------------------------------------")
    print("Strict")
    objstrictcorrect = sum(
        [x["OBJ"]["strict"]["correct"] for x in selectedsemevallistpertag]
    )
    objstrictincorrect = sum(
        [x["OBJ"]["strict"]["incorrect"] for x in selectedsemevallistpertag]
    )
    objstrictpartial = sum(
        [x["OBJ"]["strict"]["partial"] for x in selectedsemevallistpertag]
    )
    objstrictmissed = sum(
        [x["OBJ"]["strict"]["missed"] for x in selectedsemevallistpertag]
    )
    objstrictspurious = sum(
        [x["OBJ"]["strict"]["spurious"] for x in selectedsemevallistpertag]
    )
    objstrictpossible = sum(
        [x["OBJ"]["strict"]["possible"] for x in selectedsemevallistpertag]
    )
    objstrictactual = sum(
        [x["OBJ"]["strict"]["actual"] for x in selectedsemevallistpertag]
    )
    objstrictprecision = statistics.mean(
        [x["OBJ"]["strict"]["precision"] for x in selectedsemevallistpertag]
    )
    objstrictrecall = statistics.mean(
        [x["OBJ"]["strict"]["recall"] for x in selectedsemevallistpertag]
    )
    objstrictf1 = statistics.mean(
        [x["OBJ"]["strict"]["f1"] for x in selectedsemevallistpertag]
    )
    print(
        "Correct: "
        + str(objstrictcorrect)
        + " Incorrect: "
        + str(objstrictincorrect)
        + " Partial: "
        + str(objstrictpartial)
        + " Missed: "
        + str(objstrictmissed)
        + "\nSpurious: "
        + str(objstrictspurious)
        + " Possible: "
        + str(objstrictpossible)
        + " Actual: "
        + str(objstrictactual)
        + "\nPrecision: "
        + str(objstrictprecision)
        + " Recall: "
        + str(objstrictrecall)
        + "\nF1: "
        + str(objstrictf1)
    )
    print("-----------------------------------------------------------------")
    print("Exact")
    objexactcorrect = sum(
        [x["OBJ"]["exact"]["correct"] for x in selectedsemevallistpertag]
    )
    objexactincorrect = sum(
        [x["OBJ"]["exact"]["incorrect"] for x in selectedsemevallistpertag]
    )
    objexactpartial = sum(
        [x["OBJ"]["exact"]["partial"] for x in selectedsemevallistpertag]
    )
    objexactmissed = sum(
        [x["OBJ"]["exact"]["missed"] for x in selectedsemevallistpertag]
    )
    objexactspurious = sum(
        [x["OBJ"]["exact"]["spurious"] for x in selectedsemevallistpertag]
    )
    objexactpossible = sum(
        [x["OBJ"]["exact"]["possible"] for x in selectedsemevallistpertag]
    )
    objexactactual = sum(
        [x["OBJ"]["exact"]["actual"] for x in selectedsemevallistpertag]
    )
    objexactprecision = statistics.mean(
        [x["OBJ"]["exact"]["precision"] for x in selectedsemevallistpertag]
    )
    objexactrecall = statistics.mean(
        [x["OBJ"]["exact"]["recall"] for x in selectedsemevallistpertag]
    )
    objexactf1 = statistics.mean(
        [x["OBJ"]["exact"]["f1"] for x in selectedsemevallistpertag]
    )
    print(
        "Correct: "
        + str(objexactcorrect)
        + " Incorrect: "
        + str(objexactincorrect)
        + " Partial: "
        + str(objexactpartial)
        + " Missed: "
        + str(objexactmissed)
        + "\nSpurious: "
        + str(objexactspurious)
        + " Possible: "
        + str(objexactpossible)
        + " Actual: "
        + str(objexactactual)
        + "\nPrecision: "
        + str(objexactprecision)
        + " Recall: "
        + str(objexactrecall)
        + "\nF1: "
        + str(objexactf1)
    )
    print("-----------------------------------------------------------------")
    return (
        selectedsemevallist,
        selectedsemevallistpertag,
        selectedalignment,
        selectedscores,
    )


def calculateExactTripleScore(reflist: List[List[str]], candlist: List[List[str]]):
    newreflist = [[string.lower() for string in sublist] for sublist in reflist]
    newcandlist = [[string.lower() for string in sublist] for sublist in candlist]
    # First get all the classes by combining the triples in the candidatelist and referencelist
    allclasses = newcandlist + newreflist
    allclasses = [item for items in allclasses for item in items]
    allclasses = list(set(allclasses))

    lb = preprocessing.MultiLabelBinarizer(classes=allclasses)
    mcbin = lb.fit_transform(newcandlist)
    mrbin = lb.fit_transform(newreflist)

    precision = precision_score(mrbin, mcbin, average="macro")
    recall = recall_score(mrbin, mcbin, average="macro")
    f1 = f1_score(mrbin, mcbin, average="macro")

    print("Full triple scores")
    print("-----------------------------------------------------------------")
    print(
        "Precision: " + str(precision) + " Recall: " + str(recall) + "\nF1: " + str(f1)
    )


def main(reffile, candfile):
    reflist, newreflist = getRefs(reffile)
    candlist, newcandlist = getCands(candfile)
    totalsemevallist, totalsemevallistpertag = calculateAllScores(
        newreflist, newcandlist
    )
    calculateSystemScore(
        totalsemevallist, totalsemevallistpertag, newreflist, newcandlist
    )
    calculateExactTripleScore(reflist, candlist)


# main(currentpath + '/Refs.xml', currentpath + '/Cands2.xml')
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--edc_output")
    parser.add_argument("--reference")
    parser.add_argument("--max_length_diff", default=None)
    args = parser.parse_args()
    result_xml_path, gold_xml_path = convert_to_xml(
        args.edc_output, args.reference, args.max_length_diff
    )
    main(gold_xml_path, result_xml_path)
