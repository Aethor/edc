from typing import List, Literal
import re, string
import nltk
from evaluate.evaluation_script import getrefdict, nonrefwords

AttrType = Literal["SUB", "PRED", "OBJ"]
yn = Literal["y", "n"]


def evaluaterefcand_2(reference: str, candidate: str) -> tuple[dict, dict]:
    newreference = reference.split(" | ")
    newcandidate = candidate.split(" | ")

    # Make sure that reference or candidate aren't '' values originally.
    if (len(newreference) > 1) and (len(newcandidate) > 1):
        indextriple = newreference
    elif len(newreference) == 1:
        newreference = ["", "", ""]
    else:
        indextriple = newreference
        newcandidate = ["", "", ""]

    refdictlist_dict: dict[AttrType, list] = {"SUB": [], "PRED": [], "OBJ": []}
    canddictlist_dict: dict[AttrType, list] = {"SUB": [], "PRED": [], "OBJ": []}
    totaldictlist_dict: dict[AttrType, list] = {"SUB": [], "PRED": [], "OBJ": []}
    found: dict[AttrType, yn] = {"SUB": "n", "PRED": "n", "OBJ": "n"}

    # Let's go over each attribute of the triple one by one
    attr_types: List[AttrType] = ["SUB", "PRED", "OBJ"]
    for attr_i, attr_type in enumerate(attr_types):
        refsub = newreference[attr_i]
        candsub = newcandidate[attr_i]

        reflist: List[str] = nltk.word_tokenize(refsub)
        candlist: List[str] = nltk.word_tokenize(candsub)

        reflist = [
            x.lower()
            for x in reflist
            if re.search(r"^[" + re.escape(string.punctuation) + r"]+$", x) == None
        ]
        candlist = [
            x.lower()
            for x in candlist
            if re.search(r"^[" + re.escape(string.punctuation) + r"]$", x) == None
        ]

        # Start with an ngram the full number of words in the reference
        ngramlength = len(candlist)
        reflist, candlist = nonrefwords(reflist, candlist, 1, ngramlength)

        candidatefound, refdictlist, canddictlist, totallist = getrefdict(
            reflist,
            candlist,
            attr_type,
            attr_type,
            sum(len(lst) for lst in totaldictlist_dict.values()),
        )
        found[attr_type] = candidatefound
        refdictlist_dict[attr_type] = refdictlist
        canddictlist_dict[attr_type] = canddictlist
        totaldictlist_dict[attr_type] = totallist

    # Then, try:
    # sub / obj and obj / sub
    # sub / pred and pred / sub
    # pred / obj and obj / pred

    switchmatchfound = "n"
    # If no matches were found for two or more attributes, we are
    # going to try and compare different attributes to each other.
    # First let's try to match the candidate subject and reference
    # object (and vice versa)
    if (found["SUB"] == "n") and (found["OBJ"] == "n"):
        refsub = newreference[0]
        candsub = newcandidate[2]

        reflist = nltk.word_tokenize(refsub)
        candlist = nltk.word_tokenize(candsub)

        reflist = [
            x.lower()
            for x in reflist
            if re.search(r"[" + re.escape(string.punctuation) + r"]", x) is None
        ]
        candlist = [
            x.lower()
            for x in candlist
            if re.search(r"[" + re.escape(string.punctuation) + r"]", x) is None
        ]

        # Start with an ngram the full number of words in the candidate
        ngramlength = len(candlist)
        newreflist, newcandlist = nonrefwords(
            reflist.copy(), candlist.copy(), 1, ngramlength
        )
        candidatefound, refdictlist, canddictlist, totallist = getrefdict(
            newreflist, newcandlist, "SUB", "OBJ", 0
        )

        # Start with an ngram the full number of words in the candidate
        ngramlength = len(newcandlist)
        newreflist, newcandlist = nonrefwords(
            reflist.copy(), candlist.copy(), 1, ngramlength
        )
        candidatefound2, refdictlist2, canddictlist2, totallist2 = getrefdict(
            newreflist,
            newcandlist,
            "OBJ",
            "SUB",
            len(totallist) + len(totaldictlist_dict["PRED"]),
        )

        if (candidatefound == "y") or (candidatefound2 == "y"):
            found["SUB"] = candidatefound
            refdictlist_dict["SUB"] = refdictlist
            canddictlist_dict["SUB"] = canddictlist
            totaldictlist_dict["SUB"] = totallist
            found["OBJ"] = candidatefound2
            refdictlist_dict["OBJ"] = refdictlist2
            canddictlist_dict["OBJ"] = canddictlist2
            totaldictlist_dict["OBJ"] = totallist2

            candidatefound, refdictlist, canddictlist, totallist = getrefdict(
                newreflist, newcandlist, "PRED", "PRED", len(totaldictlist_dict["SUB"])
            )
            found["PRED"] = "y"
            refdictlist_dict["PRED"] = refdictlist
            canddictlist_dict["PRED"] = canddictlist
            totaldictlist_dict["PRED"] = totallist

            switchmatchfound = "y"

    # Then, let's try to switch subject and predicate
    if ((found["SUB"] == "n") and (found["PRED"] == "n")) and (switchmatchfound == "n"):
        refsub = newreference[0]
        candsub = newcandidate[1]

        reflist = nltk.word_tokenize(refsub)
        candlist = nltk.word_tokenize(candsub)

        reflist = [
            x.lower()
            for x in reflist
            if re.search(r"[" + re.escape(string.punctuation) + r"]", x) == None
        ]
        candlist = [
            x.lower()
            for x in candlist
            if re.search(r"[" + re.escape(string.punctuation) + r"]", x) == None
        ]

        newreflist = reflist.copy()
        newcandlist = candlist.copy()
        # Start with an ngram the full number of words in the candidate
        ngramlength = len(newcandlist)
        newreflist, newcandlist = nonrefwords(newreflist, newcandlist, 1, ngramlength)

        candidatefound, refdictlist, canddictlist, totallist = getrefdict(
            newreflist, newcandlist, "SUB", "PRED", 0
        )

        refsub = newreference[1]
        candsub = newcandidate[0]

        reflist = nltk.word_tokenize(refsub)
        candlist = nltk.word_tokenize(candsub)

        reflist = [
            x.lower()
            for x in reflist
            if re.search(r"[" + re.escape(string.punctuation) + r"]", x) == None
        ]
        candlist = [
            x.lower()
            for x in candlist
            if re.search(r"[" + re.escape(string.punctuation) + r"]", x) == None
        ]

        newreflist = reflist.copy()
        newcandlist = candlist.copy()
        # Start with an ngram the full number of words in the candidate
        ngramlength = len(newcandlist)
        newreflist, newcandlist = nonrefwords(newreflist, newcandlist, 1, ngramlength)

        candidatefound2, refdictlist2, canddictlist2, totallist2 = getrefdict(
            newreflist, newcandlist, "PRED", "SUB", len(totallist)
        )

        if (candidatefound == "y") or (candidatefound2 == "y"):
            subjectfound = candidatefound
            subjectreflist = refdictlist.copy()
            subjectcandlist = canddictlist.copy()
            subjecttotallist = totallist.copy()
            predicatefound = candidatefound2
            predicatereflist = refdictlist2.copy()
            predicatecandlist = canddictlist2.copy()
            predicatetotallist = totallist2.copy()
            switchmatchfound = "y"
        else:
            switchmatchfound = "n"

    # Finally, let's try to switch predicate and object
    if ((predicatefound == "n") and (objectfound == "n")) and (switchmatchfound == "n"):
        refsub = newreference[1]
        candsub = newcandidate[2]

        reflist = nltk.word_tokenize(refsub)
        candlist = nltk.word_tokenize(candsub)

        reflist = [
            x.lower()
            for x in reflist
            if re.search(r"[" + re.escape(string.punctuation) + r"]", x) == None
        ]
        candlist = [
            x.lower()
            for x in candlist
            if re.search(r"[" + re.escape(string.punctuation) + r"]", x) == None
        ]

        newreflist = reflist.copy()
        newcandlist = candlist.copy()
        # Start with an ngram the full number of words in the candidate
        ngramlength = len(newcandlist)
        newreflist, newcandlist = nonrefwords(newreflist, newcandlist, 1, ngramlength)

        candidatefound, refdictlist, canddictlist, totallist = getrefdict(
            newreflist, newcandlist, "PRED", "OBJ", len(subjecttotallist)
        )

        refsub = newreference[2]
        candsub = newcandidate[1]

        reflist = nltk.word_tokenize(refsub)
        candlist = nltk.word_tokenize(candsub)

        reflist = [
            x.lower()
            for x in reflist
            if re.search(r"[" + re.escape(string.punctuation) + r"]", x) == None
        ]
        candlist = [
            x.lower()
            for x in candlist
            if re.search(r"[" + re.escape(string.punctuation) + r"]", x) == None
        ]

        newreflist = reflist.copy()
        newcandlist = candlist.copy()
        # Start with an ngram the full number of words in the candidate
        ngramlength = len(newcandlist)
        newreflist, newcandlist = nonrefwords(newreflist, newcandlist, 1, ngramlength)

        candidatefound2, refdictlist2, canddictlist2, totallist2 = getrefdict(
            newreflist,
            newcandlist,
            "OBJ",
            "PRED",
            len(subjecttotallist) + len(totallist),
        )

        if (candidatefound == "y") or (candidatefound2 == "y"):
            predicatefound = candidatefound
            predicatereflist = refdictlist.copy()
            predicatecandlist = canddictlist.copy()
            predicatetotallist = totallist.copy()
            objectfound = candidatefound2
            objectreflist = refdictlist2.copy()
            objectcandlist = canddictlist2.copy()
            objecttotallist = totallist2.copy()
            switchmatchfound = "y"
        else:
            switchmatchfound = "n"

    allrefdict = subjectreflist + predicatereflist + objectreflist
    allcanddict = subjectcandlist + predicatecandlist + objectcandlist
    alltotallist = subjecttotallist + predicatetotallist + objecttotallist

    evaluator = Evaluator([allrefdict], [allcanddict], tags=["SUB", "PRED", "OBJ"])

    # Returns overall metrics and metrics for each tag
    results, results_per_tag = evaluator.evaluate()

    return results, results_per_tag
