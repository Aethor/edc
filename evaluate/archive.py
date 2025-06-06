import re, string
import nltk
from nervaluate import Evaluator
from evaluate.evaluation_script import getrefdict, nonrefwords


def evaluaterefcand_core(reference: str, candidate: str) -> tuple[list, list]:
    newreference = reference.split(" | ")
    newcandidate = candidate.split(" | ")

    # Make sure that reference or candidate aren't '' values originally.
    if (len(newreference) > 1) and (len(newcandidate) > 1):
        indextriple = newreference
    elif len(newreference) == 1:
        indextriple = newcandidate
        newreference = ["", "", ""]
    else:
        indextriple = newreference
        newcandidate = ["", "", ""]

    subjectreflist = None
    subjectcandlist = None
    subjecttotallist = None
    predicatereflist = None
    predicatecandlist = None
    predicatetotallist = None
    objectreflist = None
    objectcandlist = None
    objecttotallist = None
    subjectfound = ""
    predicatefound = ""
    objectfound = ""

    for idx, attrib in enumerate(indextriple):
        # Let's go over each attribute of the triple one by one
        refsub = newreference[idx]
        candsub = newcandidate[idx]

        reflist: list[str] = nltk.word_tokenize(refsub)
        candlist: list[str] = nltk.word_tokenize(candsub)

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

        newreflist = reflist.copy()
        newcandlist = candlist.copy()
        # Start with an ngram the full number of words in the reference
        ngramlength = len(newcandlist)
        newreflist, newcandlist = nonrefwords(newreflist, newcandlist, 1, ngramlength)
        if idx == 0:
            candidatefound, refdictlist, canddictlist, totallist = getrefdict(
                newreflist, newcandlist, "SUB", "SUB", 0
            )
            subjectfound = candidatefound
            subjectreflist = refdictlist.copy()
            subjectcandlist = canddictlist.copy()
            subjecttotallist = totallist.copy()
        elif idx == 1:
            candidatefound, refdictlist, canddictlist, totallist = getrefdict(
                newreflist, newcandlist, "PRED", "PRED", len(subjecttotallist)
            )
            predicatefound = candidatefound
            predicatereflist = refdictlist.copy()
            predicatecandlist = canddictlist.copy()
            predicatetotallist = totallist.copy()
        else:
            candidatefound, refdictlist, canddictlist, totallist = getrefdict(
                newreflist,
                newcandlist,
                "OBJ",
                "OBJ",
                len(subjecttotallist) + len(predicatetotallist),
            )
            objectfound = candidatefound
            objectreflist = refdictlist.copy()
            objectcandlist = canddictlist.copy()
            objecttotallist = totallist.copy()

    switchmatchfound = "n"
    # If no matches were found for two or more attributes, we are going to try and compare different attributes to each other.
    # First let's try to match the candidate subject and reference object (and vice versa)
    if (subjectfound == "n") and (objectfound == "n"):
        refsub = newreference[0]
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
            newreflist, newcandlist, "SUB", "OBJ", 0
        )

        refsub = newreference[2]
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
            newreflist,
            newcandlist,
            "OBJ",
            "SUB",
            len(totallist) + len(predicatetotallist),
        )

        if (candidatefound == "y") or (candidatefound2 == "y"):
            subjectfound = candidatefound
            subjectreflist = refdictlist.copy()
            subjectcandlist = canddictlist.copy()
            subjecttotallist = totallist.copy()
            objectfound = candidatefound2
            objectreflist = refdictlist2.copy()
            objectcandlist = canddictlist2.copy()
            objecttotallist = totallist2.copy()

            candidatefound, refdictlist, canddictlist, totallist = getrefdict(
                newreflist, newcandlist, "PRED", "PRED", len(subjecttotallist)
            )
            predicatefound = candidatefound
            predicatereflist = refdictlist.copy()
            predicatecandlist = canddictlist.copy()
            predicatetotallist = totallist.copy()

            switchmatchfound = "y"
        else:
            switchmatchfound = "n"

    # Then, let's try to switch subject and predicate
    if ((subjectfound == "n") and (predicatefound == "n")) and (
        switchmatchfound == "n"
    ):
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

    return allrefdict, allcanddict


def evaluaterefcand(reference: str, candidate: str) -> tuple[dict, dict]:
    """The original version of evaluaterefcand, used to test the new version in evaluation_script.py (see tests/test_evaluaterefcand.py)"""

    allrefdict, allcanddict = evaluaterefcand_core(reference, candidate)
    evaluator = Evaluator([allrefdict], [allcanddict], tags=["SUB", "PRED", "OBJ"])

    # Returns overall metrics and metrics for each tag
    results, results_per_tag = evaluator.evaluate()

    return results, results_per_tag
