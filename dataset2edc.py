# This script converts a generated text-to-quadruple dataset into the
# format expected by EDC. More specifically:
#
# The input dataset is a list of dictionaries, each representing a
# text/quadruple pair. Each dict has the following entries:
#
# - "subject"     : the subject of the predictate
# - "relation"    : the relation between subject/object
# - "object"      : the object of the relation
# - "timestamp"   : timestamp of the relation, in YYYY-MM-DD format
# - "description" : description of the quadruple. The description
#                   should be used to recover the quadruple.
#
# The script outputs is as follows:
#
# - ./dsets/{dataset_name}.json : quadruple description, one per line
# - ./evaluate/references/{dataset_name}.json : quadruple to extract,
#   one per line, in the format "[['subj', 'rel', 'obj', 'ts']]"
from typing import Tuple
import argparse, json, re
import pathlib as pl

Quad = Tuple[str, str, str, str]


def string_lstrip(s: str, to_strip: str) -> str:
    try:
        s = s[s.index(to_strip) + len(to_strip) :]
    except ValueError:
        pass
    return s


def clean_prefix(elt: str) -> str:
    elt = string_lstrip(elt, "yago:")
    elt = string_lstrip(elt, "schema:")
    return elt


def clean_quad_prefix(quad: Quad) -> Quad:
    subj, rel, obj, ts = quad
    return (clean_prefix(subj), clean_prefix(rel), clean_prefix(obj), clean_prefix(ts))


def parse_hex_unicode(hex_unicode: str) -> str:
    assert hex_unicode.startswith("u")
    return chr(int(hex_unicode[1:], base=16))


def clean_unicode(elt: str) -> str:
    return re.sub(r"_u[0-9A-E]{4}", lambda m: parse_hex_unicode(m.group()[1:]), elt)


def clean_quad_unicode(quad: Quad) -> Quad:
    subj, rel, obj, ts = quad
    return (clean_unicode(subj), rel, clean_unicode(obj), ts)


def clean_underscore(elt: str) -> str:
    elt = re.sub(r"_$", "", elt)
    elt = re.sub(r"_+", " ", elt)
    return elt


def clean_quad_underscore(quad: Quad) -> Quad:
    subj, rel, obj, ts = quad
    return (clean_underscore(subj), rel, clean_underscore(obj), ts)


def clean_wiki_id(elt: str) -> str:
    return re.sub(r"Q[0-9]+", "", elt)


def clean_quad_wiki_id(quad: Quad) -> Quad:
    subj, rel, obj, ts = quad
    return (clean_wiki_id(subj), rel, clean_wiki_id(obj), ts)


def escape_single_quotes(elt: str) -> str:
    return re.sub(r"'", "\\'", elt)


def escape_quad_single_quotes(quad: Quad) -> Quad:
    subj, rel, obj, ts = quad
    return (
        escape_single_quotes(subj),
        escape_single_quotes(rel),
        escape_single_quotes(obj),
        ts,
    )


def format_quad(quad: Quad) -> Quad:
    quad = clean_quad_prefix(quad)
    quad = clean_quad_unicode(quad)
    quad = clean_quad_wiki_id(quad)
    quad = clean_quad_underscore(quad)
    quad = escape_quad_single_quotes(quad)
    return quad


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", type=pl.Path)
    args = parser.parse_args()

    with open(args.input_file) as f:
        data = json.load(f)

    desc_path = pl.Path("./dsets/") / f"{args.input_file.stem}.txt"
    print(f"writing to {desc_path}...", end="")
    with open(desc_path, "w") as f:
        for quad in data:
            description = re.sub(r"\n", "", quad["description"])
            f.write(f"{description}\n")
    print("done!")

    ref_path = pl.Path("./evaluate/references/") / f"{args.input_file.stem}.txt"
    print(f"writing to {ref_path}...", end="")
    with open(ref_path, "w") as f:
        for quad in data:
            quad = format_quad(
                (quad["subject"], quad["relation"], quad["object"], quad["timestamp"])
            )
            subj, rel, obj, ts = quad
            f.write(f"[['{subj}', '{rel}', '{obj}', '{ts}']]\n")
    print("done!")
