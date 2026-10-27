"""Step 1 -- selection constraint SC3 (peer review), applied to the Scopus export.

SC1 (English) and SC2 (year >= 2018) were applied inside the Scopus query. SC3 is applied here, to the
exported records, as a rule on the publisher and source-title metadata: a record is kept if it comes
from IEEE or ACM, or from one of the venues below whose peer-review process was checked one by one
Everything else is discarded.

Run:
    python3 sc3_peer_review_filter.py                      # uses ../data/scopus_search_2026-05-04.ris
    python3 sc3_peer_review_filter.py --ris other.ris --out some_folder

Writes, to ../outputs/sc3/ (or --out):
    sc3_kept.csv         the candidate publications that enter study selection, with the rule that kept them
    sc3_discarded.csv    the publications excluded by SC3
    sc3_summary.csv      the counts reported in Section 4 (records, kept, discarded, kept per rule)
"""
from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

IEEE_ACM_HINTS = (
    "ieee",
    "acm",
    "institute of electrical and electronics engineers",
    "association for computing machinery",
)
IEEE_ACM_DOI_PREFIXES = ("10.1109", "10.23919", "10.1145")

# Springer's generic conference-proceedings series: not retained, because the conference-level review
# cannot be verified from the Scopus metadata.
SPRINGER_CONFERENCE_SERIES = (
    "lecture notes in computer science",
    "lecture notes in electrical engineering",
    "lecture notes in networks and systems",
    "lecture notes in mechanical engineering",
    "advances in intelligent systems and computing",
    "communications in computer and information science",
    "mechanisms and machine science",
    "advances in transdisciplinary engineering",
    "lecture notes of the institute for computer sciences",
)


# --------------------------------------------------------------------------------------------------
# RIS parsing
# --------------------------------------------------------------------------------------------------
def parse_ris(path: Path) -> list[dict]:
    records, current = [], None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"^([A-Z][A-Z0-9])  - (.*)$", raw)
        if not match:
            if current is not None and raw.strip() and current.get("_last"):
                current[current["_last"]][-1] += " " + raw.strip()
            continue
        tag, value = match.group(1), match.group(2).strip()
        if tag == "TY":
            current = {"TY": [value], "_last": "TY"}
            records.append(current)
        elif current is not None:
            current.setdefault(tag, []).append(value)
            current["_last"] = tag
    return records


def get(record: dict, tag: str) -> str:
    return (record.get(tag) or [""])[0].strip()


def norm_title(title: str) -> str:
    title = "".join(c for c in unicodedata.normalize("NFKD", title) if not unicodedata.combining(c)).lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", title).split())


def norm_doi(doi: str) -> str:
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", doi.lower().strip())


# --------------------------------------------------------------------------------------------------
# The SC3 rule
# --------------------------------------------------------------------------------------------------
def sc3(record: dict) -> tuple[bool, str]:
    """Return (kept, rule). `rule` names the clause that kept the record; it is empty for discarded records."""
    publisher = (get(record, "PB") + " " + get(record, "A4")).lower()
    doi = norm_doi(get(record, "DO"))
    if any(hint in publisher for hint in IEEE_ACM_HINTS) or doi.startswith(IEEE_ACM_DOI_PREFIXES):
        return True, "ieee_acm"

    publisher = get(record, "PB").lower()
    venue = get(record, "T2")
    venue_l = venue.lower()
    kind = get(record, "TY")  # JOUR = journal article

    if "mdpi" in publisher or "multidisciplinary digital publishing" in publisher:
        return True, "mdpi"
    if publisher.startswith("elsevier") and kind == "JOUR" and "procedia" not in venue_l:
        return True, "elsevier_journal"
    if ("wiley" in publisher or "sage publications" in publisher or "taylor and francis" in publisher) and kind == "JOUR":
        return True, "wiley_sage_taylorfrancis_journal"
    if venue.startswith("SAE International Journal"):
        return True, "sae_journal"
    if venue == "IFAC-PapersOnLine":
        return True, "ifac_papersonline"
    if "asme international mechanical engineering congress" in venue_l:
        return True, "asme_imece"
    if publisher == "spie":
        return True, "spie"
    if "cictp" in venue_l or "international conference on transportation and development" in venue_l:
        return True, "asce_transportation_conference"
    if venue == "Springer Proceedings in Advanced Robotics":
        return True, "springer_advanced_robotics"
    if "springer" in publisher and kind == "JOUR" and not any(s in venue_l for s in SPRINGER_CONFERENCE_SERIES):
        return True, "springer_journal"
    if "proceedings of machine learning research" in venue_l or venue_l == "journal of machine learning research":
        return True, "pmlr_jmlr"
    if "neural information processing systems" in venue_l:
        return True, "neurips"
    if publisher == "driving simulation association" or venue == "Actes (IFSTTAR)":
        return True, "driving_simulation_association"
    return False, ""


# --------------------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ris", default=str(PACKAGE_ROOT / "data" / "scopus_search_2026-05-04.ris"))
    parser.add_argument("--out", default=str(PACKAGE_ROOT / "outputs" / "sc3"))
    args = parser.parse_args()

    unique, seen = [], set()
    for record in parse_ris(Path(args.ris)):  # a title is counted once, as in the study-selection counts
        key = norm_title(get(record, "TI"))
        if key and key not in seen:
            seen.add(key)
            unique.append(record)

    kept, discarded = [], []
    for record in unique:
        keep, rule = sc3(record)
        row = {"title": get(record, "TI"), "year": get(record, "PY"), "venue": get(record, "T2"),
               "publisher": get(record, "PB"), "type": get(record, "TY"), "doi": get(record, "DO"),
               "sc3_rule": rule}
        (kept if keep else discarded).append(row)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fields = list(kept[0])
    for name, rows in (("sc3_kept.csv", kept), ("sc3_discarded.csv", discarded)):
        with (out / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    by_rule = Counter(row["sc3_rule"] for row in kept)
    summary = [("records returned by the Scopus query (unique titles)", len(unique)),
               ("excluded by SC3", len(discarded)),
               ("candidate publications kept by SC3", len(kept))]
    summary += [(f"  kept by rule: {rule}", n) for rule, n in by_rule.most_common()]
    with (out / "sc3_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["quantity", "count"])
        writer.writerows(summary)
    for label, n in summary:
        print(f"{label:62}{n:6}")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
