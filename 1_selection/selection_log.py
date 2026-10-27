"""Step 1b -- trace every SC3 candidate through the selection phases and count the outcomes.

Inputs : outputs/sc3/sc3_kept.csv                (step 1)
         data/selection/first_pass_decisions.csv  decisions taken in the preliminary screening (Zotero library)
         data/selection/manual_decisions.csv      studies read in full text and dropped, merged or renamed
         data/primary_studies_coding.csv          the included studies
         data/selection/manual_counts.csv         counts that no file records per paper (surveys of the full selection)
         outputs/sc3/sc3_summary.csv              counts of step 1
Outputs: outputs/selection/selection_log.csv     one row per candidate (plus included studies outside the candidate set)
         outputs/selection/selection_counts.csv  number of candidates per outcome
         outputs/selection/figure1_counts.csv    every number of Figure 1 and of the selection paragraphs of Section 4,
                                                  computed from the log (the paper reports exactly these)

Only decisions that are recorded in the inputs are counted. The full-selection phase (second screening) has no
per-record record of exclusions or surveys, so its non-included candidates form a single outcome.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sc3_peer_review_filter import norm_title  # noqa: E402


def read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    candidates = read(PACKAGE_ROOT / "outputs" / "sc3" / "sc3_kept.csv")
    first_pass = {norm_title(r["title"]): r["first_pass_status"] for r in read(PACKAGE_ROOT / "data/selection/first_pass_decisions.csv")}
    manual = {norm_title(r["title"]): r for r in read(PACKAGE_ROOT / "data/selection/manual_decisions.csv")}
    studies = read(PACKAGE_ROOT / "data" / "primary_studies_coding.csv")
    study_by_title = {norm_title(s["paper_title"]): s for s in studies}

    log, matched = [], set()
    for cand in candidates:
        key = norm_title(cand["title"])
        status = first_pass.get(key, "not_in_first_pass")
        decision = manual.get(key)
        study = study_by_title.get(key)
        if study is None and decision and decision["counterpart_in_coding_table"]:
            study = study_by_title.get(norm_title(decision["counterpart_in_coding_table"]))
        if study is not None and decision and decision["decision"] == "duplicate_version":
            outcome = "preliminary: selected, merged into the included journal version"
            study = None
        elif study is not None:
            matched.add(study["study_id"])
            if status == "selected_coded":
                outcome = ("preliminary: selected, replaced by its extended version"
                           if decision and decision["decision"] == "replaced_by_extended_version"
                           else "preliminary: selected and kept")
            elif status == "excluded":
                outcome = "preliminary: excluded in first pass, included later"
            else:
                outcome = "full selection: included"
        elif status == "selected_coded":
            outcome = "preliminary: selected, dropped after reading" if decision else "UNEXPLAINED: selected but absent from the table"
        else:
            outcome = {"confirmed_not_coded": "preliminary: confirmed, never coded",
                       "excluded": "preliminary: excluded (relevance)",
                       "survey": "preliminary: excluded (survey / review)",
                       "snowballing_only": "full selection: not included (excluded or survey)",
                       "not_in_first_pass": "full selection: not included (excluded or survey)"}[status]
        log.append({"title": cand["title"], "year": cand["year"], "venue": cand["venue"], "doi": cand["doi"],
                    "sc3_rule": cand["sc3_rule"], "first_pass_status": status,
                    "study_id": study["study_id"] if study else "",
                    "selection_phase": study["selection_phase"] if study else "",
                    "outcome": outcome, "note": decision["reason"] if decision else ""})

    missing = [s for s in studies if s["study_id"] not in matched]
    for s in missing:  # included studies that are not SC3 candidates under their own title
        log.append({"title": s["paper_title"], "year": "", "venue": "", "doi": "", "sc3_rule": "", "first_pass_status": "",
                    "study_id": s["study_id"], "selection_phase": s["selection_phase"],
                    "outcome": "INCLUDED STUDY NOT MATCHED TO A CANDIDATE", "note": ""})

    out = PACKAGE_ROOT / "outputs" / "selection"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "selection_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(log[0]))
        writer.writeheader()
        writer.writerows(log)

    counts = Counter(r["outcome"] for r in log)
    phases = Counter(s["selection_phase"] for s in studies)
    rows = [("SC3 candidate publications", len(candidates))]
    rows += sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    rows += [(f"included studies, selection_phase = {p}", n) for p, n in sorted(phases.items())]
    rows += [("included studies (rows of primary_studies_coding.csv)", len(studies))]
    with (out / "selection_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["outcome", "count"])
        writer.writerows(rows)
    for label, n in rows:
        print(f"{label:70}{n:6}")

    # ---- Figure 1 / Section 4 numbers ---------------------------------------------------------------
    sc3_summary = {r["quantity"]: int(r["count"]) for r in read(PACKAGE_ROOT / "outputs" / "sc3" / "sc3_summary.csv")
                   if not r["quantity"].startswith(" ")}
    manual_counts = {r["quantity"]: int(r["count"]) for r in read(PACKAGE_ROOT / "data/selection/manual_counts.csv")}
    n = Counter(r["outcome"] for r in log)
    pilot = phases["pilot"]
    # The first screening selected 91 studies besides the pilot. Six of them did not stay in the corpus: two never coded,
    # three dropped after reading, one a conference version of a paper whose journal version was also selected (merged into its full version).
    removed = {"not relevant on closer examination": n["preliminary: selected, dropped after reading"] + n["preliminary: confirmed, never coded"],
               "conference version of a paper whose journal version was also selected": n["preliminary: selected, merged into the included journal version"]}
    selected_total = (n["preliminary: selected and kept"] + n["preliminary: selected, replaced by its extended version"]
                      + sum(removed.values()))                      # everything the first screening selected, pilot included
    first_pass_exclusions = n["preliminary: excluded (relevance)"]
    first_pass_surveys = n["preliminary: excluded (survey / review)"]
    reconsidered = n["preliminary: excluded in first pass, included later"]  # a full-selection candidate (it is coded in the full phase)
    preliminary_pool = selected_total + first_pass_exclusions + first_pass_surveys
    full_pool = sc3_summary["candidate publications kept by SC3"] - preliminary_pool
    full_included = phases["full"]
    full_surveys = manual_counts["full_selection_surveys"]
    not_included_full = n["full selection: not included (excluded or survey)"]
    full_exclusions = not_included_full - full_surveys
    initial_corpus = selected_total - sum(removed.values())
    figure1 = [
        ("records returned by the Scopus query", sc3_summary["records returned by the Scopus query (unique titles)"]),
        ("excluded by SC3", sc3_summary["excluded by SC3"]),
        ("candidate publications", sc3_summary["candidate publications kept by SC3"]),
        ("preliminary pool", preliminary_pool),
        ("  pilot studies", pilot),
        ("  selected in the first screening (without pilot)", selected_total - pilot),
        ("  relevance exclusions", first_pass_exclusions),
        ("  surveys and reviews", first_pass_surveys),
        ("removed from the 91 selected studies", sum(removed.values())),
        ("  not relevant on closer examination", removed["not relevant on closer examination"]),
        ("  conference version of a paper whose journal version was also selected", removed["conference version of a paper whose journal version was also selected"]),
        ("initial corpus (pilot + selected - removed)", initial_corpus),
        ("remaining for full selection", full_pool),
        ("  included in full selection", full_included),
        ("  relevance exclusions in full selection", full_exclusions),
        ("  surveys and reviews in full selection", full_surveys),
        ("total relevance exclusions (first screening + removed + full selection)", first_pass_exclusions + sum(removed.values()) + full_exclusions),
        ("total surveys and reviews", first_pass_surveys + full_surveys),
        ("final corpus", initial_corpus + full_included),
    ]
    checks = {
        "preliminary pool + full pool = candidates": preliminary_pool + full_pool == sc3_summary["candidate publications kept by SC3"],
        "full pool = included + exclusions + surveys": full_pool == full_included + full_exclusions + full_surveys + 0,
        "all outcomes add up to the candidates": sum(n.values()) - n["INCLUDED STUDY NOT MATCHED TO A CANDIDATE"] == sc3_summary["candidate publications kept by SC3"],
        "final corpus = rows of the coding table": initial_corpus + full_included == len(studies),
        "exclusions + surveys + corpus = candidates": (first_pass_exclusions + sum(removed.values()) + full_exclusions + first_pass_surveys
                                                      + full_surveys + initial_corpus + full_included) == sc3_summary["candidate publications kept by SC3"],
    }
    # the study reinstated from the first-pass exclusions belongs to the full selection pool
    assert reconsidered == 1
    for name, ok in checks.items():
        if not ok:
            raise SystemExit(f"CHECK FAILED: {name}")
    with (out / "figure1_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["quantity", "count"])
        writer.writerows(figure1)
    print()
    for label, value in figure1:
        print(f"{label:70}{value:6}")


if __name__ == "__main__":
    main()
