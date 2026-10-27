# Replication package

Data and scripts behind the numbers, tables and figures of the paper (study selection, results, discussion counts, appendix).

```
pip install -r requirements.txt
bash run_all.sh        # about two minutes; results go to outputs/
```

```
data/
  scopus_search_2026-05-04.ris     Scopus export (4,195 unique records)
  primary_studies_coding.csv       the 235 included studies with their coding and categories
  primary_studies.bib              BibTeX of the included studies
  rq5_demographics.csv             year, venue type, affiliation and country per study
  selection/                       records of the manual screening
1_selection/                       SC3 peer-review filter and selection counts (Figure 1)
2_results/                         scripts that build the tables and figures (report_cells.py holds the category counts they use)
outputs/                           generated results (file names as included by the paper)
appendix.pdf                       the paper appendix as a separate PDF (tables listing the primary studies per category)
```

`selection_phase` in the coding table is `pilot` (10), `preliminary` (85) or `full` (140).
The Scopus export is bibliographic metadata from Scopus; check the licence before redistributing it.
