"""Step 2 -- every table and figure of Sections 5-6 and the appendix.

Reads data/primary_studies_coding.csv (the study table, including the final categories of the free-text fields),
data/primary_studies.bib and data/rq5_demographics.csv, and writes ../outputs. The category counts come from
the cell sources in report_cells.py, executed in one shared namespace (build_namespace below); the builders in
BUILDERS then turn them into the files the paper includes.
"""

from __future__ import annotations

import runpy
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path as MarkerPath
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
import sys
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from deep_analysis_common import NO_NAMED_SIMULATOR, validation_technique, BASIS_ORDER, BASIS_COLORS, row_to_bib_key  # noqa: E402
PACKAGE_ROOT = HERE.parent
REPO_ROOT = PACKAGE_ROOT
DATA_DIR = PACKAGE_ROOT / "data"
DASHBOARD_DIR = PACKAGE_ROOT / "outputs"
DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)


def build_namespace() -> dict:
    """Exec the main report's SETUP + RQ1-4 notebook-cell sources in a
    private namespace and hand back every intermediate table it computes."""
    src = runpy.run_path(str(HERE / "report_cells.py"))
    namespace = {
        "REPORT_KEY": "rq_dashboards_source",
        "REPORT_TITLE": "Dashboard source run (internal, not a standalone report)",
        "FILTER_AGGREGATES": False,
        "MIN_COUNT": 1,
        "PRINT_MODE": False,
        "SHOW_TABLES": False,
        "DASHBOARD_SOURCE_ROOT": REPO_ROOT,
    }
    for key in ("SETUP", "RQ1", "RQ2", "RQ3", "RQ4"):
        source = textwrap.dedent(src[key]).strip()
        exec(compile(source, f"<{key}>", "exec"), namespace)
    return namespace


ns = build_namespace()
studies = ns["studies"]
objective_pivot = ns["objective_pivot"]
category_order = ns["category_order"]
long_for = ns["long_for"]
state_long = ns["state_long"]
state_frame = ns["state_frame"]
OBJECTIVE_ORDER = ns["OBJECTIVE_ORDER"]
OBJECTIVE_COLORS = ns["OBJECTIVE_COLORS"]
GROUP_SCHEME = ns["GROUP_SCHEME"]
COMPONENT_SCHEME = ns["COMPONENT_SCHEME"]
STATE_ORDER = ns["STATE_ORDER"]
STATE_COLORS = ns["STATE_COLORS"]
MACRO_LABELS = ns["MACRO_LABELS"]
HOW_STANDALONE_SUBCATEGORIES = ns["HOW_STANDALONE_SUBCATEGORIES"]
what_order = ns["what_order"]
N = len(studies)

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})

# --- Citation tables (Results/Discussion Tables 5-14) -----------------------
# Built from (row, field) data, never hand-typed: every citation key comes
# from row_to_bib_key's title match between the CSV and primary_studies.bib,
# so a study can never silently drift out of the category it is cited for.
ROW_TO_KEY, _UNMATCHED_ROWS, _UNUSED_BIB_KEYS = row_to_bib_key(studies)
if _UNMATCHED_ROWS:
    print(f"CITATION TABLE WARNING: {len(_UNMATCHED_ROWS)} CSV rows have no matching bib key: {_UNMATCHED_ROWS}")


def _latex_escape(s):
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#").replace("\u03bd", r"$\nu$"))


def write_citation_table(rows_spec, filename, caption, label, col_widths=(0.20, 0.32, 0.40), as_float=False):
    """rows_spec: list of (category, subcategory, [source_csv_row, ...]).
    category == '' repeats the previous row's category (no relabel, matches
    the paper's grouped-table style); a \\midrule is inserted whenever a new
    non-empty category starts. Every source_csv_row is resolved to its bib
    key via ROW_TO_KEY -- any row not in ROW_TO_KEY raises immediately
    rather than being silently dropped from the citation list. as_float writes a one-page table
    float (RQ4.5) instead of a multi-page longtable (the execution-profile appendix table)."""
    missing = sorted({r for _, _, rows in rows_spec for r in rows if r not in ROW_TO_KEY})
    if missing:
        raise ValueError(f"{filename}: rows with no bib key: {missing}")
    lines = [
        r"{\footnotesize",
        r"\begin{longtable}{p{%.2f\textwidth} p{%.2f\textwidth} p{%.2f\textwidth}}" % col_widths,
        r"\caption{%s} \label{%s} \\" % (caption, label),
        r"\toprule", r"\textbf{Category} & \textbf{Subcategory} & \textbf{Citations} \\", r"\midrule",
        r"\endfirsthead",
        r"\toprule", r"\textbf{Category} & \textbf{Subcategory} & \textbf{Citations} \\", r"\midrule",
        r"\endhead",
        r"\bottomrule", r"\endfoot",
    ]
    if as_float:
        lines = [r"\begin{table}[t]", r"\footnotesize", r"\centering",
                 r"\caption{%s} \label{%s}" % (caption, label),
                 r"\begin{tabular}{@{}p{%.2f\textwidth} p{%.2f\textwidth} p{%.2f\textwidth}@{}}" % col_widths,
                 r"\toprule", r"\textbf{Category} & \textbf{Subcategory} & \textbf{Citations} \\", r"\midrule"]
    for i, (cat, sub, rows) in enumerate(rows_spec):
        if cat and i > 0:
            lines.append(r"\midrule")
        keys = ",".join(ROW_TO_KEY[r] for r in rows)
        lines.append(f"{_latex_escape(cat)} & {_latex_escape(sub)} & \\cite{{{keys}}} \\\\")
    if as_float:
        lines[-1] += r" \bottomrule"
        lines += [r"\end{tabular}", r"\end{table}"]
        (DASHBOARD_DIR / filename).write_text("\n".join(lines))   # no trailing newline, as in the tex file
        return
    lines += [r"\end{longtable}", r"}"]
    (DASHBOARD_DIR / filename).write_text("\n".join(lines) + "\n")


def write_graphical_table(rows_spec, filename, caption, label, first_header, flat=False, label_width=0.30,
                          variant="float", placement="!htbp", pages=1, bar_width=None, page_head="", page_tail="",
                          total=0.9, font=None, stretch=1.0, widths=None, gap=False, blank_after_centering=False,
                          comment_below=None, font_split=False, label_col=None):
    """Graphical results table built from rows_spec, a list of dicts with
        level  0 (category), 1 (subcategory) or 2 (sub-subcategory)
        label  display text (LaTeX-escaped here)
        n      study count, or None for a pure section header (no bar, no count)
        rows   source_csv_rows whose citations are listed, or None for a parent
               row that carries no citations of its own
    A row's count must equal the number of distinct studies in `rows` when `rows` is given --
    a mismatch raises instead of being silently displayed. One bar scale per table (the largest
    n), via \\ResultBar (see paper.tex). Citation keys come only from ROW_TO_KEY.

    variant
      "float"     category | bar + n | citations, as a non-splitting table float (tables that
                  fit on one page)
      "compact"   category | bar + n, no citations (main-text half of a table whose citation
                  version spans several pages); a multi-page longtable if it needs to be
      "appendix"  category | n | citations as a multi-page longtable with repeated headers
                  (the complete mapping from categories to primary studies)

    Layout options (the tables' hand-tuned shapes live in the builders, not in this function):
      widths        explicit column widths, e.g. (r"0.35\\textwidth", ...), replacing the computed ones
      gap           separate top-level groups by a 0.5em gap (with the rule kept as a comment)
                  instead of a rule
      comment_below child rows with fewer studies than this are kept in the file but commented out
      blank_after_centering / font_split / label_col   reproduce the exact line layout of the tex files"""
    missing = sorted({r for spec in rows_spec for r in (spec["rows"] or []) if r not in ROW_TO_KEY})
    if missing:
        raise ValueError(f"{filename}: rows with no bib key: {missing}")
    for spec in rows_spec:
        if spec["rows"] is not None and spec["n"] != len(set(spec["rows"])):
            raise ValueError(f"{filename}: {spec['label']!r} count {spec['n']} != {len(set(spec['rows']))} studies")
    max_n = max(spec["n"] for spec in rows_spec if spec["n"] is not None)
    # table width as a fraction of the column width (0.9 unless a table asks otherwise)
    font = font or (r"\scriptsize" if variant == "compact" else r"\small")
    lw = r"\dimexpr%.3f\columnwidth\relax" % (total * label_width)
    studies_w = r"\dimexpr\ResultBarMax+2.3em\relax" if variant != "appendix" else r"3.2em"
    cite_w = r"\dimexpr%s\columnwidth-%s-%s-4\tabcolsep\relax" % (total, lw, studies_w)
    if variant == "compact":
        lw = r"\dimexpr%s\columnwidth-%s-2\tabcolsep\relax" % (total, studies_w)
    left = r">{\raggedright\arraybackslash}p{%s}" % lw
    if variant == "compact":
        cols, head = f"@{{}}{left} p{{{studies_w}}}@{{}}", r"\textbf{%s} & \textbf{Studies} \\" % first_header
    elif variant == "appendix":
        cols = f"@{{}}{left} >{{\\raggedleft\\arraybackslash}}p{{{studies_w}}} p{{{cite_w}}}@{{}}"
        head = r"\textbf{%s} & \textbf{Studies} & \textbf{Primary studies} \\" % first_header
    else:
        cols, head = f"@{{}}{left} p{{{studies_w}}} p{{{cite_w}}}@{{}}", r"\textbf{%s} & \textbf{Studies} & \textbf{Primary studies} \\" % first_header
    if widths and variant == "float":
        cols = "@{}" + " ".join("p{%s}" % w for w in widths) + "@{}"
    if label_col and variant == "appendix":
        cols = cols.replace("p{%s}" % lw, "p{%s}" % label_col, 1)
    entries = []  # (starts a new top-level group, row line)
    for i, spec in enumerate(rows_spec):
        lvl = spec["level"]  # hanging indent: wrapped lines stay inside the row's own indent
        indent = r"\hangindent=%dem\hangafter=1\hspace{%dem}" % (lvl + 1, lvl) if lvl else r"\hangindent=1em\hangafter=1 "
        text = _latex_escape(spec["label"]).replace("/", r"/\hspace{0pt}")  # allow line breaks after slashes
        text = r"\textbf{%s}" % text if lvl == 0 and not flat else text
        if spec["n"] is None:
            count = ""
        elif variant == "appendix":
            count = "%d" % spec["n"]
        else:
            count = r"\ResultStudies{%d}{%d}" % (spec["n"], max_n)
        cite = "" if spec["rows"] is None else r"\cite{%s}" % ",".join(ROW_TO_KEY[r] for r in spec["rows"])
        row = f"{indent}{text} & {count}" + ("" if variant == "compact" else f" & {cite}") + r" \\"
        if comment_below and lvl > 0 and spec["n"] is not None and spec["n"] < comment_below:
            row = "% " + row
        entries.append((lvl == 0 and i > 0 and not flat, row))

    def block(chunk):
        out = []
        for j, (new_group, line) in enumerate(chunk):
            if new_group and j > 0:
                if gap:
                    out[-1] = out[-1][:-3] + r" \\ [0.5em]"
                    out.append(r"% \midrule")
                else:
                    out.append(r"\midrule")
            out.append(line)
        return out

    if variant == "compact":
        # Non-splitting float(s) at \\scriptsize. A table too tall for one page (pages > 1) becomes
        # consecutive floats cut at top-level category boundaries, all sharing one bar scale.
        starts = [i for i, e in enumerate(entries) if e[0]] + [len(entries)]
        starts = [0] + starts
        cuts = [0]
        for k in range(1, pages):
            target = len(entries) * k / pages
            cuts.append(min((i for i in starts if i > cuts[-1] and i < len(entries)), key=lambda i: abs(i - target)))
        cuts.append(len(entries))
        lines = []
        for k in range(pages):
            chunk = entries[cuts[k]:cuts[k + 1]]
            suffix = "" if k == 0 else "bcdefgh"[k - 1]
            if pages > 1:  # name the categories each part holds, so the text can cite one table per category
                names = ", ".join(sp["label"] for sp in rows_spec[cuts[k]:cuts[k + 1]] if sp["level"] == 0)
                cap = f"{page_head} ({k + 1}/{pages}): {names}. {page_tail}"
                lab = f"{label}-{'abcdefgh'[k]}"
            else:
                cap, lab = caption, label
            lines += [r"\begin{table}[%s]" % placement, r"\caption{%s}" % cap, r"\label{%s}" % lab,
                      r"\centering", *([""] if blank_after_centering else []), font,
                      r"\setlength{\ResultBarMax}{%s}" % (bar_width or "1.6cm"),
                      r"\begin{tabular}{%s}" % cols, r"\toprule", head, r"\midrule", *block(chunk),
                      r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    elif variant == "float":
        lines = [r"\begin{table}[%s]" % placement, r"\caption{%s}" % caption, r"\label{%s}" % label, r"\centering",
                 *([""] if blank_after_centering else []), font,
                 *([r"\renewcommand{\arraystretch}{%s}" % stretch] if stretch != 1.0 else []),
                 *([r"\setlength{\ResultBarMax}{%s}" % bar_width] if bar_width else []),
                 r"\begin{tabular}{%s}" % cols, r"\toprule", head, r"\midrule", *block(entries), r"\bottomrule",
                 r"\end{tabular}", r"\end{table}"]
    else:
        lines = [*([r"{", font] if font_split else [r"{%s" % font]), r"\begin{longtable}{%s}" % cols, r"\caption{%s} \label{%s} \\" % (caption, label),
                 r"\toprule", head, r"\midrule", r"\endfirsthead",
                 r"\toprule", head, r"\midrule", r"\endhead",
                 r"\bottomrule", r"\endfoot", *block(entries), r"\end{longtable}", r"}"]
    (DASHBOARD_DIR / filename).write_text("\n".join(lines) + "\n")


def write_split_tables(rows_spec, stem, caption, label, first_header, app_caption, app_label,
                       flat=False, compact_width=0.56, app_width=0.30, pages=1, page_head="", page_tail="",
                       compact_total=0.9, compact_font=None, compact_opts=None, app_opts=None):
    """A table whose citation version spans several pages: a compact main-text table
    (<stem>_table.tex) plus the complete appendix mapping (<stem>_appendix.tex)."""
    write_graphical_table(rows_spec, f"{stem}_table.tex", caption, label, first_header, flat=flat,
                          label_width=compact_width, variant="compact", pages=pages,
                          page_head=page_head, page_tail=page_tail, total=compact_total, font=compact_font,
                          **(compact_opts or {}))
    write_graphical_table(rows_spec, f"{stem}_appendix.tex", app_caption, app_label, first_header, flat=flat,
                          label_width=app_width, variant="appendix", **(app_opts or {}))


def _gt_row(level, label, rows=None, n=None):
    """Row spec for write_graphical_table; n defaults to the distinct studies in rows."""
    rows = None if rows is None else list(dict.fromkeys(rows))
    return {"level": level, "label": label, "rows": rows, "n": (None if rows is None else len(rows)) if n is None else n}


def _gt_group(label, parent_n, children, level=0):
    """A parent row with child rows; a single child that only repeats the parent's
    own label is folded into the parent (the parent then carries the citations)."""
    norm = lambda t: t.lower().replace("-", " ")
    if len(children) == 1 and norm(children[0][0]) == norm(label):
        return [_gt_row(level, label, children[0][1])]
    return [_gt_row(level, label, None, parent_n)] + [_gt_row(level + 1, l, r) for l, r in children]



# These dashboards are embedded at \linewidth in an acmart "manuscript"
# single-column layout (roughly 6.75in wide in print). A dashboard's
# matplotlib canvas is almost always wider than that -- 12-17in is typical
# once a dense multi-panel grid is laid out -- so it gets shrunk on the
# page. `set_scale` computes exactly how much shrink that will be and
# scales every font size (and the vertical room reserved per bar/label) up
# front, so text lands at a consistently readable size in print instead of
# shrinking along with everything else. A figure at or under
# PRINT_LINEWIDTH_IN gets scale 1.0 (no upscaling needed).
PRINT_LINEWIDTH_IN = 6.75
# Capped so a very wide multi-panel figure's uncapped labels (y-tick text,
# legends) don't grow without bound and invade a neighbouring column's
# space -- widening the figure then genuinely buys more room per panel
# instead of also inflating everyone's text demand at the same rate.
MAX_SCALE = 2.3
_SCALE = 1.0


def set_scale(fig_width_in, max_scale=MAX_SCALE):
    global _SCALE
    _SCALE = min(max(1.0, fig_width_in / PRINT_LINEWIDTH_IN), max_scale)
    plt.rcParams.update({
        "font.size": 12 * _SCALE,
        "axes.titlesize": 14 * _SCALE,
        "axes.labelsize": 13 * _SCALE,
        "xtick.labelsize": 12.5 * _SCALE,
        "ytick.labelsize": 12.5 * _SCALE,
        "legend.fontsize": 12 * _SCALE,
        "legend.title_fontsize": 12.5 * _SCALE,
    })
    return _SCALE


# ================================================================== helpers

LABEL_ALIASES = {
    "No SUT (none, or simulator/testbed fidelity only)": "No SUT (none / testbed only)",
    "No hardware / not specified": "No hardware / unspecified",
}


def short_label(text, width=20):
    """Wraps onto multiple lines instead of cropping with '...' -- the full
    category name always stays readable, just taller instead of shorter.
    Many category names are slash-joined compounds ('generation/
    translation') with no other whitespace to wrap on; treating '/' as a
    breakable point avoids textwrap's default of splitting the word itself
    mid-letter when it doesn't fit."""
    text = LABEL_ALIASES.get(text, text)
    if len(text) <= width:
        return text
    return "\n".join(textwrap.wrap(text.replace("/", "/ "), width=width, break_long_words=False))



MAIN_FACE, MAIN_EDGE = "#eaf2fb", "#2f6f9f"
SUB_FACE, SUB_EDGE, SUB_ACCENT = "#fbfbfd", "#d8dee4", "#2f6f9f"


def style_panel(ax, kind):
    """Give a panel a distinct 'card' look so a reader can tell at a glance
    whether it's the main/overview chart or one of its breakdowns."""
    if kind == "main":
        ax.set_facecolor(MAIN_FACE)
        for spine in ax.spines.values():
            spine.set_edgecolor(MAIN_EDGE)
            spine.set_linewidth(2.4)
    elif kind == "sub":
        ax.set_facecolor(SUB_FACE)
        for spine in ax.spines.values():
            spine.set_edgecolor(SUB_EDGE)
            spine.set_linewidth(1.1)


def panel_title(ax, text, kind, badge=None, show_overview_badge=True):
    """Plain, always-renderable digits (never Unicode circled numerals --
    fonts commonly stop supporting those past ⑩) styled as a small chip so
    a breakdown panel visibly cross-references its row in the main panel.
    The main panel carries no descriptive title of its own -- just the
    small OVERVIEW tag that marks it apart from its numbered breakdown
    panels; the dashboard as a whole has no title either (see `suptitle`).
    `show_overview_badge=False` keeps the "main" card styling (border,
    background) without repeating the OVERVIEW tag -- for a single logical
    overview split across several side-by-side panels, only the first
    should actually say OVERVIEW."""
    if kind == "main":
        if show_overview_badge:
            ax.text(0.996, 1.015, "OVERVIEW", transform=ax.transAxes, ha="right", va="bottom",
                     fontsize=12 * _SCALE, fontweight="bold", color="white",
                     bbox=dict(boxstyle="round,pad=0.32", facecolor=MAIN_EDGE, edgecolor="none"))
    elif kind == "sub":
        label = f"{badge}.  {text}" if badge is not None else text
        ax.set_title(label, fontsize=14.5 * _SCALE, fontweight="bold",
                     color=SUB_ACCENT if badge is not None else "#3d3d3d", loc="left")
    else:
        # Unlike the "sub" case above, callers here pass raw, unwrapped
        # title strings -- wrap them too, or a long title at a big _SCALE
        # overflows this panel's width and collides with its neighbour's
        # title (this bit us for real once font sizes grew).
        ax.set_title(short_label(text, width=32), loc="left")



def axis_width_inches(ax):
    """Return the physical width allocated to an axes. Works for axes owned
    by either a Figure or a SubFigure."""
    owner = ax.figure
    if hasattr(owner, "get_size_inches"):
        owner_width = owner.get_size_inches()[0]
    else:
        owner_width = owner.bbox.width / owner.dpi
    return ax.get_position().width * owner_width

def hbar_objective(ax, long_df, label, order, title, value_fontsize=None, label_width=20,
                    label_fontsize=None, panel_kind=None, badges=None, badge=None, bar_width=.84,
                    xlim_pad=1.12, show_overview_badge=True, x_nbins=None, inline_labels=True):
    """Stacked study counts, with labels above bars to preserve plot width.

    The hardware/target comparison keeps conventional row labels so its bar
    chart can share row coordinates with the adjacent heatmap.
    """
    pivot = objective_pivot(long_df, label, order)
    # Callers pass an explicit, often larger, size to grow text in a panel
    # that has the room for it (e.g. breakdown_grid's 16*scale) -- these
    # must NOT be re-clamped back down to the bare default, or every such
    # override is silently discarded and every panel prints at the same
    # size regardless of how much room it actually has.
    value_fontsize = value_fontsize or 12 * _SCALE
    label_fontsize = label_fontsize or 12 * _SCALE
    panel_width = axis_width_inches(ax)
    # Estimate readable characters from the actual panel allocation.
    wrap_width = max(24, int(panel_width * 72 / (label_fontsize * .52)))
    labels = [short_label(str(i), width=wrap_width if inline_labels else label_width) for i in pivot.index]
    if badges:
        labels = [f"{badges[raw]}. {text}" if raw in badges else text
                  for raw, text in zip(pivot.index, labels)]
    totals = pivot.sum(axis=1).to_numpy()
    xmax = max(totals, default=0)
    if inline_labels:
        y, cursor = [], 0.0
        for text in labels:
            cursor += .82 * (text.count("\n") + 1)
            y.append(cursor)
            cursor += 1.35
        left = np.zeros(len(pivot))
        for col, color in zip(pivot.columns, OBJECTIVE_COLORS):
            values = pivot[col].to_numpy()
            ax.barh(y, values, left=left, height=.72, color=color, label=col, zorder=3)
            left += values
        for pos, text in zip(y, labels):
            ax.text(0, pos - .48, text, transform=ax.get_yaxis_transform(),
                    ha="left", va="bottom", fontsize=label_fontsize, linespacing=1.05)
        ax.set_yticks([])
        ax.set_ylim(cursor - .35, -.45)
    else:
        y = np.arange(len(pivot))
        left = np.zeros(len(pivot))
        for col, color in zip(pivot.columns, OBJECTIVE_COLORS):
            values = pivot[col].to_numpy()
            ax.barh(y, values, left=left, height=bar_width, color=color, label=col, zorder=3)
            left += values
        ax.set_yticks(y, labels, fontsize=label_fontsize)
        ax.invert_yaxis()
    for pos, total in zip(y, totals):
        ax.annotate(str(int(total)), (total, pos), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=value_fontsize, fontweight="bold")
    ax.set_xlim(0, xmax * 1.12 if xmax else 1)
    ax.set_xlabel("Studies")
    ax.set_ylabel("")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=x_nbins or 5))
    ax.grid(axis="x", alpha=.2, zorder=0)
    # Titles wrap noticeably narrower than the row labels above: a title
    # that just barely fits `wrap_width` on one line can still visually
    # overrun into the next panel's column (matplotlib doesn't clip a
    # title to its axes' width), especially once `wspace` is tight in a
    # multi-column grid. Wrapping titles earlier trades one extra title
    # line for guaranteed clearance from the neighbouring panel.
    title_wrap_width = max(14, int(wrap_width * 0.62))
    title = short_label(title.replace("\n", " "), width=title_wrap_width)
    panel_title(ax, title, panel_kind, badge=badge, show_overview_badge=show_overview_badge)
    style_panel(ax, panel_kind)






def state_count_matrix(prefix, scheme):
    long = state_long(prefix, scheme)
    matrix = long.pivot(index="group", columns="state", values="studies").fillna(0)
    return matrix.reindex(index=list(scheme.keys()), columns=STATE_ORDER, fill_value=0)






def state_heatmap_panel(ax, prefix, scheme, title, hm_scale=None):
    # Cell/tick text must stay legible without eating into the narrow
    # gridspec column a dense multi-panel dashboard gives this heatmap, so
    # its font growth is capped well below the full-figure _SCALE by
    # default -- callers with a roomier, single-column panel can pass a
    # larger `hm_scale` explicitly to size past that cap.
    hm_scale = hm_scale if hm_scale is not None else min(_SCALE, 2.1)
    matrix = state_count_matrix(prefix, scheme)
    # Partially real only arises when a group's two raw components disagree
    # (Section 4.4); a single raw component can never produce it, so for
    # this component-level heatmap the column is always zero. Dropping it
    # here avoids a column that reads as "always empty" for every row.
    matrix = matrix.loc[:, (matrix != 0).any(axis=0)]
    im = ax.imshow(matrix.values, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=30, ha="right", fontsize=9 * hm_scale)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=10 * hm_scale)
    vmax = matrix.values.max() or 1
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = int(matrix.values[i, j])
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=9 * hm_scale, fontweight="bold",
                         color="white" if v > vmax * .55 else "#222222")
    ax.set_title(short_label(title, width=32), fontsize=10 * hm_scale, loc="left")
    return im


def heatmap_panel(ax, matrix, title, xlabel="", cmap="Blues", hm_scale=None, xtick_scale=None,
                   cell_scale=None, show_ylabels=True):
    hm_scale = hm_scale if hm_scale is not None else min(_SCALE, 2.1)
    # x-tick labels and cell numbers can each be sized past the general
    # hm_scale independently (e.g. when the panel has vertical room to
    # spare but the same tight column width, so y-label growth needs to
    # stay capped even as the x-axis category names or in-cell counts run
    # bigger -- especially once show_ylabels=False frees that budget up).
    xtick_scale = xtick_scale if xtick_scale is not None else hm_scale
    cell_scale = cell_scale if cell_scale is not None else hm_scale
    im = ax.imshow(matrix.values, cmap=cmap, aspect="auto", vmin=0)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8.5 * xtick_scale)
    ax.set_yticks(range(len(matrix.index)))
    # Skippable when a neighbouring panel in the same dashboard already
    # carries the identical row labels (e.g. a bar chart sharing this
    # heatmap's row order) -- repeating them here would just double up.
    if show_ylabels:
        ax.set_yticklabels([short_label(i, width=28) for i in matrix.index], fontsize=9 * hm_scale)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", left=False)
    vmax = matrix.values.max() or 1
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = int(matrix.values[i, j])
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=8 * cell_scale, fontweight="bold",
                         color="white" if v > vmax * .55 else "#222222")
    ax.set_title(short_label(title, width=32), fontsize=10 * hm_scale, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9 * hm_scale)
    return im


STATE_ICONS = {
    "Not modelled": ("x", "full"),
    "Simulated": ("s", "none"),
    "Pseudo-sim": ("s", "left"),
    "Pseudo-real": ("o", "left"),
    "Mixed": ("D", "left"),
    "Both": (MarkerPath.make_compound_path(
        MarkerPath.unit_circle().transformed(Affine2D().scale(.45).translate(-.6, 0)),
        MarkerPath.unit_circle().transformed(Affine2D().scale(.45).translate(.6, 0))), "none"),
    "Partially real": ("^", "none"),
    "Real": ("o", "full"),
}


def state_icon(state, size):
    marker, fill = STATE_ICONS[state]
    return Line2D([], [], linestyle="none", marker=MarkerStyle(marker, fillstyle=fill),
                  markersize=size, markeredgewidth=1.3, color="#202020",
                  markerfacecolor="#202020", markerfacecoloralt="white", label=state)


def combo_state_matrix_panel(ax_top, ax_grid, prefix, scheme, title, top_n=None, min_count=1):
    """Every distinct state combination actually used across a scheme's
    groups, as the full report's own two-part chart (`state_combination_chart`
    in report_cells.py) -- an objective-coloured bar per
    combination on `ax_top`, and a monochrome icon grid below it on
    `ax_grid` spelling out the literal state of every group for that
    combination, so a combination is named by its actual states rather
    than an abstract score. Reimplemented here (instead of calling that
    function directly) so it draws into caller-supplied axes and can be
    embedded as one panel of a larger dashboard instead of its own figure.

    `min_count` drops combinations with fewer than that many studies before
    `top_n` is applied -- with six raw components the long tail is mostly
    one-off combinations, and a chart trying to show every single one of
    them becomes unreadable well before `top_n` alone would trim it down.
    Nothing is silently lost: the dropped count is folded into the title."""
    order = list(scheme.keys())
    k = len(order)
    sf = state_frame(prefix, scheme).copy()
    sf["combo"] = list(zip(*[sf[c] for c in order]))
    order_counts_all = sf.groupby("combo")["source_csv_row"].nunique().sort_values(ascending=False)
    order_counts = order_counts_all[order_counts_all >= min_count]
    n_dropped_combos = len(order_counts_all) - len(order_counts)
    n_dropped_studies = int(order_counts_all[order_counts_all < min_count].sum())
    combos = order_counts.index.tolist()
    n = len(combos)
    combo_index = {c: i for i, c in enumerate(combos)}
    sf["combo_idx"] = sf.combo.map(combo_index)
    pivot_all = objective_pivot(sf, "combo_idx", list(range(n)))
    m = min(top_n, n) if top_n is not None else n
    shown_combos = combos[:m]
    pivot = pivot_all.loc[range(m)]
    chart_title = title + (f" -- top {m} of {n} combinations by frequency" if n > m else "")
    if n_dropped_combos:
        chart_title += (f" (+{n_dropped_combos} combinations used by fewer than "
                         f"{min_count} studies each, {n_dropped_studies} studies total, not shown)")

    pivot.plot.bar(stacked=True, ax=ax_top, width=.82, color=OBJECTIVE_COLORS, legend=False)
    totals = pivot.sum(axis=1)
    tmax = totals.max() if len(totals) else 0
    for x, total in enumerate(totals):
        ax_top.text(x, total + tmax * .02, str(int(total)), ha="center", va="bottom", fontsize=11 * _SCALE)
    # Headroom above the tallest bar/count-label so the title never crowds
    # or overlaps them, even once this panel is squeezed into a short row.
    ax_top.set_ylim(0, tmax * 1.22 if tmax else 1)
    ax_top.set_ylabel("Studies", fontsize=13 * _SCALE)
    ax_top.set_title(chart_title, fontsize=15 * _SCALE, loc="left", pad=12 * _SCALE)
    ax_top.grid(axis="y", alpha=.25)
    ax_top.set_xticks([])
    ax_top.set_xlabel("")
    ax_top.tick_params(axis="y", labelsize=11 * _SCALE)
    # Kept inside the axes (rather than the previous bbox_to_anchor past
    # the right edge) so this panel can use the plot's full width instead
    # of ceding a column to the legend -- the tallest bars sit at the left
    # (frequency-ranked), leaving the upper-right corner clear.
    ax_top.legend(title="Objective", loc="upper right", framealpha=0.9,
                   fontsize=11 * _SCALE, title_fontsize=11.5 * _SCALE)

    ax_top.set_xlim(-.6, m - .4)
    ax_grid.set_xlim(ax_top.get_xlim())
    ax_grid.set_ylim(k - .5, -.5)
    ax_grid.set_facecolor("white")
    icon_size = min(12 * _SCALE, axis_width_inches(ax_grid) * 72 / max(m, 1) * .52)
    for x, combo in enumerate(shown_combos):
        if x % 2 == 0:
            ax_grid.axvspan(x - .5, x + .5, color="#f1f3f5", zorder=0)
        for row, state in enumerate(combo):
            artist = state_icon(state, icon_size)
            artist.set_data([x], [row])
            ax_grid.add_line(artist)
    for row in np.arange(.5, k, 1):
        ax_grid.axhline(row, color="#dddddd", linewidth=.7)
    ax_grid.set_yticks(range(k))
    ax_grid.set_yticklabels(order, fontsize=13 * _SCALE)
    ax_grid.set_xticks([])
    ax_grid.tick_params(axis="y", length=0)
    for spine in ax_grid.spines.values():
        spine.set_visible(False)
    handles = [state_icon(state, 11 * _SCALE) for state in STATE_ORDER]
    ax_grid.legend(handles=handles, title="Execution category", loc="upper center", bbox_to_anchor=(0.5, -0.12),
                   ncol=4, fontsize=10 * _SCALE, title_fontsize=11 * _SCALE,
                   handletextpad=.6, columnspacing=1.4)


def what_target_heatmap_matrix(column, min_studies):
    """Recreates the RQ4.1/4.2 platform/hardware x WHAT-target heatmap
    matrices exactly as the main report computes them."""
    cw_long = studies.assign(category=studies[column].str.split(" | ", regex=False)).explode("category")
    cw_long = cw_long.merge(ns["motivation_long"][["source_csv_row", "atomic_what"]], on="source_csv_row", how="left")
    cw_counts_all = cw_long.groupby(["category", "atomic_what"])["source_csv_row"].nunique().rename("studies").reset_index()
    cw_totals = cw_long.groupby("category").source_csv_row.nunique().sort_values(ascending=False)
    cw_order = cw_totals[cw_totals >= min_studies].index.tolist()
    cw_counts = cw_counts_all[cw_counts_all.category.isin(cw_order)]
    return cw_counts.pivot(index="category", columns="atomic_what", values="studies").fillna(0).reindex(
        index=cw_order, columns=what_order, fill_value=0)






def legend_for_objective(fig, y=0.005):
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in OBJECTIVE_COLORS]
    fig.legend(handles, OBJECTIVE_ORDER, title="Objective", loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, y), fontsize=13 * _SCALE, title_fontsize=14 * _SCALE)


def save(fig, name):
    """Always saves with a 'tight' bbox: the canvas grows to fit whatever
    was actually drawn, so nothing -- a wrapped label, a wide legend -- is
    ever cropped, even if that means the page is no longer an exact A4."""
    fig.savefig(DASHBOARD_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(DASHBOARD_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)




# =============================================================== RQ1 dashboards



def build_rq1_dashboard():
    """RQ1.1 (primary objective), RQ1.2 (addressed gap aspect), and RQ1.3
    (direction) merged into a single figure -- each RQ's own marginal
    distribution only, side by side rather than split across three
    separate floats. The RQ1.2 x RQ1.3 cross-tab (direction broken down by
    gap aspect) previously lived in RQ1.3's own panel here; it is a
    cross-cutting read relating two RQs, not RQ1.3's own distribution, so
    it moved to Discussion as build_rq1_3c, next to the existing
    direction-by-objective cross-cut (build_rq1_3b)."""
    gap_long = long_for(studies.gap_type, "gap_type")
    gap_order = category_order(gap_long, "gap_type")
    direction_order = ["sim2real", "real2sim", "both"]
    dir_long = long_for(studies.direction, "direction")

    width = 30
    scale = set_scale(width)
    top_h = 6.6
    bottom_h = 2.6 + 0.72 * len(gap_order)
    fig = plt.figure(figsize=(width, (top_h + bottom_h) * scale), constrained_layout=True)
    subfig_top, subfig_bottom = fig.subfigures(2, 1, height_ratios=[top_h, bottom_h])
    axes_top = subfig_top.subplots(1, 2)

    # RQ1.1: a plain bar, not stacked by objective -- the rows here already
    # are the objective categories, so stacking each bar by objective would
    # be circular (every bar would be one solid colour, its own category).
    ax = axes_top[0]
    obj_counts = studies.objective_group.value_counts().reindex(OBJECTIVE_ORDER).fillna(0)
    ax.barh(OBJECTIVE_ORDER[::-1], obj_counts[::-1], color=OBJECTIVE_COLORS[::-1], height=.68, zorder=3)
    for y, v in enumerate(obj_counts[::-1]):
        ax.text(v + obj_counts.max() * .02, y, f"{int(v)} ({100 * v / N:.1f}%)", va="center",
                 fontsize=15 * scale, fontweight="bold")
    ax.set_xlim(0, obj_counts.max() * 1.22)
    ax.set_xlabel("Studies")
    ax.tick_params(axis="y", labelsize=15 * scale)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="x", alpha=.25, zorder=0)
    panel_title(ax, "RQ1.1 -- Primary objective", "sub")
    style_panel(ax, "sub")

    hbar_objective(axes_top[1], dir_long, "direction", direction_order, "RQ1.3 -- Direction",
                    panel_kind="sub", value_fontsize=15 * scale, label_fontsize=15 * scale)

    ax_bottom = subfig_bottom.subplots(1, 1)
    hbar_objective(ax_bottom, gap_long, "gap_type", gap_order, "RQ1.2 -- Addressed gap aspect",
                    panel_kind="sub", label_width=30, value_fontsize=15 * scale, label_fontsize=15 * scale)

    save(fig, "RQ1_dashboard")


# =============================================================== RQ2 dashboards





def build_rq2_3():
    """Approach execution-state dashboard, page-sized like the RQ1.2
    dashboard: the full six-component grid on top (finest granularity, one
    row per raw component), and below it the full report's own group-level
    (Scene/Sensing/Control) combination chart -- every distinct combination
    actually used, named by its literal per-group states rather than an
    abstract score, with its own objective/state legends. The previous
    'top 10' abbreviated bar chart and the middle 'state shares' panel from
    the original two-figure (group/component) layout are both dropped in
    favour of this single, richer, full-page chart."""
    width = 34
    scale = set_scale(width)
    fig = plt.figure(figsize=(width, 16 * scale), constrained_layout=True)
    subfig_top, subfig_bot = fig.subfigures(2, 1, height_ratios=[1, 1.6])

    ax_heat = subfig_top.subplots(1, 1)
    state_heatmap_panel(ax_heat, "approach", COMPONENT_SCHEME, "Component execution states (counts)",
                         hm_scale=scale * 1.35)

    ax_bar, ax_grid = subfig_bot.subplots(2, 1, gridspec_kw={"height_ratios": [1.6, 1.1], "hspace": 0.03})
    combo_state_matrix_panel(ax_bar, ax_grid, "approach", GROUP_SCHEME,
                              "Approach -- every group combination", top_n=None, min_count=2)

    save(fig, "RQ2_3_dashboard")




# =============================================================== RQ3 dashboards



def build_rq3_2():
    """Validation-reference execution-state dashboard, built the same way
    as RQ2.3: the full six-component grid on top, and below it the full
    report's own group-level (Scene/Sensing/Control) combination chart --
    every distinct combination actually used, named by its literal
    per-group states. The previous 'top 10' abbreviated bar chart and the
    middle 'state shares' panel from the original two-figure
    (group/component) layout are both dropped, as is the approach-vs-
    reference comparison panel this dashboard briefly carried -- it
    duplicated RQ2.3/RQ3.2's own state data without adding a distinct
    view, so it was dropped rather than kept as a third panel here."""
    width = 34
    scale = set_scale(width)
    fig = plt.figure(figsize=(width, 16 * scale), constrained_layout=True)
    subfig_top, subfig_bot = fig.subfigures(2, 1, height_ratios=[1, 1.6])

    ax_heat = subfig_top.subplots(1, 1)
    state_heatmap_panel(ax_heat, "reference", COMPONENT_SCHEME, "Component execution states (counts)",
                         hm_scale=scale * 1.35)

    ax_bar, ax_grid = subfig_bot.subplots(2, 1, gridspec_kw={"height_ratios": [1.6, 1.1], "hspace": 0.03})
    combo_state_matrix_panel(ax_bar, ax_grid, "reference", GROUP_SCHEME,
                              "Reference -- every group combination", top_n=None, min_count=2)

    save(fig, "RQ3_2_dashboard")


def rq3_3_metrics():
    """Metric ordering, badge numbers and per-category subtag breakdowns --
    shared by the overview (build_rq3_3) and its breakdown grid
    (build_rq3_3b)."""
    metric_order = ns["metric_counts"].sort_values("studies", ascending=False).metric_category.tolist()
    cats = ["Sensor/perception comparison", "ADS behaviour/trajectory comparison",
            "Vehicle dynamics/actuation comparison", "Safety/event outcome",
            "Runtime/computational", "Qualitative assessment"]
    badges = {cat: i + 1 for i, cat in enumerate(cats)}
    metric_sub_long = ns["metric_sub_long"]
    top_n = 8
    sub_data = []
    for cat in cats:
        sub = (metric_sub_long[metric_sub_long.metric_category == cat]
               .groupby("metric_subtag")["source_csv_row"].nunique()
               .sort_values(ascending=False))
        sub_data.append((cat, sub.index.tolist()[:top_n], len(sub) > top_n))
    return metric_order, badges, metric_sub_long, sub_data


def build_rq3_2_matched():
    """Component states and independent audited study-level execution sources."""
    approach = state_frame("approach", COMPONENT_SCHEME).set_index("source_csv_row")
    reference = state_frame("reference", COMPONENT_SCHEME).set_index("source_csv_row")
    rows = []
    for component in COMPONENT_SCHEME:
        idx = approach.index[approach[component] != "Not modelled"]
        values = reference.loc[idx, component]
        counts = values.value_counts()
        rows.append({"component": component, "approach_present": len(values),
                     **{state: int(counts.get(state, 0)) for state in STATE_ORDER}})
    pd.DataFrame(rows).to_csv(DASHBOARD_DIR / "RQ3_2_matched_components.csv", index=False)












# =============================================================== RQ4 dashboards





def build_rq4_2():
    """Hardware platform by WHAT target (the cross-analysis). The overall hardware
    counts that used to sit beside it are now in the integrated hardware table
    (build_rq4_2_table), so the heatmap carries its own row labels."""
    width = 20
    scale = set_scale(width)
    hw_long = studies.assign(category=studies.hardware_tags.str.split(" | ", regex=False)).explode("category")
    # "No hardware / not specified" is an absence, not a platform, so it is not a heatmap row.
    hw_long = hw_long[hw_long.category != "No hardware / not specified"]
    hw_order = hw_long.groupby("category")["source_csv_row"].nunique().sort_values(ascending=False).index.tolist()
    height = (3.0 + 0.5 * len(hw_order)) * scale
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    hw_what_matrix = what_target_heatmap_matrix("hardware_tags", 1).drop(
        index="No hardware / not specified", errors="ignore").reindex(index=hw_order, fill_value=0)
    heatmap_panel(ax, hw_what_matrix, "Hardware platform by research target",
                   hm_scale=scale * 1.0, xtick_scale=scale * 1.3, cell_scale=scale * 1.3)
    save(fig, "RQ4_2_dashboard")


def rq4_3_suts():
    """SUT ordering, badge numbers and per-category technology breakdowns --
    shared by the overview (build_rq4_3) and its breakdown grid
    (build_rq4_3b)."""
    top_n, min_studies = 8, 5
    sut_long = studies.assign(category=studies.sut_tags.str.split(" | ", regex=False)).explode("category")
    # "No SUT" is an absence of a system under test, not a system, so it is
    # reported in the text but kept out of the figures.
    sut_long = sut_long[sut_long.category != "No SUT (none, or simulator/testbed fidelity only)"]
    sut_counts = sut_long.groupby("category")["source_csv_row"].nunique().sort_values(ascending=False)
    sut_order = sut_counts.index.tolist()
    non_technology = ns["SUT_NON_TECHNOLOGY_CATEGORIES"]

    sub_data = []
    for cat in sut_order:
        if cat in non_technology or sut_counts[cat] < min_studies:
            continue
        members = studies[studies.sut_tags.str.contains(cat, regex=False, na=False)].copy()
        members["sut_subtag"] = members.apply(
            lambda row: ns["sut_subtags"](cat, row.sut, row.source_csv_row), axis=1)
        sub_long = members.assign(
            sut_subtag=members.sut_subtag.str.split(" | ", regex=False)).explode("sut_subtag")
        order = (sub_long.groupby("sut_subtag")["source_csv_row"].nunique()
                 .sort_values(ascending=False).index.tolist())
        if len(order) <= 1:
            continue
        sub_data.append((cat, sub_long, order[:top_n], len(order) > top_n))

    badges = {cat: i + 1 for i, (cat, *_rest) in enumerate(sub_data)}
    return sut_long, sut_order, badges, sub_data










# ======================================================== extra: technique taxonomies

# Every RQ2.2 atomic HOW tag that is itself a validation/measurement act
# (not a mitigation mechanism) -- same set used to derive the HOW macro
# categories in the source report, kept here so a study's *mitigation* strategy can
# be read off its how_tags with the measurement-only tags excluded first.
MEASUREMENT_HOW_TAGS_STRICT = {
    "Unclassified sim–real comparison (needs manual review)", "SUT-behaviour sim–real comparison",
    "Ego-vehicle-dynamics comparison", "Task-based sim–real measurement", "Task-based planner evaluation",
    "Feature-space sim–real measurement", "Radar sim–real measurement", "LiDAR sim–real measurement",
    "Multi-sensor sim–real measurement", "Sim–generated–real measurement", "Image-quality measurement",
    "Perception-performance measurement", "Simulation-only scenario scoring",
    "Reconstruction-error trajectory-anomaly scoring calibrated to human authenticity ratings",
}

# The 6 mitigation-strategy families -- every non-measurement HOW tag in
# the mapping resolves to exactly one of these (verified: zero unassigned
# tags across all 145 Mitigate-only/Measure-and-mitigate studies).
MITIGATION_FAMILIES = {
    "Generative content synthesis": {
        "Diffusion-based generation", "GAN-based generation/translation", "Neural rendering (NeRF/3DGS)",
        "Synthetic perturbation/augmentation", "Image compositing", "Point-cloud/image reprojection & warping",
        "Physics-based multi-sensor object insertion", "Synthetic data generation/augmentation",
    },
    "Real-world evidence reconstruction": {
        "Recorded driving/sensor-log reconstruction", "Crash/accident-evidence reconstruction",
        "Map/aerial-derived reconstruction", "Reconstruction-strategy comparison (classical vs. neural assets)",
        "Scenario-generation framework/toolchain", "Retrieval-augmented domain adaptation",
        "Domain adaptation", "Data-driven simulator construction",
    },
    "Real-equipment integration (ViL/HiL/MR/XiL)": {
        "Vehicle-in-the-loop (on-road)", "Vehicle-in-the-loop (testbench)", "Small-scale vehicle-in-the-loop",
        "Physical scale-model / surrogate-platform validation",
        "Mixed reality", "Controller/ECU hardware-in-the-loop", "Steering-actuator hardware-in-the-loop",
        "Camera sensor-in-the-loop", "Radar sensor-in-the-loop", "LiDAR sensor-in-the-loop",
        "Modular X-in-the-loop framework", "Co-simulation", "Human-in-the-loop", "XiL/HIL",
    },
    "Sensor physics modelling": {
        "LiDAR sensor modelling", "Camera sensor modelling", "Radar sensor modelling",
        "Camera+LiDAR+radar sensor modelling", "Compute/timing-fidelity emulation",
    },
    "Traffic/agent behaviour modelling": {
        "Learning-based (imitation/RL) traffic models", "Rule-based/behaviour-tree traffic models",
        "Traffic/behaviour-model calibration", "Route/trajectory-conditioned behaviour modelling",
        "Learned/generative trajectory models", "Optimization/search-based generation",
        "Statistical/distributional traffic calibration", "Field-test/traffic-flow calibration",
        "Human-reactive traffic models", "Traffic/NPC behaviour modelling",
    },
    "Ego policy/control transfer": {
        "Vehicle-dynamics/controller calibration", "Reinforcement-learning-based policy transfer",
        "Imitation-learning-based policy transfer", "Generative-video-assisted policy transfer",
    },
}


def mitigation_family(tag):
    for name, members in MITIGATION_FAMILIES.items():
        if tag in members:
            return name
    return "Other/unclassified"


def mitigation_tags(how_tags):
    return [t for t in how_tags.split(" | ") if t not in MEASUREMENT_HOW_TAGS_STRICT]


def rq2_2b_families():
    """Cohort, family ordering, badge numbers and per-family tag breakdowns --
    shared by the macro-category overview (build_rq2_2b) and its breakdown grid
    (build_rq2_2f) so the two figures can never disagree.

    Family assignment is multi-label: a study using techniques from more than
    one of the six families is counted in every family it touches, not
    collapsed into a separate "Multiple families" bucket that would make the
    per-family counts undercount how many studies actually use each family's
    techniques."""
    cohort = studies[studies.objective_group.isin(["Mitigate only", "Measure and mitigate"])].copy()
    cohort["mit_tags"] = cohort.how_tags.map(mitigation_tags)
    cohort["families"] = cohort.mit_tags.map(
        lambda ts: sorted(set(mitigation_family(t) for t in ts)) if ts else ["No mitigation mechanism coded"])
    long = cohort.explode("families").rename(columns={"families": "family_label"})
    family_order = long.groupby("family_label").source_csv_row.nunique().sort_values(ascending=False).index.tolist()
    badges = {name: i + 1 for i, name in enumerate(f for f in family_order if f in MITIGATION_FAMILIES)}
    sub_data = []
    for name in badges:
        fam_tags = long[long.family_label == name].assign(
            tag=lambda d: d.mit_tags).explode("tag")[["source_csv_row", "objective_group", "tag"]]
        fam_tags = fam_tags[fam_tags.tag.map(mitigation_family) == name]
        order = fam_tags.groupby("tag").source_csv_row.nunique().sort_values(ascending=False).index.tolist()
        sub_data.append((name, fam_tags, order))
    return long, family_order, badges, sub_data























FAMILY_SHORT = {
    "Traffic/agent behaviour modelling": "Traffic/agent",
    "Generative content synthesis": "Generative",
    "Real-equipment integration (ViL/HiL/MR/XiL)": "Real-equipment",
    "Multiple families": "Multiple",
    "Ego policy/control transfer": "Ego policy",
    "Sensor physics modelling": "Sensor physics",
    "Real-world evidence reconstruction": "Real-world recon.",
}


TIER_ORDER = ["Fully simulated", "Calibrated simulation", "Real-data reuse",
              "Single-component physical integration", "Multi-component physical integration", "Fully real"]
TIER_COLORS = {"Fully simulated": "#4E79A7", "Calibrated simulation": "#76B7B2",
                "Real-data reuse": "#59A14F", "Single-component physical integration": "#EDC948",
                "Multi-component physical integration": "#F28E2B", "Fully real": "#E15759"}
LIVE_EQUIPMENT_TAGS = {"Real vehicle", "Real hardware"}
REAL_ADJACENT_COMPONENT_STATES = {"Real", "Mixed", "Both", "Pseudo-real"}


def execution_tier(n_real_components, basis_tags_set):
    """Places one study into one of six ordered execution-realness tiers,
    using the audited approach-execution-basis tags (BASIS_ORDER) together
    with n_real_components -- how many of the six raw RQ2.3 components
    (scenario, actors, perception, localization, actuation, ECU/control)
    are coded Real, Mixed, Both, or Pseudo-real, i.e. carry any real-world
    content at all. The basis tags draw the same live-vs-reused distinction
    on both sides of the sim/real divide: 'Real vehicle'/'Real hardware'
    (the study itself drove a car or ran a bench) versus 'Real dataset' (an
    existing real recording was reused), and 'Simulator' (the study itself
    ran a simulation) versus 'Simulated dataset' (an existing synthetic
    recording was reused). Only a live Simulator run -- not a reused
    synthetic dataset -- counts as 'simulation' for the Calibrated-
    simulation tier below: a study that only combines a reused real dataset
    with a reused synthetic dataset never actually ran anything, so it is
    data reuse, not a calibrated simulation, however synthetic the reused
    data is. A reused synthetic dataset with nothing else present still
    lands in Fully simulated, since it carries no real-world grounding
    either way. Among studies that used a physical vehicle or hardware and
    also ran a simulator, 'Single-component' (one of the six components
    carries real-world content) versus 'Multi-component' (two or more do)
    is a plain count, not a claim about which components or how they
    relate. A study that used a physical vehicle or hardware and never ran
    a simulator at all is Fully real regardless of that count."""
    live_simulator = "Simulator" in basis_tags_set
    any_simulated = live_simulator or "Simulated dataset" in basis_tags_set
    live_equipment = bool(LIVE_EQUIPMENT_TAGS & basis_tags_set)
    uses_real_dataset = "Real dataset" in basis_tags_set
    if live_equipment:
        if not live_simulator:
            return "Fully real"
        if n_real_components >= 2:
            return "Multi-component physical integration"
        return "Single-component physical integration"
    if uses_real_dataset:
        return "Calibrated simulation" if live_simulator else "Real-data reuse"
    if any_simulated:
        return "Fully simulated"
    return "Other profiles"


def s5_tiered_cohort():
    """The 145 mitigation-objective studies, each placed into an execution
    tier and its RQ2.2 mitigation-family label. Shared by the figure and
    its citation table so the two can never disagree."""
    m = studies[studies.objective_group.isin(["Mitigate only", "Measure and mitigate"])].copy()
    m["mit_tags"] = m.how_tags.map(mitigation_tags)
    m["families"] = m.mit_tags.map(
        lambda ts: sorted(set(mitigation_family(t) for t in ts)) if ts else ["No mechanism coded"])
    m["family_label"] = m.families.map(lambda fs: fs[0] if len(fs) == 1 else "Multiple families")
    comp_cols = list(COMPONENT_SCHEME.keys())
    sf = state_frame("approach", COMPONENT_SCHEME)[["source_csv_row", "approach_basis"] + comp_cols].copy()
    sf["n_real_components"] = sf[comp_cols].apply(
        lambda row: sum(v in REAL_ADJACENT_COMPONENT_STATES for v in row), axis=1)
    sf["basis_set"] = sf.approach_basis.map(lambda v: set(str(v).split(" + ")))
    sf["tier"] = sf.apply(lambda r: execution_tier(r.n_real_components, r.basis_set), axis=1)
    return sf[["source_csv_row", "tier"]].merge(m[["source_csv_row", "family_label"]], on="source_csv_row", how="inner")


def build_s5_execution_taxonomy():
    """Cross-cutting figure for the Discussion's 'Execution Sources by
    Mitigation Family' subsubsection: an icicle chart placing the
    mitigation-objective studies into six execution-realness tiers ordered
    sim-to-real, each split into its dominant mitigation families, with a
    top strip mapping the traditional SiL/HiL/ViL/RW in-the-loop ladder
    onto the same axis -- MiL is left off since it is usually distinguished
    from SiL by software maturity, which this taxonomy cannot see -- to
    show which of our tiers that ladder has no rung for. A single shared
    axes (rather than three stacked subplots) lets all three bands share
    one coordinate system and lets thin connector lines join a tier
    segment directly to its family sub-segments below."""
    merged = s5_tiered_cohort()
    others_n = int((merged.tier == "Other profiles").sum())
    merged.groupby(["tier", "family_label"]).source_csv_row.nunique().rename("studies").reset_index().to_csv(
        DASHBOARD_DIR / "S5_execution_tiers_by_family.csv", index=False)

    tier_counts = merged.tier.value_counts().reindex(TIER_ORDER, fill_value=0)
    total = int(tier_counts.sum())

    lefts, x = {}, 0
    for t in TIER_ORDER:
        lefts[t] = x
        x += tier_counts[t]

    # Family band made taller (0.85 -> 1.25 data-units) so a rotated label
    # has more room to read top-to-bottom before it runs out of band; every
    # other band keeps the same offset and height it always had. The whole
    # y-layout is computed before the figure exists so the figure's own
    # height can be derived from it (in inches-per-data-unit, held at the
    # same ratio the old, shorter layout used) -- otherwise a taller family
    # band just steals room from a fixed-height canvas instead of actually
    # being physically taller on the page.
    y_fam, h_fam = 0.15, 1.25
    y_tier, h_tier = y_fam + h_fam + 0.35, 0.85
    n_stagger_levels = 3
    stagger_span = 0.14 + 0.22 * (n_stagger_levels - 1)
    y_ladder, h_ladder = y_tier + h_tier + stagger_span + 0.35, 0.45
    y_span = y_ladder + h_ladder + 0.25
    INCHES_PER_Y_UNIT = 6.2 / 4.05  # calibrated against the original (h_fam=0.85) layout's span

    width = 34
    scale = set_scale(width)
    fig, ax = plt.subplots(figsize=(width, y_span * INCHES_PER_Y_UNIT * scale), constrained_layout=True)
    ax.set_xlim(0, total)
    ax.set_ylim(0, y_span)
    # A fixed "n >= threshold" cutoff (the tier band's threshold was tuned
    # for an older, more even tier split) silently stops working once one
    # tier's share shifts -- e.g. the 38-study 'Component-level physical
    # integration' segment is *wider* than before yet its label now
    # overflows, because the label text itself is longer than any old tier
    # name. Measuring each candidate label's actual rendered width against
    # its segment's actual rendered width (both in the same pixel space)
    # is what the old fixed counts were really a proxy for, and stays
    # correct regardless of how the tier/family counts shift in a future
    # re-audit.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_per_unit_x = ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0]
    px_per_unit_y = ax.transData.transform((0, 1))[1] - ax.transData.transform((0, 0))[1]

    def text_fits(n, text, fontsize, fontweight="normal", pad=0.85):
        """Does this (unrotated) label's rendered width fit within its
        n-study-wide segment?"""
        probe = ax.text(0, -10, text, fontsize=fontsize, fontweight=fontweight)
        text_width_px = probe.get_window_extent(renderer=renderer).width
        probe.remove()
        return n * px_per_unit_x * pad >= text_width_px

    def rotated_fits(text, fontsize, band_height, pad=0.85):
        """Does this label, rotated 90 degrees to read down the band, fit
        within the band's height? A rotated Text's window extent already
        swaps width/height for the rotation, so the *string's* length ends
        up in the extent's height, not its width -- checking width here
        (as the unrotated helper does) would always trivially pass, since
        a single line's rotated width is just its unrotated line height."""
        probe = ax.text(0, -10, text, fontsize=fontsize, rotation=90)
        text_height_px = probe.get_window_extent(renderer=renderer).height
        probe.remove()
        return band_height * px_per_unit_y * pad >= text_height_px

    # Connectors: thin vertical guides from each tier's boundaries down to
    # its family row, so the "branching" reads as one flow, not two
    # unrelated stacked bars.
    for t in TIER_ORDER:
        n = tier_counts[t]
        if n == 0:
            continue
        ax.plot([lefts[t], lefts[t]], [y_fam + h_fam, y_tier], color="#dee2e6", linewidth=1, zorder=0)
        ax.plot([lefts[t] + n, lefts[t] + n], [y_fam + h_fam, y_tier], color="#dee2e6", linewidth=1, zorder=0)

    # Smaller than the family/ladder text: the tier band carries the
    # longest labels ("Single-component physical integration", 37 chars),
    # so it needs the most help fitting inside its own segment.
    TIER_FONT, TIER_FALLBACK_FONT = 11 * scale, 9.5 * scale
    n_narrow = 0
    for t in TIER_ORDER:
        n = tier_counts[t]
        if n == 0:
            continue
        ax.barh(y_tier, n, left=lefts[t], height=h_tier, color=TIER_COLORS[t],
                edgecolor="white", linewidth=1.5, align="edge")
        cx = lefts[t] + n / 2
        inside_text = f"{t}\n({n})"
        if text_fits(n, inside_text, TIER_FONT, fontweight="bold"):
            # Wide enough: label sits inside the segment.
            text_color = "black" if t == "Single-component physical integration" else "white"
            ax.text(cx, y_tier + h_tier / 2, inside_text, ha="center", va="center",
                    fontsize=TIER_FONT, fontweight="bold", color=text_color)
        else:
            # Too narrow for inside text: short label above the band with a
            # thin leader line down to the segment. Consecutive narrow
            # segments stagger the label height across three levels so
            # adjacent leader lines don't collide even when several narrow
            # segments sit side by side (two levels was enough only while
            # at most every other tier needed the fallback).
            y0 = y_tier + h_tier + 0.14 + 0.22 * (n_narrow % n_stagger_levels)
            ax.plot([cx, cx], [y_tier + h_tier, y0], color="#495057", linewidth=0.8)
            ax.text(cx, y0 + 0.04, f"{t}\n({n})", ha="center", va="bottom",
                    fontsize=TIER_FALLBACK_FONT, fontweight="bold", color="#212529")
            n_narrow += 1

    for t in TIER_ORDER:
        sub = merged[merged.tier == t]
        if len(sub) == 0:
            continue
        fam_counts = sub.family_label.value_counts()
        big = fam_counts[fam_counts >= 2].copy()
        small_n = int(fam_counts[fam_counts < 2].sum())
        if small_n:
            big["other"] = small_n
        x0 = lefts[t]
        fam_max = big.max()
        for fam, n in big.items():
            is_other = fam == "other"
            short = "other" if is_other else FAMILY_SHORT.get(fam, fam)
            color = "#e9ecef" if is_other else TIER_COLORS[t]
            alpha = 1.0 if is_other else 0.5 + 0.5 * (n / fam_max)
            ax.barh(y_fam, n, left=x0, height=h_fam, color=color, alpha=alpha,
                    edgecolor="white", linewidth=1.2, align="edge")
            cx = x0 + n / 2
            wide_text = f"{short}\n({n})"
            rotated_text = f"{short} ({n})"
            if text_fits(n, wide_text, 12 * scale):
                # Wide segment: horizontal label fits inside.
                ax.text(cx, y_fam + h_fam / 2, wide_text, ha="center", va="center", fontsize=12 * scale)
            elif rotated_fits(rotated_text, 10.5 * scale, h_fam):
                # Narrow segment: rotate 90 degrees so the label reads down
                # the (now taller) band instead of overflowing sideways
                # into its neighbours.
                ax.text(cx, y_fam + h_fam / 2, rotated_text, ha="center", va="center",
                        fontsize=10.5 * scale, rotation=90)
            # Anything that doesn't fit even rotated gets no label: its own
            # text would be taller than the family band and bleed into the
            # tier band above -- the citation table and CSV carry that
            # detail instead.
            x0 += n

    # MiL (model-in-the-loop) is deliberately left off this ladder: it is
    # usually distinguished from SiL by software maturity (an abstract
    # model versus production code), a distinction our real/simulated
    # execution-level taxonomy cannot see, so placing it on this axis would
    # be a guess rather than a finding. HiL and ViL are shown as separate
    # spans rather than combined, each mapped to its own tier.
    ladder_spans = [
        ("SiL", ["Fully simulated"], "#4E79A7"),
        ("no traditional term", ["Calibrated simulation", "Real-data reuse"], "#adb5bd"),
        ("HiL", ["Single-component physical integration"], "#EDC948"),
        ("ViL", ["Multi-component physical integration"], "#F28E2B"),
        ("RW", ["Fully real"], "#E15759"),
    ]
    for label, tiers_in_span, color in ladder_spans:
        x0 = lefts[tiers_in_span[0]]
        x1 = lefts[tiers_in_span[-1]] + tier_counts[tiers_in_span[-1]]
        n_span = int(sum(tier_counts[t] for t in tiers_in_span))
        ax.barh(y_ladder, x1 - x0, left=x0, height=h_ladder, color=color, alpha=0.35,
                edgecolor=color, linewidth=1.5, align="edge")
        weight = "bold" if label == "no traditional term" else "normal"
        text = f"{label}\n({n_span})" if (x1 - x0) < 12 else f"{label} ({n_span} studies)"
        ax.text((x0 + x1) / 2, y_ladder + h_ladder / 2, text, ha="center", va="center",
                fontsize=13 * scale, fontweight=weight)

    ax.axis("off")
    ax.set_title("Traditional in-the-loop ladder mapped onto our execution profiles",
                 loc="left", fontsize=16 * scale, pad=10)

    save(fig, "S5_execution_taxonomy")




# =============================================================== RQ5 dashboard

def _rq5_frame():
    """Join the demographics table (data/rq5_demographics.csv) onto
    `studies` by title, so the RQ5 panels can carry the same objective split
    as every other figure in the paper."""
    demo = pd.read_csv(DATA_DIR / "rq5_demographics.csv")

    def key(s):
        return s.str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii") \
                .str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()

    demo["_k"] = key(demo.paper_title)
    left = studies[["source_csv_row", "paper_title", "objective_group"]].copy()
    left["_k"] = key(left.paper_title)
    merged = left.merge(demo.drop(columns=["paper_title"]), on="_k", how="left", validate="1:1")
    missing = merged.year.isna().sum()
    if missing:
        raise ValueError(f"{missing} studies have no demographics row")
    merged["year"] = merged.year.astype(int).astype(str)
    return merged


def build_rq5():
    """RQ5.1-5.4 on one page. The three short panels (year, venue, affiliation)
    stack down the left column and the long country list takes the full-height
    right column, so the page carries no large blank block and the panels still
    read in RQ order."""
    demo = _rq5_frame()
    width = 26
    scale = set_scale(width)

    year_order = sorted(demo.year.unique())
    venue_order = demo.venue_type.value_counts().index.tolist()
    affil_order = [a for a in ["Academia", "Mixed", "Industry", "Not reported"]
                   if a in set(demo.affiliation_type)]
    counts = demo.country.value_counts()
    country_order = counts[counts >= 2].index.tolist()

    left_rows = [len(year_order), len(venue_order), len(affil_order)]
    # +1.6 row-equivalents per panel for its own title/axis furniture
    left_units = sum(r + 1.6 for r in left_rows)
    height = 1.2 + 0.46 * max(left_units, len(country_order) + 1.6)
    fig = plt.figure(figsize=(width, height * scale), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1],
                           height_ratios=[r + 1.6 for r in left_rows],
                           hspace=0.18, wspace=0.18)

    hbar_objective(fig.add_subplot(gs[0, 0]), demo, "year", year_order,
                    "RQ5.1  Publication year", label_width=14, panel_kind="sub",
                    value_fontsize=12 * scale, label_fontsize=12 * scale,
                    bar_width=.72, show_overview_badge=False)

    hbar_objective(fig.add_subplot(gs[1, 0]), demo, "venue_type", venue_order,
                    "RQ5.2  Venue type", label_width=16, panel_kind="sub",
                    value_fontsize=12 * scale, label_fontsize=12 * scale,
                    bar_width=.6, show_overview_badge=False)

    hbar_objective(fig.add_subplot(gs[2, 0]), demo, "affiliation_type", affil_order,
                    "RQ5.3  Affiliation type", label_width=16, panel_kind="sub",
                    value_fontsize=12 * scale, label_fontsize=12 * scale,
                    bar_width=.6, show_overview_badge=False)

    hbar_objective(fig.add_subplot(gs[:, 1]), demo[demo.country.isin(country_order)],
                    "country", country_order, "RQ5.4  Country (2 or more studies)",
                    label_width=20, panel_kind="sub", value_fontsize=12 * scale,
                    label_fontsize=12 * scale, bar_width=.72, show_overview_badge=False)

    legend_for_objective(fig, y=-0.018)
    fig.get_layout_engine().set(rect=[0, 0.07, 1, 1])
    save(fig, "RQ5_dashboard")


# ========================================================== synthesis figures

SYNTHESIS_TARGET_ORDER = [
    "camera perception", "LiDAR perception", "radar perception",
    "traffic/NPC realism", "test-scenario validity", "crash-scenario validity",
    "pedestrian/VRU realism", "road/environment realism",
    "ego-vehicle dynamics/actuation", "SUT behaviour realism",
    "ADS-policy transfer", "ego localization/pose estimation",
]




def target_long_frame(min_studies=5):
    # Uses motivation_long's what_detail (the authoritative RQ2.1 source, see
    # build_rq2_1) rather than a naive studies.what_tags split -- the naive
    # split under-counts some targets by one or two studies each (e.g. 48
    # instead of 49 for camera perception), the same discrepancy already
    # root-caused and fixed for RQ2.1's own reported counts. This keeps the
    # cross-cutting S1/S2 denominators consistent with the RQ2.1 prose.
    long = (ns["motivation_long"][["source_csv_row", "what_detail"]]
            .dropna(subset=["what_detail"]).rename(columns={"what_detail": "what_tag"}))
    # motivation_long capitalizes this one label inconsistently with the
    # rest ("Ego localization/pose estimation" vs. lower-case everywhere
    # else); normalize so it matches SYNTHESIS_TARGET_ORDER instead of
    # silently dropping all 5 of its studies.
    long["what_tag"] = long["what_tag"].replace({"Ego localization/pose estimation": "ego localization/pose estimation"})
    counts = long.groupby("what_tag").source_csv_row.nunique()
    order = [tag for tag in SYNTHESIS_TARGET_ORDER if counts.get(tag, 0) >= min_studies]
    return long[long.what_tag.isin(order)].copy(), order








def build_s2_target_technique_validation():
    """Two heatmaps: mitigation macro categories by target, and measurement
    techniques by target (restricted to Measure only and Measure and
    mitigate studies, since Mitigate only studies do not measure the gap).
    This directly visualizes fragmentation by target."""
    long, order = target_long_frame()

    mit = studies[studies.objective_group.isin(["Mitigate only", "Measure and mitigate"])].copy()
    mit["mit_tags"] = mit.how_tags.map(mitigation_tags)
    mit["family"] = mit.mit_tags.map(lambda ts: sorted(set(mitigation_family(t) for t in ts)) if ts else [])
    mit_long = mit.assign(family=mit.family).explode("family")
    mit_long = mit_long[mit_long.family.isin(MITIGATION_FAMILIES)]
    target_mit = long[["source_csv_row", "what_tag"]].merge(mit_long[["source_csv_row", "family"]], on="source_csv_row")
    family_order = list(MITIGATION_FAMILIES.keys())
    mit_matrix = (target_mit.groupby(["what_tag", "family"]).source_csv_row.nunique()
                  .reset_index().pivot(index="what_tag", columns="family", values="source_csv_row")
                  .fillna(0).reindex(index=order, columns=family_order, fill_value=0))

    val = studies[studies.objective_group.isin(["Measure only", "Measure and mitigate"])].copy()
    val["validation_mode"] = validation_technique(val)
    val_long = val.assign(technique=val.validation_mode.str.split(" + ", regex=False)).explode("technique")
    val_long = val_long[val_long.technique.ne("No substantive validation/demonstration only")]
    val_order = ["Direct or matched sim–real comparison", "Task/SUT-performance comparison",
                 "Quantitative/distribution comparison", "Feature/image-quality comparison",
                 "Human/qualitative assessment"]
    target_val = long[["source_csv_row", "what_tag"]].merge(val_long[["source_csv_row", "technique"]], on="source_csv_row")
    val_matrix = (target_val.groupby(["what_tag", "technique"]).source_csv_row.nunique()
                  .reset_index().pivot(index="what_tag", columns="technique", values="source_csv_row")
                  .fillna(0).reindex(index=order, columns=val_order, fill_value=0))

    mit_matrix.to_csv(DASHBOARD_DIR / "S2_target_by_mitigation_family.csv")
    val_matrix.to_csv(DASHBOARD_DIR / "S2_target_by_measurement_technique.csv")

    width = 28
    scale = set_scale(width)
    fig, axes = plt.subplots(1, 2, figsize=(width, (4.0 + .64 * len(order)) * scale),
                             constrained_layout=True, gridspec_kw={"width_ratios": [1.15, 1]})
    x_explanations = [
        "1 Generative synthesis; 2 Real-world reconstruction; 3 Real-equipment integration; "
        "4 Sensor physics; 5 Traffic/agent behaviour; 6 Policy/control transfer",
        "1 Direct/matched sim-real; 2 Task/SUT performance; 3 Quantitative/distribution; "
        "4 Feature/image quality; 5 Human/qualitative",
    ]
    for ax, matrix, title, cmap, explanation in [
        (axes[0], mit_matrix, "Mitigation macro category by target", "Blues", x_explanations[0]),
        (axes[1], val_matrix, "Measurement technique by target", "Greens", x_explanations[1]),
    ]:
        im = ax.imshow(matrix.values, cmap=cmap, aspect="auto", vmin=0)
        ax.set_xticks(range(len(matrix.columns)), [str(i + 1) for i in range(len(matrix.columns))],
                      fontsize=11 * scale)
        ax.set_xlabel(short_label(explanation, width=58), fontsize=10 * scale, labelpad=8)
        ax.set_yticks(range(len(matrix.index)), [short_label(i, 25) for i in matrix.index],
                      fontsize=11 * scale)
        vmax = matrix.values.max() or 1
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = int(matrix.iloc[i, j])
                if value:
                    ax.text(j, i, str(value), ha="center", va="center", fontsize=10 * scale,
                            fontweight="bold", color="white" if value > vmax * .55 else "#222222")
        ax.set_title(title, loc="left", fontsize=14 * scale)
        fig.colorbar(im, ax=ax, shrink=.7)
    save(fig, "S2_target_technique_validation")








# ======================================================== citation tables (auto-generated)
# Every table below derives its (category, subcategory, [source_csv_row,...])
# rows from the same computed columns the dashboards use, then resolves
# citation keys via ROW_TO_KEY (bib title match) inside write_citation_table.
# No citation key is ever hand-typed here.

# --- Integrated graphical citation tables (replace the former plot + citation-table pairs) -------
# Every row's count and citations come from the same frames the former plots and citation tables
# used; write_graphical_table refuses to emit a row whose count differs from its citation list.

def _check_counts(name, got, expected):
    """Cross-check table counts against the frames the retired plot used."""
    if dict(got) != dict(expected):
        raise ValueError(f"{name}: table counts {dict(got)} differ from plot counts {dict(expected)}")


def build_rq2_1_table():
    ml = ns["motivation_long"]
    macro_display = {
        "Perception": "Perception", "Scenario and actor realism": "Scenario and actor realism",
        "Ego-system behaviour and planning": "Ego system behaviour and planning",
        "Ego localization/pose estimation": "Ego localization/pose estimation",
        "communication/network fidelity": "Communication/network fidelity",
        "small-scale vehicle suitability": "Small scale vehicle suitability",
    }
    macro_order = ml.groupby("what_macro").source_csv_row.nunique().sort_values(ascending=False).index.tolist()
    shown = ns["what_macro_shown"].set_index("atomic_what").studies.to_dict()
    _check_counts("RQ2.1", {m: ml[ml.what_macro == m].source_csv_row.nunique() for m in macro_order}, shown)
    spec = []
    for macro in macro_order:
        sub = ml[ml.what_macro == macro]
        counts = sub.groupby("what_detail").source_csv_row.nunique().sort_values(ascending=False)
        shown = {"Multi-sensor perception (combination not named in text)": "Multi-sensor perception"}
        children = [(shown.get(d[0].upper() + d[1:], d[0].upper() + d[1:]),
                     sub[sub.what_detail == d].source_csv_row.unique().tolist()) for d in counts.index]
        spec += _gt_group(macro_display.get(macro, macro[0].upper() + macro[1:]),
                          sub.source_csv_row.nunique(), children)
    write_graphical_table(spec, "RQ2_1_table.tex",
        r"RQ\textsubscript{2.1} research targets and supporting primary studies. Bars show the number of "
        r"studies assigned to each target, and the final column lists the corresponding primary studies. "
        r"Studies may be assigned to several targets, so counts across categories are not mutually "
        r"exclusive and subcategory counts may not sum to the parent count.",
        "tab:rq2-1", "Research target", placement="t", font=r"\footnotesize", gap=True,
        widths=(r"0.35\textwidth", r"0.20\textwidth", r"0.4\textwidth"))


def build_rq2_2_table():
    """Family assignment is multi-label (see rq2_2b_families)."""
    long, family_order, _, sub_data = rq2_2b_families()
    fam_n = long.groupby("family_label").source_csv_row.nunique().to_dict()
    spec = []
    for family in family_order:
        sub = long[long.family_label == family]
        if family == "No mitigation mechanism coded":
            spec.append(_gt_row(0, family, sub.source_csv_row.tolist()))
            continue
        exploded = sub.assign(tag=sub.mit_tags).explode("tag")
        exploded = exploded[exploded.tag.map(mitigation_family) == family]
        counts = exploded.groupby("tag").source_csv_row.nunique().sort_values(ascending=False)
        # same per-tag counts as the retired subcategory panels (rq2_2b_families.sub_data)
        plot = dict(next(t for n_, t, _ in sub_data if n_ == family).groupby("tag").source_csv_row.nunique())
        _check_counts(f"RQ2.2 {family}", counts.to_dict(), plot)
        spec += _gt_group(family, fam_n[family],
                          [(t, exploded[exploded.tag == t].source_csv_row.unique().tolist()) for t in counts.index])
    write_split_tables(spec, "RQ2_2",
        r"RQ\textsubscript{2.2} mitigation technique families and techniques, among the studies with a "
        r"mitigation objective. Bars show the number of studies using each family or technique. A study "
        r"can use several families and several techniques within a family, so counts are not mutually "
        r"exclusive and technique counts may exceed the family total.",
        "tab:rq2-2", "Mitigation technique",
        r"Complete mapping of RQ\textsubscript{2.2} mitigation families and techniques to primary studies. "
        r"A study can appear under several families and techniques.", "tab:app-rq2-2",
        compact_total=0.95, compact_font=r"\footnotesize", app_opts=dict(label_col=r"\dimexpr0.5\columnwidth\relax"))


RQ3_1_DISPLAY = {
    "Task/SUT-performance comparison": "Task/SUT performance comparison",
    "Direct or matched sim–real comparison": "Direct or matched sim-to-real comparison",
    "Quantitative/distribution comparison": "Quantitative or distributional comparison",
    "Human/qualitative assessment": "Human or qualitative assessment",
    "Feature/image-quality comparison": "Feature or image-quality comparison",
    "No substantive validation/demonstration only": "No substantive evaluation",
}


def build_rq3_1_table():
    modes = validation_technique(studies)
    long = studies.assign(mode=modes)
    long = long.assign(technique=long["mode"].str.split(" + ", regex=False)).explode("technique")
    counts = long.groupby("technique").source_csv_row.nunique().sort_values(ascending=False)
    spec = [_gt_row(0, RQ3_1_DISPLAY.get(t, t), long[long.technique == t].source_csv_row.unique().tolist())
            for t in counts.index]
    write_graphical_table(spec, "RQ3_1_table.tex",
        r"RQ\textsubscript{3.1} evaluation techniques and supporting primary studies. Bars show the number "
        r"of studies using each technique, and the final column lists the corresponding primary studies. "
        r"Studies may use several techniques, so counts are not mutually exclusive.",
        "tab:rq3-1", "Evaluation technique", flat=True, label_width=0.34, placement="tbp", font=r"\footnotesize",
        widths=(r"0.30\textwidth", r"0.20\textwidth", r"0.45\textwidth"))


def build_rq3_3_table():
    long = ns["metric_sub_long"]
    metric_n = ns["metric_counts"].set_index("metric_category").studies.to_dict()
    cat_order = long.groupby("metric_category").source_csv_row.nunique().sort_values(ascending=False).index.tolist()
    _check_counts("RQ3.3", {c: long[long.metric_category == c].source_csv_row.nunique() for c in cat_order}, metric_n)
    spec = []
    for cat in cat_order:
        sub = long[long.metric_category == cat]
        counts = sub.groupby("metric_subtag").source_csv_row.nunique()
        counts = counts.iloc[sorted(range(len(counts)), key=lambda i: (-counts.iloc[i], counts.index[i]))]
        spec += _gt_group(cat, metric_n[cat],
                          [(t, sub[sub.metric_subtag == t].source_csv_row.unique().tolist()) for t in counts.index])
    # Main text: one table, every category with its studies, plus its eight most frequent specific
    # metrics (ties broken alphabetically), each with its studies; the appendix lists all of them.
    top_n = 8
    norm = lambda t: t.lower().replace("-", " ")
    main = []
    for cat in cat_order:
        sub = long[long.metric_category == cat]
        counts = sub.groupby("metric_subtag").source_csv_row.nunique()
        counts = counts.iloc[sorted(range(len(counts)), key=lambda i: (-counts.iloc[i], counts.index[i]))]
        if len(counts) == 1 and norm(counts.index[0]) == norm(cat):
            main.append(_gt_row(0, cat, sub.source_csv_row.unique().tolist()))  # the category is its own metric
            continue
        main.append(_gt_row(0, cat, None, sub.source_csv_row.nunique()))  # parent: count only, studies are in its rows
        for t in counts.index[:top_n]:
            main.append(_gt_row(1, t, sub[sub.metric_subtag == t].source_csv_row.unique().tolist()))
    write_graphical_table(main, "RQ3_3_table.tex",
        r"RQ\textsubscript{3.3} evaluation metric categories and their eight most frequent specific metrics "
        r"(ties alphabetical), with primary studies.",
        "tab:rq3-3", "Metric category / metric", placement="tbp", font=r"\footnotesize", stretch=0.9,
        bar_width="1.2cm", gap=True, widths=(r"0.35\textwidth", r"0.13\textwidth", r"0.47\textwidth"))
    # Appendix: every specific metric of every category with its primary studies.
    write_graphical_table(spec, "RQ3_3_appendix.tex",
        r"Complete mapping of RQ\textsubscript{3.3} evaluation-metric categories and specific metrics to "
        r"primary studies. A study can appear under several metrics.", "tab:app-rq3-3",
        "Metric category / metric", label_width=0.30, variant="appendix", font=r"\footnotesize", font_split=True,
        label_col=r"\dimexpr0.5\columnwidth\relax")


def build_rq4_1_table():
    sim_long = studies.assign(category=studies.simulator_tags.str.split(" | ", regex=False)).explode("category")
    sim_counts = sim_long.groupby("category").source_csv_row.nunique().sort_values(ascending=False)
    special = {"Dataset/reference data", NO_NAMED_SIMULATOR}
    platforms = [c for c in sim_counts[sim_counts > 1].index if c not in special]
    rows_of = lambda frame, col, v: frame[frame[col] == v].source_csv_row.unique().tolist()
    ds_long = ns["ds_sub_long"]
    ds_order = ns["ds_sub"].sort_values("studies", ascending=False).dataset_subtag.tolist()  # every dataset
    # studies naming at least one simulation platform (the rows below list only the more frequent ones)
    n_platform_studies = int(studies.simulator_tags.str.split(" | ", regex=False).map(
        lambda tags: any(t not in special for t in tags)).sum())
    spec = [_gt_row(0, "Simulation platforms", None, n_platform_studies)]
    spec += [_gt_row(1, p, rows_of(sim_long, "category", p)) for p in platforms]
    ds_studies = studies[studies.simulator_tags.str.contains("Dataset/reference data", regex=False, na=False)]
    _check_counts("RQ4.1 dataset total", {"d": len(ds_studies)}, {"d": sim_counts["Dataset/reference data"]})
    spec.append(_gt_row(0, "Dataset/reference data", None, len(ds_studies)))
    spec += [_gt_row(1, d, rows_of(ds_long, "dataset_subtag", d)) for d in ds_order]
    spec.append(_gt_row(0, NO_NAMED_SIMULATOR, rows_of(sim_long, "category", NO_NAMED_SIMULATOR)))
    write_split_tables(spec, "RQ4_1",
        r"RQ\textsubscript{4.1} simulation platforms and reference datasets reported by at least five studies.",
        "tab:rq4-1", "Simulation platform / dataset",
        r"Complete mapping of RQ\textsubscript{4.1} simulation platforms and reference datasets to primary "
        r"studies. A study can name several platforms and datasets.", "tab:app-rq4-1",
        compact_font=r"\footnotesize",
        compact_opts=dict(placement="t", blank_after_centering=True, comment_below=5),
        app_opts=dict(font=r"\footnotesize", font_split=True))


def build_rq4_2_table():
    hw_long = studies.assign(category=studies.hardware_tags.str.split(" | ", regex=False)).explode("category")
    hw_long = hw_long[hw_long.category != "No hardware / not specified"]
    no_hw = studies[~studies.source_csv_row.isin(hw_long.source_csv_row)]
    hw_counts = hw_long.groupby("category").source_csv_row.nunique().sort_values(ascending=False)
    spec = [_gt_row(0, "No hardware or unspecified", no_hw.source_csv_row.tolist())]
    spec += [_gt_row(0, c, hw_long[hw_long.category == c].source_csv_row.unique().tolist()) for c in hw_counts.index]
    write_graphical_table(spec, "RQ4_2_table.tex",
        r"RQ\textsubscript{4.2} hardware platforms and supporting primary studies.",
        "tab:rq4-2", "Hardware platform", flat=True, placement="tbp", font=r"\footnotesize",
        widths=(r"0.28\textwidth", r"0.20\textwidth", r"0.47\textwidth"))


def build_rq4_3_table():
    sut_long, sut_order, _, sub_data = rq4_3_suts()
    no_sut_label = "No SUT (none, or simulator/testbed fidelity only)"
    full = studies.assign(category=studies.sut_tags.str.split(" | ", regex=False)).explode("category")
    sub_of = {cat: sub_long for cat, sub_long, _, _ in sub_data}
    spec = [_gt_row(0, "No SUT", full[full.category == no_sut_label].source_csv_row.unique().tolist())]
    for cat in sut_order:  # current manuscript order: descending study count
        members = sut_long[sut_long.category == cat]
        if cat not in sub_of:
            spec.append(_gt_row(0, cat, members.source_csv_row.unique().tolist()))
            continue
        sub_long = sub_of[cat]
        if sub_long.source_csv_row.nunique() != members.source_csv_row.nunique():
            raise ValueError(f"RQ4.3 {cat}: technology panel and category counts differ")
        counts = sub_long.groupby("sut_subtag").source_csv_row.nunique().sort_values(ascending=False)
        spec += _gt_group(cat, members.source_csv_row.nunique(),
                          [(t, sub_long[sub_long.sut_subtag == t].source_csv_row.unique().tolist()) for t in counts.index])
    write_split_tables(spec, "RQ4_3",
        r"RQ\textsubscript{4.3} systems under test and specific technologies.",
        "tab:rq4-3", "System under test",
        r"Complete mapping of RQ\textsubscript{4.3} systems under test and specific technologies to primary "
        r"studies. A study can appear under several systems.", "tab:app-rq4-3",
        compact_font=r"\footnotesize", compact_opts=dict(placement="tbp", gap=True),
        app_opts=dict(font=r"\footnotesize"))


def build_rq4_4_table():
    long = ns["generation_long"]
    expected = ns["generation_counts"].set_index("generation_role").studies.to_dict()
    counts = long.groupby("category").source_csv_row.nunique().sort_values(ascending=False)
    _check_counts("RQ4.4", counts.to_dict(), expected)
    spec = [_gt_row(0, c, long[long.category == c].source_csv_row.unique().tolist()) for c in counts.index]
    write_graphical_table(spec, "RQ4_4_table.tex",
        r"RQ\textsubscript{4.4} test-case origin and supporting primary studies.",
        "tab:rq4-4", "Test-case origin", flat=True, placement="tbp", font=r"\footnotesize", blank_after_centering=True,
        widths=(r"0.20\textwidth", r"0.20\textwidth", r"0.55\textwidth"))


def build_rq4_5_cites_table():
    counts = studies.groupby("artefact_status").source_csv_row.nunique().sort_values(ascending=False)
    rows_spec = []
    for i, (cat, n) in enumerate(counts.items()):
        rownums = studies[studies.artefact_status == cat].source_csv_row.tolist()
        cat_label = "Testing artefacts" if i == 0 else ""
        rows_spec.append((cat_label, f"{cat} ({n})", rownums))
    write_citation_table(rows_spec, "RQ4_5_cites_table.tex",
                          r"RQ\textsubscript{4.5} testing artefacts and citations.", "tab:rq4-5-cites",
                          col_widths=(0.12, 0.15, 0.68), as_float=True)


def build_s5_tier_cites_table():
    merged = s5_tiered_cohort()
    rows = [("Execution profile" if i == 0 else "", t, merged[merged.tier == t].source_csv_row.tolist())
            for i, t in enumerate(TIER_ORDER)]
    others = merged[merged.tier == "Other profiles"]
    if len(others):
        rows.append(("", "Outside all six profiles (described individually in the text)",
                     others.source_csv_row.tolist()))
    write_citation_table(rows, "S5_tier_cites_table.tex",
                        r"Mitigation studies by execution profile. Profiles are mutually exclusive.",
                        "tab:s5-tier-cites", col_widths=(0.15, 0.40, 0.40))


CITATION_TABLE_BUILDERS = [
    build_rq2_1_table, build_rq2_2_table, build_rq3_1_table, build_rq3_3_table,
    build_rq4_1_table, build_rq4_2_table, build_rq4_3_table, build_rq4_4_table,
    build_rq4_5_cites_table, build_s5_tier_cites_table,
]


BUILDERS = [
    build_rq1_dashboard,
    build_rq2_3,
    build_rq3_2, build_rq3_2_matched,
    build_rq4_2,
    build_rq5,
    build_s2_target_technique_validation,
    build_s5_execution_taxonomy,
    *CITATION_TABLE_BUILDERS,
]


if __name__ == "__main__":
    for builder in BUILDERS:
        print(f"Building {builder.__name__} ...")
        builder()
    print(f"Wrote {len(BUILDERS)} sub-RQ dashboards to {DASHBOARD_DIR}")
