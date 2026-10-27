"""Cell sources (SETUP, RQ1-RQ4) that derive every count and table used by the paper.

They are executed, in order, by rq_tables_and_figures.py in one shared namespace."""

SETUP = r'''
from pathlib import Path
from itertools import combinations
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from IPython.display import display, HTML

REPO_ROOT=Path(DASHBOARD_SOURCE_ROOT)
ANALYSIS_DIR=REPO_ROOT/"2_results"
if str(ANALYSIS_DIR) not in sys.path:sys.path.insert(0,str(ANALYSIS_DIR))
from deep_analysis_common import *

df,studies=load_mapping()
OUTPUT_DIR=REPO_ROOT/"outputs"/"intermediate"/REPORT_KEY
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
for old_output in OUTPUT_DIR.iterdir():
    if old_output.is_file():old_output.unlink()


assert studies.source_csv_row.is_unique

pd.set_option("display.max_rows",300);pd.set_option("display.max_columns",None);pd.set_option("display.max_colwidth",None)
display(HTML(f"<div class='note'><b>{REPORT_TITLE}</b><br>Studies: {len(studies)} · RQ5 excluded · Output: {OUTPUT_DIR}</div>"))
display(HTML("""<style>
.note{border-left:5px solid #2f6f9f;background:#eef5fa;padding:10px 14px}
table.dataframe{font-size:9px;border-collapse:collapse;width:100%;max-width:100%}
table.dataframe th,table.dataframe td{white-space:normal;overflow-wrap:anywhere;text-align:left;vertical-align:top;padding:4px 6px}
@media print{@page{size:A4 landscape;margin:9mm}.jp-OutputArea-output{overflow:visible!important}
table.dataframe{table-layout:fixed;font-size:7.2pt}table.dataframe thead{display:table-header-group}
table.dataframe tr{break-inside:avoid;page-break-inside:avoid}h1,h2,h3,h4{break-after:avoid-page}
img,svg{max-width:100%!important;height:auto!important;break-inside:avoid}}
</style>"""))

def counts(series,name):
    out=series.fillna("[MISSING]").astype(str).value_counts().rename("studies").rename_axis(name).reset_index()
    out["percentage"]=(100*out.studies/len(studies)).round(1)
    return out

def visible(table):
    return table[table.studies>MIN_COUNT].copy() if FILTER_AGGREGATES else table.copy()

def explain(text):
    display(HTML(f"<p class='note'>{text}</p>"))

def show_table(title,table,filename=None,max_rows=50,force=False):
    if filename:table.to_csv(OUTPUT_DIR/filename,index=False)
    if not SHOW_TABLES and not force:return
    rows=15 if PRINT_MODE else max_rows
    shown=table.head(rows) if rows else table
    display(HTML(f"<h3>{title}</h3>"));display(shown)

OBJECTIVE_ORDER=["Mitigate only","Measure and mitigate","Measure only"]
OBJECTIVE_COLORS=["#E15759","#B07AA1","#4E79A7"]

def long_for(series,label):
    """Build a (source_csv_row, objective_group, label) long frame aligned
    with `studies` from a single- or ' | '-delimited series, so every simple
    counts()-based chart can be objective-split the same way as the richer
    macro/sub-category ones."""
    frame=pd.DataFrame({"source_csv_row":studies.source_csv_row.values,"objective_group":studies.objective_group.values,label:series.fillna("[MISSING]").astype(str).values})
    if frame[label].str.contains(r" \| ",regex=True).any():
        frame=frame.assign(**{label:frame[label].str.split(" | ",regex=False)}).explode(label)
    return frame

def category_order(long_df,label):
    return long_df.groupby(label)["source_csv_row"].nunique().sort_values(ascending=False).index.tolist()

def objective_pivot(long_df,label,order):
    grouped=long_df.groupby([label,"objective_group"])["source_csv_row"].nunique().rename("studies").reset_index()
    pivot=grouped.pivot(index=label,columns="objective_group",values="studies").fillna(0)
    for col in OBJECTIVE_ORDER:
        if col not in pivot.columns:pivot[col]=0
    return pivot[OBJECTIVE_ORDER].reindex(order,fill_value=0)

def _draw_stacked(pivot,title,filename,width=10):
    height=max(3.4,.36*len(pivot)+1.3)
    fig,ax=plt.subplots(figsize=(width,height))
    pivot.plot.barh(stacked=True,ax=ax,width=.68,color=OBJECTIVE_COLORS)
    ax.invert_yaxis();ax.set_xlabel("Studies");ax.set_ylabel("");ax.set_title(title)
    totals=pivot.sum(axis=1);xmax=totals.max() if len(totals) else 0
    for y,total in enumerate(totals):ax.text(total+xmax*.012,y,str(int(total)),va="center",fontsize=8)
    ax.legend(title="Objective",bbox_to_anchor=(1.02,1),loc="upper left");ax.grid(axis="x",alpha=.25)
    fig.tight_layout();fig.savefig(OUTPUT_DIR/f"{filename}.png",dpi=220,bbox_inches="tight");plt.show();plt.close(fig)

def single_bar(long_df,label,title,filename,order=None):
    """Every bar is a stack of Mitigate only / Measure and mitigate / Measure
    only, in that legend order, so any chart's color coding matches RQ1.2's
    objective-split view. `order` fixes the category row order (top to
    bottom); pass a parent chart's order here so a sub-category breakdown
    lines up with its macro bar above it."""
    order=order if order is not None else category_order(long_df,label)
    if not order:return
    _draw_stacked(objective_pivot(long_df,label,order),title,filename)


def bar_report(long_df,label,title,filename,order=None):
    order=order if order is not None else category_order(long_df,label)
    if FILTER_AGGREGATES:
        counts_now=long_df.groupby(label)["source_csv_row"].nunique()
        order=[c for c in order if counts_now.get(c,0)>MIN_COUNT]
    if not order:
        display(HTML(f"<p><b>{title}:</b> no category reaches MIN_COUNT={MIN_COUNT}.</p>"));return
    chunks=[order[i:i+28] for i in range(0,len(order),28)]
    for number,chunk_order in enumerate(chunks,1):
        pivot=objective_pivot(long_df,label,chunk_order)
        _draw_stacked(pivot,title+(f" — part {number}" if len(chunks)>1 else ""),f"{filename}_{number}")


GROUP_SCHEME={"Scene":["scenario","actors"],"Sensing":["perception","localization"],"Control":["actuation","ecu_control"]}
COMPONENT_SCHEME={"Scenario":["scenario"],"Actors":["actors"],"Perception":["perception"],"Localization":["localization"],"Actuation":["actuation"],"ECU/Control":["ecu_control"]}
STATE_ORDER=["Not modelled","Simulated","Pseudo-sim","Pseudo-real","Mixed","Both","Partially real","Real"]
STATE_COLORS={"Not modelled":"#dcdcdc","Simulated":"#0072B2","Pseudo-sim":"#56B4E9","Pseudo-real":"#F0E442","Mixed":"#CC79A7","Both":"#E69F00","Partially real":"#D55E00","Real":"#009E73"}

def group_state(vals):
    """Collapse one or two raw per-component levels into one of eight plain
    states. The same function serves both granularities used throughout
    RQ2.2/RQ3.2: the three functional groups (each built from a pair of raw
    components) and the six raw components taken individually (each a
    'pair' of one). MIXED and BOTH are the two raw codes the original
    coding uses for a live-real presence that stops short of fully Real,
    and are kept distinct rather than merged: MIXED is a live real+simulated
    blend at runtime (e.g. mixed-reality/AR testing), BOTH is a study that
    reports/exercises a separate simulated instance and a separate real
    instance of the same component (not fused live). Partially real is the
    residual case — a group whose two raw components disagree (e.g. one
    REAL, the other SIM) without either being literally coded MIXED/BOTH."""
    s={v for v in vals if v not in ("NO","[MISSING]")}
    if not s:return "Not modelled"
    if s=={"REAL"}:return "Real"
    if s=={"SIM"}:return "Simulated"
    if "MIXED" in s:return "Mixed"
    if "BOTH" in s:return "Both"
    if "REAL" in s and len(s)>1:return "Partially real"
    if "PSEUDO REAL" in s:return "Pseudo-real"
    if "PSEUDO SIM" in s:return "Pseudo-sim"
    return "Partially real"

def scheme_titles(scheme):
    return {name:(f"{name}\n({' + '.join(cols)})" if len(cols)>1 else name) for name,cols in scheme.items()}

def state_frame(prefix,scheme):
    base_cols=["source_csv_row","objective_group"]
    basis_col=f"{prefix}_basis"
    if basis_col in studies.columns:base_cols.append(basis_col)
    out=studies[base_cols].copy()
    for name,cols in scheme.items():
        out[name]=studies[[f"{prefix}_{c}" for c in cols]].apply(lambda r:group_state(r.tolist()),axis=1)
    return out

def state_long(prefix,scheme):
    order=list(scheme.keys())
    long=state_frame(prefix,scheme).melt(id_vars=["source_csv_row"],value_vars=order,var_name="group",value_name="state")
    return long.groupby(["group","state"])["source_csv_row"].nunique().rename("studies").reset_index()

def state_legend(scheme,scheme_label):
    """A structured, table-based definition of the groups and states, meant
    to be displayed before any chart that uses them, so a reader never has
    to infer the taxonomy from the plot itself."""
    group_table=pd.DataFrame([{"Group":name,"Built from raw component(s)":" + ".join(cols)} for name,cols in scheme.items()])
    state_table=pd.DataFrame([
        {"State":"Not modelled","Meaning":"Entirely absent — no active component coded"},
        {"State":"Simulated","Meaning":"Every active component is purely simulated (SIM)"},
        {"State":"Pseudo-sim","Meaning":"Simulated but calibrated/tuned using real data (PSEUDO SIM) — nothing real-data-replayed, blended, or live"},
        {"State":"Pseudo-real","Meaning":"At least one component is a real recording replayed rather than executed live (PSEUDO REAL) — nothing live or blended"},
        {"State":"Mixed","Meaning":"A live real+simulated blend within the group at runtime (MIXED) — e.g. mixed-reality/AR testing where real and simulated elements are fused live"},
        {"State":"Both","Meaning":"The study reports/exercises a separate simulated instance and a separate real instance of the group (BOTH), rather than a live runtime blend"},
        {"State":"Partially real","Meaning":"The group's raw components disagree (e.g. one REAL, the other SIM) without either being literally coded MIXED or BOTH"},
        {"State":"Real","Meaning":"Every active component has real-world content (REAL); execution sources are coded separately"},
    ])
    display(HTML(f"<h4>Reference — {scheme_label}, read before the charts below</h4>"))
    display(group_table)
    display(HTML("<p class='note'>Six plain states, applied independently to each group:</p>"))
    display(state_table)

def state_proportion_chart(prefix,scheme,title,filename):
    """The same group x state counts as state_heatmap, redrawn as a
    100%-stacked bar per group so relative state shares are read directly
    off one shared scale, rather than inferred from heatmap cell-colour
    intensity."""
    order=list(scheme.keys());titles=scheme_titles(scheme)
    sf=state_frame(prefix,scheme)
    long=sf.melt(id_vars=["source_csv_row"],value_vars=order,var_name="group",value_name="state")
    counts_tab=long.groupby(["group","state"])["source_csv_row"].nunique().unstack(fill_value=0).reindex(index=order,columns=STATE_ORDER,fill_value=0)
    props=counts_tab.div(counts_tab.sum(axis=1),axis=0)*100
    fig,ax=plt.subplots(figsize=(10,max(3.2,.55*len(order)+1)))
    labels=[titles[a].replace("\n"," ") for a in props.index]
    left=pd.Series(0.0,index=props.index)
    for state in STATE_ORDER:
        ax.barh(labels,props[state],left=left,color=STATE_COLORS[state],label=state)
        for i,g in enumerate(props.index):
            v=props.loc[g,state]
            if v>=7:ax.text(left[g]+v/2,i,f"{v:.0f}%",ha="center",va="center",fontsize=8.5,color="white" if state in ("Simulated","Partially real","Real") else "#222222")
        left=left+props[state]
    ax.set_xlim(0,100);ax.set_xlabel("% of studies");ax.set_title(title)
    ax.tick_params(axis="y",labelsize=12)
    ax.legend(title="State",bbox_to_anchor=(1.02,1),loc="upper left");ax.grid(axis="x",alpha=.25)
    fig.tight_layout();fig.savefig(OUTPUT_DIR/f"{filename}.png",dpi=220,bbox_inches="tight");plt.show();plt.close(fig)
    return counts_tab.reset_index()

def approach_vs_reference_chart(approach_df,reference_df,scheme,title,filename):
    """Cross-cuts RQ2.2 and RQ3.2: for each group, the share of studies that
    are real-bearing (Real, Partially real, Mixed, or Both) in the proposed
    approach versus in the validation reference, side by side — a direct
    view of which groups are tested against reality more than they are
    built on it, or vice versa, that neither RQ2.2 nor RQ3.2 shows on its
    own."""
    order=list(scheme.keys());titles=scheme_titles(scheme)
    def reality_share(sf):
        long=sf.melt(id_vars=["source_csv_row"],value_vars=order,var_name="group",value_name="state")
        real=long[long.state.isin(["Real","Partially real","Mixed","Both"])]
        return (real.groupby("group").source_csv_row.nunique()/len(studies)*100).reindex(order,fill_value=0)
    comp=pd.DataFrame({"Approach (RQ2.2)":reality_share(approach_df),"Reference (RQ3.2)":reality_share(reference_df)})
    fig,ax=plt.subplots(figsize=(max(8.5,1.4*len(order)),4.2))
    x=range(len(order));width=0.35
    ax.bar([i-width/2 for i in x],comp["Approach (RQ2.2)"],width,label="Approach (RQ2.2)",color="#4E79A7")
    ax.bar([i+width/2 for i in x],comp["Reference (RQ3.2)"],width,label="Reference (RQ3.2)",color="#E15759")
    labels=[titles[a].replace("\n"," ") for a in order]
    rot=15 if (len(order)>3 or max(len(l) for l in labels)>15) else 0
    ax.set_xticks(list(x),labels,rotation=rot,ha="right" if rot else "center")
    ax.set_ylabel("% of studies real-bearing\n(Real/Partially real/Mixed/Both)");ax.set_title(title);ax.legend(bbox_to_anchor=(1.02,1),loc="upper left");ax.grid(axis="y",alpha=.25)
    for i,a in enumerate(order):
        ax.text(i-width/2,comp.loc[a,"Approach (RQ2.2)"]+1,f"{comp.loc[a,'Approach (RQ2.2)']:.0f}%",ha="center",fontsize=8)
        ax.text(i+width/2,comp.loc[a,"Reference (RQ3.2)"]+1,f"{comp.loc[a,'Reference (RQ3.2)']:.0f}%",ha="center",fontsize=8)
    fig.tight_layout();fig.savefig(OUTPUT_DIR/f"{filename}.png",dpi=220,bbox_inches="tight");plt.show();plt.close(fig)
    return comp.reset_index().rename(columns={"index":"group"})

def state_heatmap(long,scheme,title,filename):
    order=list(scheme.keys());titles=scheme_titles(scheme)
    matrix=long.pivot(index="group",columns="state",values="studies").fillna(0)
    matrix=matrix.reindex(index=order,columns=STATE_ORDER,fill_value=0)
    fig,ax=plt.subplots(figsize=(10,max(3.6,.55*len(order)+1)));im=ax.imshow(matrix,cmap="Blues",aspect="auto",vmin=0)
    ax.set_xticks(range(len(matrix.columns)),matrix.columns,rotation=20,ha="right",fontsize=11)
    ax.set_yticks(range(len(matrix.index)),[titles[a] for a in matrix.index],fontsize=12)
    vmax=matrix.values.max() or 1
    for i in range(len(matrix.index)):
      for j in range(len(matrix.columns)):
        v=int(matrix.iloc[i,j])
        color="white" if v>vmax*0.55 else "#222222"
        ax.text(j,i,str(v) if v else "–",ha="center",va="center",fontsize=10,color=color)
    ax.set_xticks([x-.5 for x in range(1,len(matrix.columns))],minor=True)
    ax.set_yticks([y-.5 for y in range(1,len(matrix.index))],minor=True)
    ax.grid(which="minor",color="white",linewidth=1.6);ax.tick_params(which="minor",bottom=False,left=False)
    ax.set_title(title);fig.colorbar(im,ax=ax,label="Studies",shrink=.8);fig.tight_layout()
    fig.savefig(OUTPUT_DIR/f"{filename}.png",dpi=220,bbox_inches="tight");plt.show();plt.close(fig)

def state_combination_chart(prefix,scheme,title,filename,top_n=30):
    """Every distinct state combination actually used across the given
    groups, as a two-part figure sharing one x-axis: an objective-coloured
    bar per combination on top (matching every other chart's colour coding),
    and a state-coloured grid below spelling out the literal state of every
    group for that combination — so every combination is named by its actual
    states, not by an abstract tier, score, or short code. With six raw
    components the number of distinct combinations actually used can run
    past a hundred; rather than shrink one very wide chart down to page width
    (which shrinks its text past legibility) or split it across several
    pages, only the `top_n` most frequent combinations are drawn, on a single
    page-sized chart. No information is lost: the returned table (and the
    CSV/data table shown alongside the chart) always lists every distinct
    combination that occurs, ranked by frequency, not just the ones drawn."""
    order=list(scheme.keys());titles=scheme_titles(scheme);k=len(order)
    sf=state_frame(prefix,scheme)
    sf["combo"]=list(zip(*[sf[c] for c in order]))
    order_counts=sf.groupby("combo")["source_csv_row"].nunique().sort_values(ascending=False)
    combos=order_counts.index.tolist();n=len(combos)
    combo_index={c:i for i,c in enumerate(combos)}
    sf["combo_idx"]=sf.combo.map(combo_index)
    pivot_all=objective_pivot(sf,"combo_idx",list(range(n)))
    state_idx={s:i for i,s in enumerate(STATE_ORDER)}
    cmap=ListedColormap([STATE_COLORS[s] for s in STATE_ORDER])
    m=min(top_n,n)
    shown_combos=combos[:m]
    pivot=pivot_all.loc[range(m)]
    chart_title=title+(f" — top {m} of {n} combinations by frequency" if n>m else "")
    fig,(ax_top,ax_grid)=plt.subplots(2,1,figsize=(max(11,.42*m),5.6+0.22*k),gridspec_kw={"height_ratios":[2.4,0.32*k+0.4]})
    pivot.plot.bar(stacked=True,ax=ax_top,width=.82,color=OBJECTIVE_COLORS,legend=False)
    totals=pivot.sum(axis=1);tmax=totals.max() if len(totals) else 0
    for x,total in enumerate(totals):ax_top.text(x,total+tmax*.02,str(int(total)),ha="center",va="bottom",fontsize=10)
    ax_top.set_ylabel("Studies",fontsize=12);ax_top.set_title(chart_title,fontsize=13);ax_top.grid(axis="y",alpha=.25)
    ax_top.set_xticks([]);ax_top.set_xlabel("")
    ax_top.tick_params(axis="y",labelsize=10)
    ax_top.legend(title="Objective",bbox_to_anchor=(1.01,1),loc="upper left",fontsize=10,title_fontsize=10)
    grid=[[state_idx[c[i]] for c in shown_combos] for i in range(k)]
    ax_grid.imshow(grid,cmap=cmap,vmin=-.5,vmax=len(STATE_ORDER)-.5,aspect="auto")
    ax_grid.set_yticks(range(k),[titles[a].split("\n")[0] for a in order],fontsize=13)
    xlabel=(f"{m} of {n} distinct combinations shown here, ranked by frequency (counts above) — the complete ranked list of all {n} is in the accompanying data table"
            if n>m else f"All {n} distinct combinations, ranked by frequency (counts above)")
    ax_grid.set_xticks([]);ax_grid.set_xlabel(xlabel,fontsize=10.5)
    for spine in ax_grid.spines.values():spine.set_visible(False)
    handles=[Patch(facecolor=STATE_COLORS[s],label=s) for s in STATE_ORDER]
    ax_grid.legend(handles=handles,title="State",loc="upper center",bbox_to_anchor=(0.5,-0.16),ncol=4,fontsize=10,title_fontsize=10)
    fig.subplots_adjust(hspace=0.08)
    fig.savefig(OUTPUT_DIR/f"{filename}.png",dpi=220,bbox_inches="tight");plt.show();plt.close(fig)
    table=pd.DataFrame({"rank":range(1,n+1)})
    for i,name in enumerate(order):
        table[re.sub(r"[^a-z0-9]+","_",name.lower()).strip("_")]=[c[i] for c in combos]
    table["studies"]=order_counts.values
    table["percentage"]=(100*table.studies/len(studies)).round(1)
    return table

def metric_status(value):
    t=norm(value).lower()
    if t in {"no","none","-","[missing]","not specified","no validation"} or "not specified" in t:return "No quantitative metric reported"
    if "qualitative" in t or "just show" in t:return "Qualitative only"
    return "Metric(s) reported"

METRIC_CATEGORY_ROW_OVERRIDES={
  27: ["Vehicle dynamics/actuation comparison"],
  46: ["ADS behaviour/trajectory comparison"],
  48: ["ADS behaviour/trajectory comparison"],
  51: ["ADS behaviour/trajectory comparison"],
  52: ["Sensor/perception comparison"],
  62: ["Sensor/perception comparison"],
  64: ["ADS behaviour/trajectory comparison"],
  71: ["Sensor/perception comparison"],
  73: ["ADS behaviour/trajectory comparison"],
  75: ["Sensor/perception comparison"],
  76: ["ADS behaviour/trajectory comparison"],
  84: ["ADS behaviour/trajectory comparison"],
  97: ["ADS behaviour/trajectory comparison"],
  115: ["Vehicle dynamics/actuation comparison"],
  119: ["ADS behaviour/trajectory comparison"],
  160: ["ADS behaviour/trajectory comparison"],
  166: ["ADS behaviour/trajectory comparison"],
  175: ["ADS behaviour/trajectory comparison"],
  183: ["Sensor/perception comparison"],
  185: ["ADS behaviour/trajectory comparison"],
  # Row 207 compares human driver/participant conduct between VR and the
  # real world, not ADS/vehicle behaviour or trajectory -- a distinct,
  # nameable category rather than the generic "Other specified metric"
  # fallback.
  207: ["Human/participant behavioural comparison"],
  # Rows 212 and 227 report only that results were found "comparable"
  # between two setups, without naming any specific quantitative metric --
  # this matches the definition of "No metric reported" (no evaluation
  # metric is reported) more precisely than the vague "Other specified
  # metric" fallback.
  212: ["No metric reported"],
  227: ["No metric reported"],
}

def metric_categories(value,source_csv_row=None):
    if source_csv_row in METRIC_CATEGORY_ROW_OVERRIDES:
        return " | ".join(METRIC_CATEGORY_ROW_OVERRIDES[source_csv_row])
    t=norm(value).lower()
    if (t in {"no","none","-","[missing]","not specified","no validation","n/a","na"} or "not specified" in t
            or re.search(r"not mentioned|no val(?:idation)?\b",t)):
        return "No metric reported"
    labels=[]
    rules=[
      (r"psnr|ssim|lpips|\bfid\b|\bfvd\b|\bkid\b|image quality|\bpixel\b|photoreal|image realism|visual realism|\bvgg\b|feature.?space|\bglcm\b|wavelet|haralick|\blbp\b|brisque|\bniqe\b|\bpiqe\b|\bblur\b|\bnoise\b|\bcontrast\b|\bclipping\b|perceptual|reflectance|transmittance|permittivity|\bbrdf\b|precision|recall|\bf-?1\b|\bfppi\b|average precision|\bmap\b(?!e)|\bmap\d\d|\biou\b|\bmiou\b|detect|segment|classif|panoptic|\bdice\b|point.?cloud|pointcloud|chamfer|\bospa\b|\bmi\b|\bm?ncc\b|cosine similarity|radar cross section|\brcs\b|signal.to.noise|\bsnr\b|beam pattern|azimuth|elevation angle|range doppler|lidar intensity|lidar.{0,15}(hit rate|dropout|return)|scan.{0,10}(layer|beam)|point count|points per second|sensor model qualification|received power|scanning field of view|point.cloud size|fme\b|domain adaptation performance|perception accuracy|perception performance|perception.based|image.level|camera model fidelity|vision system validation|sensor measurement error|minkowski|centroid","Sensor/perception comparison"),
      (r"trajector|\bpath\b|headway|\blane\b|\bspeed\b|\bvelocity\b|driving behavio|sim behavior|real behavior|behavioral cloning|difference of behaviour|endpoint offset|car.?follow|\bade\b|\bfde\b|sim agents|scenario realism|planning|route control|route plan|disengagement|\bdrift\b|\bhota\b|\bmota\b|\bmotp\b|\bidf1\b|\bdeta\b|\bassa\b|multi.?object track|object.?tracking|inter-vehicle|leading distance|yielding|nearest distance|\bdtw\b|dynamic time warping|hausdorff|fr[ée]chet distance|discrete fr[ée]chet|scenario reproduction|lane fitting|position deviation|translation error|interaction model|traffic management","ADS behaviour/trajectory comparison"),
      (r"acceler|steer|brak|\byaw\b|\bjerk\b|longitudinal|lateral|vehicle dynamics|\bdynamics\b|terrain gap|nu.?gap|energy consumption|fuel.?use|internal dynamics","Vehicle dynamics/actuation comparison"),
      (r"collision|crash|time.?to.?collision|\bttc\b|\bpet\b|gttc|safety|violation|pass.?fail|failure rate|avoidance rate|drivable area|route completion|comfort\b|out.of.bound|out.of.track|accident rate|accident cases","Safety/event outcome"),
      (r"runtime|run time|latency|computation|processing time|frames per second|\bfps\b|\bmemory\b|real.?time|channel load|bytesize|packet|bandwidth|\bgpu\b|model size|\bflops\b|speedup|inference time|topic period|training time|round trip time|\brtt\b","Runtime/computational"),
      (r"qualitative|visual inspect|\bhuman\b|user stud|\bexpert\b|likert|whiteness","Qualitative assessment"),
    ]
    for pattern,label in rules:
        if re.search(pattern,t):labels.append(label)
    return " | ".join(dict.fromkeys(labels)) if labels else "Other specified metric"

METRIC_SUBTAG_ROW_OVERRIDES={
  ("Sensor/perception comparison",75): "Sensor/signal-measurement difference (radar)",
  ("ADS behaviour/trajectory comparison",76): "Generated-trajectory diversity & realism score (fidelity/diversity/reactivity/feasibility)",
  ("Vehicle dynamics/actuation comparison",27): "Signal-level RMSE/NRMSE (sim vs. real vehicle-dynamics signals)",
  ("ADS behaviour/trajectory comparison",46): "Crash-scenario reconstruction accuracy/coverage",
  ("ADS behaviour/trajectory comparison",48): "Copula-based driver-interaction model fit (reconstruction/simulation accuracy, MSE)",
  ("ADS behaviour/trajectory comparison",51): "DNN driving-decision scenario-matching error (steering-angle similarity)",
  ("Sensor/perception comparison",52): "DNN scenario-matching error (image similarity)",
  ("ADS behaviour/trajectory comparison",64): "Driver-behaviour distributional comparison",
  ("Sensor/perception comparison",71): "3D-scene reconstruction depth/consistency error (RMSE/MAE/iMAR/iRMSE)",
  ("ADS behaviour/trajectory comparison",73): "Composite multivariate scenario-scoring metric (collection of sub-metrics)",
  ("ADS behaviour/trajectory comparison",84): "LLM-as-judge scenario-realism robustness score",
  ("ADS behaviour/trajectory comparison",97): "VIL-platform signal-curve reliability metrics (Geers/Zilliacus)",
  ("Vehicle dynamics/actuation comparison",115): "Vehicle-dynamics parameter-estimation RMSE",
  ("ADS behaviour/trajectory comparison",119): "QD scenario-generation coverage/sample-efficiency",
  ("ADS behaviour/trajectory comparison",160): "LLM-as-judge scenario-realism robustness/success rate",
  ("ADS behaviour/trajectory comparison",166): "Scenario-description quality & coverage",
  ("ADS behaviour/trajectory comparison",175): "Full-stack co-simulation toolchain end-to-end validation outcome",
  ("Sensor/perception comparison",183): "Procedurally-reconstructed environment coverage/fidelity",
  ("ADS behaviour/trajectory comparison",185): "Per-model simulation-reliability quantitative score",
  # Rows whose metric text matched a category but no sub-metric rule, which used to fall through to a
  # "No specific <category> metric named" label. The extraction text (and, for 203 and 228, the paper)
  # names the quantity:
  ("Safety/event outcome",203): "Scenario criticality score (composite risk indicators)",  # scenario scoring rule over multi-dimensional risk metrics (Grey Relational Analysis)
  ("Safety/event outcome",228): "Safety distance/near-miss",  # minimum clearance to vehicles/obstacles kept above a speed-dependent safety margin
  ("ADS behaviour/trajectory comparison",205): "Moving-object trajectory reproduction accuracy (vs. real ground truth)",
  ("ADS behaviour/trajectory comparison",211): "Inter-vehicle gap and traffic-flow distribution comparison",
  ("ADS behaviour/trajectory comparison",226): "Trajectory/path adaptation performance (simulation vs. field test)",
  ("Sensor/perception comparison",219): "Sensor-model fidelity validation (acceptance/qualification criteria)",
  ("Vehicle dynamics/actuation comparison",222): "Longitudinal-control tracking performance (speed/following distance)",
}

STAT_FORMULA_SUBTAGS=[
    (r"wasserstein|kolmo\w*|smirnov|\bks\b|jensen.?shannon|\bjsd\b|kl divergence|earth mover|\bemd\b","Distribution-distance test"),
    (r"n?rmse\b|\brrmse\b|\brsme\b|\bmae\b|\bmse\b","RMSE/MAE-family error"),
    (r"correlation","Correlation coefficient"),
    (r"variance|standard deviation","Variance/standard deviation"),
    (r"statistical significance|confidence interval|p-?value|t-?test","Significance/confidence testing"),
    (r"\bmape\b|\br.square\b|\bsse\b|goodness.of.fit","Regression fit/error statistic"),
    (r"jaccard|hamming distance","Set-overlap distance (Jaccard/Hamming)"),
]

METRIC_SUBTAG_RULES={
  "Sensor/perception comparison":[
    (r"\bfid\b|frechet inception|fr[ée]chet inception|\bfvd\b|\bkid\b","Distributional image/video realism (FID/FVD/KID)"),
    (r"\bpsnr\b","Signal fidelity (PSNR)"),
    (r"\bssim\b|structural similarity|t-ssim","Structural similarity (SSIM)"),
    (r"lpips|perceptual similarity metric","Learned perceptual similarity (LPIPS)"),
    (r"\bvgg\b|feature.?space|style loss|\bniqe\b|\bpiqe\b|\bbrisque\b","Learned/perceptual feature comparison"),
    (r"\bblur\b|\bnoise\b(?!\s*ratio)|\bcontrast\b|\bclipping\b","Signal degradation/artifact score (blur/noise/contrast)"),
    (r"centroid|cosine similarity.{0,15}featur|featur.{0,15}cosine similarity|eucl.{0,20}featur|featur.{0,20}eucl","Feature-embedding distance"),
    (r"\bglcm\b|wavelet|haralick|\blbp\b|multi.criteria combination","Texture/hand-crafted feature metric"),
    (r"\bpixel\b.{0,10}(error|integrity|accuracy)|pixel.level","Pixel-level error"),
    (r"average precision|\bmap\b(?!e)(?!ping)|\bmap\d\d|\bap50\b|\bap\b(?!i)","Average precision (mAP/AP)"),
    (r"\biou\b|intersection over union|panoptic|\bmiou\b|mean iou","IoU/segmentation overlap"),
    (r"precision|recall|\bf-?1\b|\bfppi\b|\btp\b|\bfn\b|\bdice\b","Precision/recall/F1"),
    (r"\bclassif","Classification accuracy"),
    (r"\bdetect","Detection accuracy/rate"),
    (r"3d pose error|detection reliability","Human/object 3D pose-estimation accuracy"),
    (r"chamfer|point.to.point cloud|pointcloud.to.pointcloiud|point number error|\bminkowski\b|histogram.{0,10}pointcloud","Point-cloud distance (Chamfer/point-to-point/Minkowski)"),
    (r"wasserstein|\bemd\b|earth mover|gaussian wasserstein|feature.wise wasserstein","Point-cloud/feature distributional distance (EMD/Wasserstein)"),
    (r"radar cross section|\brcs\b(?!\w)","Radar cross-section (RCS) accuracy"),
    (r"signal.to.noise|\bsnr\b|received power|beam pattern|range doppler|signal power|sensor measurement error|two.?way beam","Sensor/signal-measurement difference"),
    (r"range \(r\)|detection outcome.{0,15}(angle|distance)|detections.{0,10}distance|scanning field of view|point.cloud size|point count|points per second|relative radial velocity|azimuth estimation|elevation estimation","Sensor field-of-view/detection-range (hardware coverage, not driving behaviour)"),
    (r"relative intensity decrease|valid.{0,10}returns|spurious returns|lidar.{0,15}(hit rate|dropout|return)|lidar intensity","LiDAR return-rate/intensity accuracy"),
    (r"occupied cells ratio|\bocr\b|occupancy grid","Occupancy-grid comparison"),
    (r"reflectance|transmittance|permittivity|\bbrdf\b","Material/reflectance-property fidelity"),
    (r"domain adaptation performance|perception accuracy|perception performance(?! measurement)|perception.based metrics|perceptual sim.to.real gap|perceptual gap","Domain-adaptation/perception-transfer performance"),
    (r"image.level metrics|distribution level and image level","Image/perception-level distributional quality metrics"),
    (r"image characteristic distribution","GAN-generated image-realism distributional comparison"),
    (r"neuron coverage|surprise adequacy|fault.revealing ability","ML-testing coverage metrics (neuron coverage/surprise adequacy/fault-revealing ability)"),
    (r"sensor model qualification|vision system validation|camera model fidelity","Sensor-model fidelity validation (acceptance/qualification criteria)"),
  ]+STAT_FORMULA_SUBTAGS,
  "ADS behaviour/trajectory comparison":[
    (r"\bade\b|\bfde\b|displacement error|trajector.{0,15}(error|rmse)|absolute trajectory error|\bate\b|\brpe\b|translation error|c-ate|m-ate|end.pose difference|x and y error|position error.{0,15}trajector","Trajectory error (ADE/FDE/RMSE/ATE)"),
    (r"\bspeed\b|\bvelocity\b","Speed/velocity comparison"),
    (r"\blane\b|headway|car.?follow|following gap|gap match","Lane-keeping/car-following behaviour"),
    (r"distance.{0,45}trajector|trajector.{0,45}distance|spatio.?temporal.{0,15}(distance|euclid)|euclid.{0,20}trajector|position deviation|scenario reproduction fidelity|lane fitting|trajectory extraction|l2 norm.{0,20}trajector|distance of driving profiles|driving profile.{0,20}(trajector|velocity)|trajector.{0,10}differen","Trajectory/path point-wise distance (sim vs. real)"),
    (r"vel\.?,\s*trajectory,\s*yaw","Trajectory/kinematic-state value list (raw, per timestep)"),
    (r"trajectory similarity|trajectory distance measures|memory access time and trajectory","Trajectory similarity/distance score"),
    (r"just measured difference of behaviour|behavioral cloning performance|behavior vs.{0,5}real behavior|sim behavior.{0,15}real behavior|difference of behaviour|real vehicle.{0,20}driving behavior|driving behavior.{0,30}(against|compared|reference)|no explicit trajectory-error metric reported","Real-vs-simulated driving-behaviour comparison"),
    (r"trajector.{0,15}comparison(?! across time)","Point-wise trajectory comparison (raw positions)"),
    (r"trajectory comparison across time","Trajectory comparison over simulation time"),
    (r"\bdtw\b|dynamic time warping|hausdorff|fr[ée]chet distance|discrete fr[ée]chet|\block?ss\b|longest common subsequence","Trajectory-shape distance (DTW/Hausdorff/Fréchet/LCSS)"),
    (r"inter-vehicle distance|leading distance|yielding distance|relative inter-vehicle|relative target distance|distance to object","Inter-vehicle/relative-spacing distance"),
    (r"policy based measures|traffic measures","Policy/traffic-behaviour descriptive measures"),
    (r"average endpoint offset|endpoint offset","Endpoint/waypoint offset error"),
    (r"\bhota\b|\bmota\b|\bmotp\b|\bidf1\b|\bdeta\b|\bassa\b|multi.?object track|object.?tracking(?! performance)","Multi-object tracking accuracy"),
    (r"detection, association, trajector|traffic flow reconstruction","Multi-vehicle tracking + traffic-flow reconstruction accuracy"),
    (r"planning|route control|trajector planning","Planning/route-control outcome"),
    (r"path.?tracking|steering response.{0,15}tracking|control error|command.{0,10}(vs|comparison)|nmpc cost","Path-tracking control performance"),
    (r"lane detection.{0,5}\(lka\)|crash avoidance.{0,5}\(acc\)|cutin.?cutout|\baeb\b","ADAS-function pass/fail (LKA/ACC/AEB)"),
    (r"collision-free success rate|adaptive lane-changing ratio|total loss|angle loss|throttle loss","Policy-training success rate/loss"),
    (r"physical validity","Generated-scenario kinematic-validity check (acceleration/jerk within physical limits)"),
    (r"observed.?data fitness|reconstruction accuracy|optimization fitness|convergence(?! loss)","Trajectory-reconstruction optimization fit/convergence"),
    (r"variance of trajector|manual disengagement|disengagement rate|trajectory validity rate","Trajectory variance/disengagement/validity rate"),
    (r"matching score|diversity.{0,10}\(div\)|multimodality|\bfidelity\b.{0,20}\bdiversity\b|\breactivity\b|\bfeasib","Generated-trajectory diversity & realism score (FID/MMD)"),
    (r"interaction model|traffic management performance|interaction rate score|social compliance score","Interaction/traffic-behaviour modelling accuracy"),
    (r"sub.meter localization accuracy|localization accuracy|localization rmse|\bdrift\b(?! loss)","Localization/pose-tracking accuracy"),
    (r"waymo.{0,10}(realism|sim agents)|scenario realism","Waymo Sim Agents realism benchmark score"),
    (r"schema validation|validation.feedback","Schema/format validation success rate"),
  ]+STAT_FORMULA_SUBTAGS,
  "Vehicle dynamics/actuation comparison":[
    (r"acceler","Acceleration"),
    (r"steer","Steering"),
    (r"brak","Braking/deceleration"),
    (r"\byaw\b","Yaw"),
    (r"\bjerk\b","Jerk"),
    (r"velocity error|longitudinal position error|cross-track error|\bxte\b|heading error|displacement\b(?! error)","Vehicle-state tracking error"),
    (r"energy consumption|fuel.?use|penetration rate|speed variability|flow error","Energy/fuel-consumption & traffic-flow comparison"),
    (r"terrain gap|nu.?gap|internal dynamics","Vehicle-dynamics model-mismatch metric (ν-gap/terrain-gap/internal-dynamics-gap)"),
    (r"vehicle dynamics\b(?! parameter)","Bare 'vehicle dynamics' comparison (combinations of ACC/AEB/LKA)"),
    (r"mean value and standard deviation","Velocity/spacing mean & standard deviation"),
  ]+STAT_FORMULA_SUBTAGS,
  "Safety/event outcome":[
    (r"collision|crash","Collision/crash rate"),
    (r"time.?to.?collision|\bttc\b|\bpet\b|gttc","Time-to-collision/PET"),
    (r"safety distance|near.?miss|nearest distance","Safety distance/near-miss"),
    (r"pass.?fail|disengagement|failure rate|violation|avoidance rate|route completion|drivable area|comfort\b|out.of.bound|out.of.track","Pass/fail or disengagement rate"),
    (r"accident rate|accident cases","Traffic-safety distributional statistics"),
  ],
  "Runtime/computational":[
    (r"latency|runtime|run time|processing time|real.?time|topic period","Latency/runtime"),
    (r"\bfps\b|frame rate|frames per second","Frame rate (FPS)"),
    (r"memory|\bgpu\b|computation|model size|\bflops\b|speedup","Memory/compute utilization"),
    (r"inference time","Inference time"),
    (r"channel load|bytesize|packet|bandwidth","Network/communication load"),
    (r"training time","Model training time"),
    (r"round trip time|\brtt\b","Round-trip time (RTT)"),
    (r"pareto efficiency","Accuracy-vs-compute tradeoff (Pareto efficiency)"),
    (r"odd coverage|ads failures|semantic validator","ODD-coverage & ADS-failure testing overhead"),
  ],
  "Qualitative assessment":[
    (r"visual inspect|qualitative","Visual/qualitative inspection"),
    (r"\bhuman\b|user stud|intervention","Human/user study"),
    (r"\bexpert\b","Expert judgement"),
    (r"likert","Likert-scale rating"),
    (r"whiteness","Whiteness/subjective realism score"),
  ],
}

def metric_subtags(category,value,source_csv_row=None):
    if (category,source_csv_row) in METRIC_SUBTAG_ROW_OVERRIDES:
        return METRIC_SUBTAG_ROW_OVERRIDES[(category,source_csv_row)]
    if category not in METRIC_SUBTAG_RULES:return category
    t=norm(value).lower();labels=[]
    for pattern,label in METRIC_SUBTAG_RULES[category]:
        if re.search(pattern,t):labels.append(label)
    return " | ".join(dict.fromkeys(labels)) if labels else f"No specific {category.lower()} metric named"

def validation_status(value):
    t=norm(value).lower()
    return "No validation reported" if t in {"no","none","-","[missing]","not evaluated"} or re.search(r"no(?: proper)? validation|just show",t) else "Validation/measurement reported"

GENERATION_ROLE_OVERRIDES={
    4: ["Manual/predefined","Real-world executed"],
    5: ["Real-world reconstruction"],
    7: ["Real-world reconstruction"],
    8: ["Real-world executed","Real-world reconstruction"],
    12: ["Simulation-generated","Recorded/replayed"],
    16: ["Recorded/replayed","Real-world executed"],
    17: ["Real-world executed"],
    20: ["Recorded/replayed"],
    26: ["Real-world executed","Real-world reconstruction"],
    30: ["Recorded/replayed","Simulation-generated","Real-world executed"],
    31: ["Hybrid/mixed"],
    32: ["Hybrid/mixed"],
    33: ["Real-world reconstruction","Simulation-generated"],
    35: ["Simulation-generated","Recorded/replayed"],
    37: ["Manual/predefined"],
    38: ["Simulation-generated","Real-world reconstruction"],
    40: ["Real-world executed","Simulation-generated"],
    41: ["Manual/predefined","Real-world executed","Real-world reconstruction","Recorded/replayed"],
    42: ["Manual/predefined","Real-world executed","Hybrid/mixed"],
    43: ["Manual/predefined","Simulation-generated","Hybrid/mixed","Real-world executed"],
    45: ["Simulation-generated","Real-world reconstruction","Real-world executed"],
    46: ["Real-world reconstruction"],
    47: ["Manual/predefined","Real-world executed","Hybrid/mixed"],
    48: ["Simulation-generated"],
    49: ["Manual/predefined","Simulation-generated","Real-world executed"],
    50: ["Manual/predefined","Simulation-generated","Real-world executed"],
    51: ["Real-world reconstruction","Recorded/replayed"],
    52: ["Real-world reconstruction","Real-world executed"],
    53: ["Manual/predefined","Simulation-generated"],
    54: ["Simulation-generated","Recorded/replayed"],
    55: ["Manual/predefined","Real-world reconstruction","Real-world executed","Simulation-generated"],
    56: ["Manual/predefined","Simulation-generated","Real-world executed"],
    59: ["Simulation-generated","Real-world executed"],
    60: ["Simulation-generated","Real-world executed"],
    64: ["Simulation-generated","Real-world executed"],
    65: ["Simulation-generated","Recorded/replayed"],
    67: ["Simulation-generated","Recorded/replayed"],
    68: ["Real-world reconstruction"],
    73: ["Real-world reconstruction","Recorded/replayed"],
    78: ["Simulation-generated","Real-world executed"],
    80: ["Simulation-generated","Real-world reconstruction","Recorded/replayed","Real-world executed"],
    82: ["Simulation-generated","Recorded/replayed"],
    85: ["Real-world reconstruction"],
    88: ["Real-world reconstruction","Recorded/replayed","Real-world executed"],
    92: ["Manual/predefined","Simulation-generated","Real-world executed"],
    111: ["Simulation-generated","Real-world executed"],
    116: ["Real-world reconstruction","Simulation-generated"],
    117: ["Recorded/replayed","Simulation-generated"],
    121: ["Hybrid/mixed"],
    122: ["Recorded/replayed","Real-world reconstruction"],
    124: ["Simulation-generated"],
    127: ["Simulation-generated"],
    128: ["Simulation-generated"],
    132: ["Simulation-generated"],
    133: ["Hybrid/mixed"],
    137: ["Simulation-generated"],
    138: ["Simulation-generated"],
    139: ["Recorded/replayed"],
    140: ["Recorded/replayed"],
    142: ["Simulation-generated"],
    146: ["Real-world reconstruction","Simulation-generated"],
    148: ["Real-world reconstruction","Simulation-generated"],
    150: ["Manual/predefined"],
    151: ["Simulation-generated"],
    126: ["Recorded/replayed"],
    131: ["Manual/predefined","Simulation-generated","Real-world executed"],
    144: ["Real-world reconstruction","Simulation-generated"],
    145: ["Manual/predefined","Simulation-generated"],
    152: ["Recorded/replayed","Simulation-generated","Real-world reconstruction","Real-world executed"],
    153: ["Real-world executed","Simulation-generated"],
    154: ["Simulation-generated"],
    157: ["Manual/predefined","Real-world executed","Simulation-generated"],
    158: ["Simulation-generated","Recorded/replayed"],
    160: ["Simulation-generated"],
    162: ["Manual/predefined","Real-world reconstruction"],
    165: ["Manual/predefined"],
    167: ["Real-world reconstruction"],
    168: ["Simulation-generated"],
    169: ["Recorded/replayed","Simulation-generated"],
    174: ["Real-world reconstruction","Simulation-generated"],
    177: ["Real-world reconstruction"],
    179: ["Real-world reconstruction"],
    180: ["Real-world executed","Real-world reconstruction"],
    181: ["Manual/predefined","Real-world executed","Real-world reconstruction"],
    182: ["Simulation-generated","Recorded/replayed"],
    184: ["Recorded/replayed"],
    185: ["Manual/predefined","Real-world executed","Real-world reconstruction"],
    186: ["Simulation-generated","Real-world executed"],
    188: ["Real-world executed"],
    189: ["Simulation-generated","Real-world executed"],
    194: ["Real-world executed","Simulation-generated"],
    197: ["Simulation-generated","Recorded/replayed"],
}

def generation_role(value,source_csv_row=None):
    if source_csv_row in GENERATION_ROLE_OVERRIDES:
        return " | ".join(GENERATION_ROLE_OVERRIDES[source_csv_row])
    t=norm(value).lower();labels=[]
    for p,l in [(r"manual|hand.?craft|predefin|expert|custom","Manual/predefined"),(r"generat|synthetic|simulation|simulat|render|augment|diffusion|\bgan\b|process(?:ed|ing)?|creat|transform|synthesi[sz]e","Simulation-generated"),(r"reconstruct|recreat|digital twin|map.*real|real.*map|replicat|extract.*(?:dataset|scenario)","Real-world reconstruction"),(r"replay|recorded|log|rosbag|ros bag|\bdataset\b|\breuse\b",  "Recorded/replayed"),(r"real world|real-world|proving|test field|on-road|test track|highway|real (?:vehicle|road|car)","Real-world executed"),(r"mix|hybrid|synchron","Hybrid/mixed")]:
        if re.search(p,t):labels.append(l)
    return " | ".join(dict.fromkeys(labels)) if labels else "Unclear/not reported"

def artefact_status(value):
    t=norm(value).lower()
    # Bare "-"/"--"/"---" is a placeholder for "nothing reported", the same
    # as "no"/"none" -- without this, 28 studies that used a dash instead
    # of a word fell through to "Other artefact" instead of here.
    if t in {"no","none","na","n/a","[missing]","todo",""} or re.fullmatch(r"-{1,}",t):return "None/unspecified"
    # "NA (...)" is a negative with an explanation (e.g. "NA (training data stated as confidential ...)"):
    # nothing public, whatever nouns the explanation happens to contain.
    if re.match(r"n/?a\b",t):return "None/unspecified"
    code=bool(re.search(r"code|github|repository|software",t));data=bool(re.search(r"data|dataset",t))
    if code and data:return "Code and dataset"
    if code:return "Code"
    if data:return "Dataset"
    return "Other artefact"
'''

RQ1 = r'''
display(HTML("<h3>RQ1.1 — What is the primary objective of the study?</h3>"));explain("Counts the directly coded objective after consolidating its wording into Measure only, Mitigate only, or Measure and mitigate, split by that same objective (so each bar is naturally one solid segment), consistent with every other chart in this report. The thresholded report shows only objectives occurring in more than MIN_COUNT studies.")
objective=counts(studies.objective_group,"objective");objective_long=long_for(studies.objective_group,"objective")
bar_report(objective_long,"objective","RQ1.1 Primary objective","rq1_1_objective");show_table("RQ1.1 Reported counts",visible(objective),"rq1_1_objective.csv")
display(HTML("<h3>RQ1.2 — What motivates the research?</h3>"));explain("The first view reports the atomic WHAT targets independently of whether papers measure or mitigate them, with every bar split by objective (Mitigate only / Measure and mitigate / Measure only), matching the detailed breakdown further below. A multi-target paper contributes once to every distinct target it addresses. Every family of closely related targets with more than 5 studies is aggregated into a single macro bar — Perception, Scenario &amp; actor realism (including road/map fidelity), and Ego-system behaviour &amp; planning — with each macro's own sub-target breakdown charted and tabulated immediately below it, in the same order as the macro bar above. Remaining targets, including ego localization/pose estimation, are specific enough to stand on their own and are shown individually.")
motivation_long=studies.assign(atomic_what=studies.what_tags.str.split(" | ",regex=False)).explode("atomic_what")
what_counts=motivation_long.groupby("atomic_what")["source_csv_row"].nunique().rename("studies").reset_index().sort_values("studies",ascending=False);what_counts["percentage"]=(100*what_counts.studies/len(studies)).round(1)

PERCEPTION_WHAT_TAGS={"camera perception","LiDAR perception","radar perception","multi-sensor perception"}
SCENARIO_WHAT_TAGS={"traffic/NPC realism","pedestrian/VRU realism","crash-scenario validity","test-scenario validity","road/environment realism"}
EGO_SYSTEM_WHAT_TAGS={"ego-vehicle dynamics/actuation","SUT behaviour realism","ADS-policy transfer"}
WHAT_MACRO_LABELS=["Perception","Scenario and actor realism","Ego-system behaviour and planning"]

def what_macro(tag):
    if tag in PERCEPTION_WHAT_TAGS:return "Perception"
    if tag in SCENARIO_WHAT_TAGS:return "Scenario and actor realism"
    if tag in EGO_SYSTEM_WHAT_TAGS:return "Ego-system behaviour and planning"
    return tag

motivation_long["what_macro"]=motivation_long.atomic_what.map(what_macro)
what_macro_counts=motivation_long.groupby("what_macro")["source_csv_row"].nunique().rename("studies").rename_axis("atomic_what").reset_index().sort_values("studies",ascending=False)
what_macro_counts["percentage"]=(100*what_macro_counts.studies/len(studies)).round(1)
what_macro_shown=visible(what_macro_counts)
if what_macro_shown.empty:
    display(HTML(f"<p><b>RQ1.2 Research targets:</b> no category reaches MIN_COUNT={MIN_COUNT}.</p>"))
else:
    single_bar(motivation_long,"what_macro","RQ1.2 Research targets (macro-aggregated)","rq1_2_what_macro",order=what_macro_shown.atomic_what.tolist())
show_table("RQ1.2 Research target counts (macro)",what_macro_shown,"rq1_2_what_macro_counts.csv",max_rows=None)

def sensor_combo(row):
    """Name the specific sensing modalities behind a 'multi-sensor
    perception' tag, recovered from the paper's own text (title, detailed
    WHAT/HOW, hardware, and SUT fields) where possible."""
    text=" ".join(str(row[c]) for c in ["paper_title","detailed_what_how","hardware","sut"]).lower()
    parts=[]
    if re.search(r"camera|image|rgb|monocular|multi.?camera|mobileye",text):parts.append("Camera")
    if re.search(r"lidar",text):parts.append("LiDAR")
    if re.search(r"radar",text):parts.append("Radar")
    if re.search(r"event.?camera|event.?based",text):parts.append("Event camera")
    if len(parts)>=2:return " + ".join(parts)+" perception"
    if len(parts)==1:return {"Camera":"camera perception","LiDAR":"LiDAR perception","Radar":"radar perception","Event camera":"camera perception"}[parts[0]]
    return "Multi-sensor perception (combination not named in text)"

motivation_long["what_detail"]=motivation_long.apply(lambda r:sensor_combo(r) if r.atomic_what=="multi-sensor perception" else r.atomic_what,axis=1)

what_macro_breakdowns={
    label:(motivation_long[motivation_long.what_macro==label].groupby("what_detail")["source_csv_row"].nunique()
           .rename("studies").rename_axis("atomic_what").reset_index().sort_values("studies",ascending=False))
    for label in WHAT_MACRO_LABELS
}
for macro_label in [m for m in what_macro_shown.atomic_what if m in WHAT_MACRO_LABELS]:
    prefix="rq1_2_"+re.sub(r"[^a-z0-9]+","_",macro_label.lower()).strip("_")+"_sub"
    sub=what_macro_breakdowns[macro_label]
    if sub.empty:continue
    sub["percentage"]=(100*sub.studies/len(studies)).round(1)
    if macro_label=="Perception":
        explain("\"Multi-sensor perception\" (naming any combination of sensing modalities without saying which) is shown here already split by the specific sensor combination named in the paper's own text, rather than as a separate follow-up chart.")
    single_bar(motivation_long[motivation_long.what_macro==macro_label],"what_detail",f"RQ1.2 {macro_label} subtypes",prefix,order=sub.atomic_what.tolist())
    show_table(f"RQ1.2 {macro_label} subtype counts",sub,f"{prefix}_counts.csv",max_rows=None)

for standalone_label,standalone_count in what_macro_shown.set_index("atomic_what").studies.items():
    if standalone_label in WHAT_MACRO_LABELS or standalone_count<=1:continue
    prefix="rq1_2_"+re.sub(r"[^a-z0-9]+","_",standalone_label.lower()).strip("_")+"_papers"
    explain(f"\"{standalone_label}\" does not belong to a larger macro family and is too small ({standalone_count} studies) to support a meaningful sub-category breakdown, so no separate chart is drawn for it — its papers are listed directly below instead.")
    paper_table=(motivation_long[motivation_long.atomic_what==standalone_label][["source_csv_row","paper_title"]]
                 .drop_duplicates("source_csv_row").sort_values("source_csv_row"))
    show_table(f"RQ1.2 {standalone_label} — papers",paper_table,f"{prefix}.csv",max_rows=None)

show_table("RQ1.2 Atomic WHAT counts (full detail)",visible(what_counts),"rq1_2_what_counts.csv",max_rows=None)
display_what_macro=set(what_macro_shown.atomic_what)
if not FILTER_AGGREGATES:
    motivation_by_study=(motivation_long.groupby(["source_csv_row","paper_title","simplified_what"],as_index=False)
        .agg(atomic_what=("atomic_what",lambda x:" | ".join(x)),objective=("objective_group","first")))
    motivation_by_study.to_csv(OUTPUT_DIR/"rq1_2_what_objective_by_study.csv",index=False)
display(HTML("<h3>RQ1.3 — What is the direction of the simulation–reality relationship?</h3>"));explain("Counts the directly coded sim2real, real2sim, and both values, split by research objective like every other chart. Each study contributes once.")
direction=counts(studies.direction,"direction");direction_long=long_for(studies.direction,"direction")
bar_report(direction_long,"direction","RQ1.3 Gap direction","rq1_3_direction");show_table("RQ1.3 Reported counts",visible(direction),"rq1_3_direction.csv")
explain("The second view keeps the WHAT macro category first and places three direction sub-bars beside it: sim2real, real2sim, and both. Bar lengths are distinct-paper counts within that macro. Because direction is single-coded for every paper, the three sub-bars for a macro sum to its macro count.")
direction_order=["sim2real","real2sim","both"]
what_direction=(motivation_long.groupby(["what_macro","direction"])["source_csv_row"].nunique().rename("studies").reset_index())
direction_plot=what_direction[what_direction.what_macro.isin(display_what_macro)&what_direction.direction.isin(direction_order)]
direction_pivot=direction_plot.pivot(index="what_macro",columns="direction",values="studies").fillna(0).reindex(columns=direction_order,fill_value=0);direction_pivot=direction_pivot.loc[what_macro_shown.atomic_what]
height=max(5,0.42*len(direction_pivot)+1.8);ax=direction_pivot.plot.barh(figsize=(11,height),width=.68,stacked=True,color=["#59A14F","#F28E2B","#76B7B2"]);ax.invert_yaxis();ax.set_xlabel("Distinct papers");ax.set_ylabel("WHAT macro category");ax.set_title("RQ1.3 WHAT macro categories subdivided by gap direction");ax.legend(title="Direction",bbox_to_anchor=(1.02,1),loc="upper left");ax.grid(axis="x",alpha=.25);plt.tight_layout();plt.savefig(OUTPUT_DIR/"rq1_3_what_by_direction.png",dpi=220,bbox_inches="tight");plt.show();plt.close()
show_table("RQ1.3 WHAT macro × direction counts",what_direction.sort_values(["what_macro","direction"]),"rq1_3_what_by_direction.csv")
'''

RQ2 = r'''
display(HTML("<h3>RQ2.1 — How is the proposed simulation–reality technique implemented?</h3>"));explain("Atomic HOW tags are counted once per study and mechanism. A multi-mechanism study therefore contributes to several bars. Every mechanism family with more than 5 studies is aggregated into a single macro bar in the main chart below and broken into its own sub-technique chart and table immediately underneath: Vehicle-in-the-loop; Hardware-in-the-loop (actuator/controller and sensor-in-the-loop setups); Sensor modelling (by sensing modality); Generative &amp; synthetic-data techniques; Direct sim–real measurement/comparison; Real-data scenario reconstruction (by evidence source); Traffic/NPC behaviour modelling (by modelling technique, including safety-critical/high-risk trajectory and scenario generation and calibration against real traffic data); and Policy training for sim-to-real transfer (by training paradigm: imitation-, reinforcement-, or generative-video-guided). Mechanisms with 5 or fewer studies are shown individually. The original textual technique descriptions are included only in the complete report.")
how_long=studies.assign(how_tag=studies.how_tags.str.split(" | ",regex=False)).explode("how_tag")
how_counts=how_long.how_tag.value_counts().rename("studies").rename_axis("how_tag").reset_index();how_counts["percentage"]=(100*how_counts.studies/len(studies)).round(1)

GENERATIVE_HOW_TAGS={"GAN-based generation/translation","Diffusion-based generation","Neural rendering (NeRF/3DGS)","Point-cloud/image reprojection & warping","Synthetic perturbation/augmentation","Image compositing"}
MEASUREMENT_HOW_TAGS={"Unclassified sim–real comparison (needs manual review)","SUT-behaviour sim–real comparison","Ego-vehicle-dynamics comparison","Task-based sim–real measurement","Task-based planner evaluation","Feature-space sim–real measurement","Radar sim–real measurement","LiDAR sim–real measurement","Multi-sensor sim–real measurement","Sim–generated–real measurement","Image-quality measurement","Perception-performance measurement","Simulation-only scenario scoring","Reconstruction-error trajectory-anomaly scoring calibrated to human authenticity ratings"}
RECONSTRUCTION_HOW_TAGS={"Crash/accident-evidence reconstruction","Recorded driving/sensor-log reconstruction","Map/aerial-derived reconstruction","Manual scenario recreation","Reconstruction-strategy comparison (classical vs. neural assets)"}
TRAFFIC_MODEL_HOW_TAGS={"Route/trajectory-conditioned behaviour modelling","Rule-based/behaviour-tree traffic models","Learning-based (imitation/RL) traffic models","Statistical/distributional traffic calibration","Human-reactive traffic models","Traffic/behaviour-model calibration","Field-test/traffic-flow calibration","Learned/generative trajectory models","Optimization/search-based generation"}
POLICY_LEARNING_HOW_TAGS={"Imitation-learning-based policy transfer","Reinforcement-learning-based policy transfer","Generative-video-assisted policy transfer"}
MACRO_LABELS=["Vehicle-in-the-loop (ViL)","Hardware-in-the-loop (HiL)","Sensor modelling","Generative and synthetic-data techniques","Direct sim–real measurement/comparison","Real-data scenario reconstruction","Traffic/NPC behaviour modelling","Policy training for sim-to-real transfer"]

def xil_macro(tag):
    t=tag.lower()
    if "vehicle-in-the-loop" in t:return "Vehicle-in-the-loop (ViL)"
    if "hardware-in-the-loop" in t or "hardware emulation-in-the-loop" in t or "sensor-in-the-loop" in t:return "Hardware-in-the-loop (HiL)"
    if "sensor modelling" in t:return "Sensor modelling"
    if tag in GENERATIVE_HOW_TAGS:return "Generative and synthetic-data techniques"
    if tag in MEASUREMENT_HOW_TAGS:return "Direct sim–real measurement/comparison"
    if tag in RECONSTRUCTION_HOW_TAGS:return "Real-data scenario reconstruction"
    if tag in TRAFFIC_MODEL_HOW_TAGS:return "Traffic/NPC behaviour modelling"
    if tag in POLICY_LEARNING_HOW_TAGS:return "Policy training for sim-to-real transfer"
    return tag

how_long["how_macro"]=how_long.how_tag.map(xil_macro)
macro_counts=how_long.groupby("how_macro")["source_csv_row"].nunique().rename("studies").rename_axis("how_tag").reset_index().sort_values("studies",ascending=False)
macro_counts["percentage"]=(100*macro_counts.studies/len(studies)).round(1)
macro_shown=visible(macro_counts)

macro_breakdowns={
    label:(how_long[how_long.how_macro==label].groupby("how_tag")["source_csv_row"].nunique()
           .rename("studies").reset_index().sort_values("studies",ascending=False))
    for label in MACRO_LABELS
}

if macro_shown.empty:
    display(HTML(f"<p><b>RQ2.1 Technique mechanisms:</b> no category reaches MIN_COUNT={MIN_COUNT}.</p>"))
else:
    single_bar(how_long,"how_macro","RQ2.1 Technique mechanisms (macro-aggregated)","rq2_1_how_macro",order=macro_shown.how_tag.tolist())
show_table("RQ2.1 Mechanism counts (macro)",macro_shown,"rq2_1_how_macro_counts.csv",max_rows=None)

for macro_label in [m for m in macro_shown.how_tag if m in MACRO_LABELS]:
    prefix="rq2_1_"+re.sub(r"[^a-z0-9]+","_",macro_label.lower()).strip("_")+"_sub"
    sub=macro_breakdowns[macro_label]
    if sub.empty:continue
    sub["percentage"]=(100*sub.studies/len(studies)).round(1)
    single_bar(how_long[how_long.how_macro==macro_label],"how_tag",f"RQ2.1 {macro_label} subtypes",prefix,order=sub.how_tag.tolist())
    show_table(f"RQ2.1 {macro_label} subtype counts",sub,f"{prefix}_counts.csv",max_rows=None)

HOW_STANDALONE_SUBCATEGORIES={
  "Vehicle-in-the-loop (on-road)":{
    # Read from each paper's own description of what is virtualized in the
    # loop (not merely whether "Mixed reality" also co-occurs as a tag --
    # several studies below are tagged Mixed reality too, and are counted
    # there independently; this axis instead asks what kind of real-vehicle
    # ViL study it is).
    5:"Virtual traffic/actors around a real vehicle",19:"Virtual traffic/actors around a real vehicle",
    25:"Virtual traffic/actors around a real vehicle",29:"Virtual traffic/actors around a real vehicle",
    32:"Virtual traffic/actors around a real vehicle",40:"Virtual traffic/actors around a real vehicle",
    42:"Virtual traffic/actors around a real vehicle",56:"Virtual traffic/actors around a real vehicle",
    64:"Virtual traffic/actors around a real vehicle",98:"Virtual traffic/actors around a real vehicle",
    106:"Virtual traffic/actors around a real vehicle",107:"Virtual traffic/actors around a real vehicle",
    59:"Virtual/synthetic sensor injection into a real vehicle",
    99:"Virtual/synthetic sensor injection into a real vehicle",
    105:"Virtual/synthetic sensor injection into a real vehicle",
    194:"Virtual/synthetic sensor injection into a real vehicle",
    9:"ViL/XiL testing framework or platform study",
    63:"ViL/XiL testing framework or platform study",
    78:"ViL/XiL testing framework or platform study",
    162:"ViL/XiL testing framework or platform study",
    190:"ViL/XiL testing framework or platform study",
  },
  "Mixed reality":{
    5:"Real vehicle + virtual traffic/actors",19:"Real vehicle + virtual traffic/actors",
    31:"Real vehicle + virtual traffic/actors",40:"Real vehicle + virtual traffic/actors",
    42:"Real vehicle + virtual traffic/actors",47:"Real vehicle + virtual traffic/actors",
    59:"Real vehicle + virtual traffic/actors",60:"Real vehicle + virtual traffic/actors",
    194:"Real vehicle + virtual traffic/actors",
    29:"Roadside/infrastructure (V2X/cloud)-mediated fusion",32:"Roadside/infrastructure (V2X/cloud)-mediated fusion",
    98:"Roadside/infrastructure (V2X/cloud)-mediated fusion",106:"Roadside/infrastructure (V2X/cloud)-mediated fusion",
    111:"Roadside/infrastructure (V2X/cloud)-mediated fusion",
    25:"Real pedestrian/VRU-in-the-loop",
    49:"Modular XiL framework for CAV validation",
    64:"Real vehicle teleoperated via virtual-environment view",
    107:"Real vehicle + synchronized digital-twin replica",
    157:"Cross-modality comparison study (SiL/ViL/MR/RW)",
  },
  "Human-in-the-loop":{
    111:"Driver-in-the-loop with reactive virtual traffic",145:"Driver-in-the-loop with reactive virtual traffic",
    104:"Multi-actor VR-mediated in-the-loop (driver + pedestrian)",
  },
  "Software-in-the-loop (SiL)":{
    102:"SiL-to-physical correlation study",174:"XiL laboratory validation study",
    157:"Cross-modality comparison study (SiL/ViL/MR/RW)",
  },
  "Modular X-in-the-loop framework":{
    49:"Modular XiL framework/architecture",162:"Multi-fidelity XiL architecture (SiL/HiL/ViL/proving-ground/real-world)",
  },
  "LLM-based scenario-realism assessment":{
    84:"LLM-as-judge scenario-realism assessment",160:"LLM-as-judge scenario-realism assessment",
  },
  "Physical scale-model / surrogate-platform validation":{
    23:"Small-scale/surrogate-platform suitability validation",188:"Small-scale/surrogate-platform suitability validation",
  },
}

def standalone_subcategory_chart(subset,tag,title,filename):
    """A standalone HOW tag has no macro family to fall back on for a
    sub-breakdown. Where the papers behind it have been read and grouped by
    what actually distinguishes their approach (HOW_STANDALONE_SUBCATEGORIES)
    into two or more real subcategories, draw that breakdown as a chart.
    Otherwise — no mapping, incomplete coverage, or every mapped paper lands
    in the same single subcategory — there is nothing genuinely distinct to
    chart, so no chart is drawn at all; the papers are still listed in the
    returned table."""
    mapping=HOW_STANDALONE_SUBCATEGORIES.get(tag,{})
    rows=subset.drop_duplicates("source_csv_row")
    if mapping and set(rows.source_csv_row).issubset(mapping):
        labeled=rows.copy()
        labeled["subcategory"]=labeled.source_csv_row.map(mapping)
        if labeled.subcategory.nunique()>=2:
            counts_tab=labeled.groupby("subcategory")["source_csv_row"].nunique().rename("studies").reset_index().sort_values("studies",ascending=False)
            counts_tab["percentage"]=(100*counts_tab.studies/len(studies)).round(1)
            single_bar(labeled,"subcategory",title,filename,order=counts_tab.subcategory.tolist())
            return counts_tab,True
    return rows[["source_csv_row","paper_title"]].sort_values("source_csv_row"),False

for standalone_label,standalone_count in macro_shown.set_index("how_tag").studies.items():
    if standalone_label in MACRO_LABELS or standalone_count<=1:continue
    prefix="rq2_1_"+re.sub(r"[^a-z0-9]+","_",standalone_label.lower()).strip("_")+"_papers"
    paper_table,charted=standalone_subcategory_chart(how_long[how_long.how_macro==standalone_label],standalone_label,f"RQ2.1 {standalone_label} — breakdown",prefix)
    if charted:
        explain(f"\"{standalone_label}\" does not belong to a larger macro family, but its papers have been read closely enough to name two or more genuinely distinct approaches, shown here as a category breakdown.")
    else:
        explain(f"\"{standalone_label}\" does not belong to a larger macro family and is too small or too homogeneous a group ({standalone_count} studies) to support a genuinely distinct sub-category breakdown, so no chart is drawn — its papers are listed directly below.")
    show_table(f"RQ2.1 {standalone_label} — breakdown",paper_table,f"{prefix}.csv",max_rows=None)

if not FILTER_AGGREGATES:show_table("RQ2.1 Complete study-level technique descriptions",studies[["source_csv_row","paper_title","detailed_what_how","simplified_how","how_tags"]],"rq2_1_techniques_by_study.csv")
display(HTML("<h3>RQ2.2 — What are the execution levels of the proposed approach?</h3>"));explain("The six independently coded execution components are shown at two granularities, back to back: first grouped into the three functional groups Scene, Sensing, and Control (each built from a pair of raw components), then broken back out into all six raw components individually. Both use the same six-state taxonomy, defined below before each batch of charts, and both go through the same three chart types: state shares, a state count matrix, and every exact combination actually used. Nothing is dropped between the two granularities — the group-level charts are a coarser view of exactly the same coded data as the component-level charts.")
approach_group=state_frame("approach",GROUP_SCHEME)
approach_component=state_frame("approach",COMPONENT_SCHEME)
display(HTML("<h4>Three functional groups</h4>"))
state_legend(GROUP_SCHEME,"three functional groups")
display(HTML("<h5>Group state shares</h5>"));explain("For each group, the share of studies in each state, read directly off one shared 0-100% scale.")
state_proportion_chart("approach",GROUP_SCHEME,"RQ2.2 Approach group state shares","rq2_2_approach_group_shares")
display(HTML("<h5>Group × state counts</h5>"));explain("The same counts in matrix form, for looking up an exact group/state cell value.")
state_heatmap(visible(state_long("approach",GROUP_SCHEME)),GROUP_SCHEME,"RQ2.2 Approach group execution states","rq2_2_approach_group_heatmap")
display(HTML("<h5>Every Scene/Sensing/Control combination actually used</h5>"));explain("Every distinct combination of the three groups' states, ranked by how many studies use it (top panel, objective-coloured like every other chart) with the literal state of each group spelled out directly below it. No combination is merged into an abstract tier, score, or short code — the exact breakdown is also in the table and CSV.")
approach_group_combo_table=state_combination_chart("approach",GROUP_SCHEME,"RQ2.2 Approach — every group combination","rq2_2_approach_group_combinations");show_table("RQ2.2 Approach group combinations",approach_group_combo_table,"rq2_2_approach_group_combinations.csv",max_rows=None)
display(HTML("<h4>All six raw components individually</h4>"));explain("The same three charts, now at full component resolution instead of paired into groups — nothing here is a new coding, only a finer-grained view of the same six-component data behind the group charts above.")
state_legend(COMPONENT_SCHEME,"six raw components")
display(HTML("<h5>Component state shares</h5>"))
state_proportion_chart("approach",COMPONENT_SCHEME,"RQ2.2 Approach component state shares","rq2_2_approach_component_shares")
display(HTML("<h5>Component × state counts</h5>"))
state_heatmap(visible(state_long("approach",COMPONENT_SCHEME)),COMPONENT_SCHEME,"RQ2.2 Approach component execution states","rq2_2_approach_component_heatmap")
display(HTML("<h5>Every six-component combination actually used</h5>"));explain("Every distinct combination of all six components' states, ranked by how many studies use it — the full, un-grouped detail behind the group-level combination chart above.")
approach_component_combo_table=state_combination_chart("approach",COMPONENT_SCHEME,"RQ2.2 Approach — every six-component combination","rq2_2_approach_component_combinations");show_table("RQ2.2 Approach six-component combinations",approach_component_combo_table,"rq2_2_approach_component_combinations.csv",max_rows=None)
display(HTML("<h3>RQ2.3 — Which aspect(s) of the gap are addressed?</h3>"));explain("Counts the directly coded gap-type field, including combined categories when several aspects were equally central, split by research objective like every other chart.")
gap=counts(studies.gap_type,"gap_type");gap_long=long_for(studies.gap_type,"gap_type")
bar_report(gap_long,"gap_type","RQ2.3 Addressed gap aspects","rq2_3_gap_type");show_table("RQ2.3 Gap-aspect counts",visible(gap),"rq2_3_gap_type.csv")
explain("The directly coded gap-type field (perception/scenario/actuation/behaviour/entire, split into its atomic components wherever a study combines several) is independently coded from the RQ1.2 atomic WHAT targets. This heatmap crosses the two against the complete list of atomic WHAT targets, showing which concrete WHAT targets sit behind each coded gap aspect — for example whether studies coded 'perception' concentrate on camera/LiDAR/radar targets as expected, or also cover targets a reader might not anticipate from the gap-type label alone.")
gap_component_long=studies.assign(gap_component=studies.gap_type.str.split(r"\s*\+\s*",regex=True)).explode("gap_component")
gap_what_long=gap_component_long.merge(motivation_long[["source_csv_row","atomic_what"]],on="source_csv_row",how="left")
gap_what_counts=gap_what_long.groupby(["gap_component","atomic_what"])["source_csv_row"].nunique().rename("studies").reset_index()
gap_order=gap_component_long.groupby("gap_component").source_csv_row.nunique().sort_values(ascending=False).index.tolist()
what_order=motivation_long.groupby("atomic_what").source_csv_row.nunique().sort_values(ascending=False).index.tolist()
gap_what_matrix=gap_what_counts.pivot(index="gap_component",columns="atomic_what",values="studies").fillna(0).reindex(index=gap_order,columns=what_order,fill_value=0)
fig,ax=plt.subplots(figsize=(max(11,.55*len(what_order)),max(3.2,.5*len(gap_order)+1)));im=ax.imshow(gap_what_matrix,cmap="Blues",aspect="auto",vmin=0)
ax.set_xticks(range(len(gap_what_matrix.columns)),gap_what_matrix.columns,rotation=45,ha="right",fontsize=8)
ax.set_yticks(range(len(gap_what_matrix.index)),gap_what_matrix.index)
gap_what_vmax=gap_what_matrix.values.max() or 1
for i in range(len(gap_what_matrix.index)):
  for j in range(len(gap_what_matrix.columns)):
    v=int(gap_what_matrix.iloc[i,j])
    if v:ax.text(j,i,str(v),ha="center",va="center",fontsize=7.5,color="white" if v>gap_what_vmax*0.55 else "#222222")
ax.set_title("RQ2.3 Gap aspect × WHAT target");fig.colorbar(im,ax=ax,label="Studies",shrink=.8);fig.tight_layout()
fig.savefig(OUTPUT_DIR/"rq2_3_gap_type_by_what.png",dpi=220,bbox_inches="tight");plt.show();plt.close(fig)
show_table("RQ2.3 Gap aspect × WHAT target counts",gap_what_counts.sort_values(["gap_component","studies"],ascending=[True,False]),"rq2_3_gap_type_by_what.csv",max_rows=None)
'''

RQ3 = r'''
display(HTML("<h3>RQ3.1 — How is the proposed technique validated or the gap measured?</h3>"));explain("The raw plot distinguishes records with a reported validation/measurement procedure from explicit or missing non-validation, split by research objective like every other chart. The complete report then provides the original procedure text.")
validation_counts=counts(studies.validation_status,"validation_status")
validation_long=long_for(studies.validation_status,"validation_status")
bar_report(validation_long,"validation_status","RQ3.1 Validation/measurement status","rq3_1_validation");show_table("RQ3.1 Status counts",visible(validation_counts),"rq3_1_validation_status.csv")
if not FILTER_AGGREGATES:show_table("RQ3.1 Complete procedures by study",studies[["source_csv_row","paper_title","validation_status","validation"]],"rq3_1_validation_by_study.csv")
explain("Every study originally coded with a blank validation-procedure field ('NO', '-', or '[MISSING]') was individually re-read against its source PDF, since a blank field could mean either a genuine absence of validation or simply that the field was never filled in. Three studies turned out to have real, previously-uncaptured validation (a localization comparison against DGPS/SLAM; a domain-adaptation evaluation on real BEV LiDAR data with a quantified AP improvement; a co-simulation framework validated against real track data with quantified calibration and behavioural error) and have been moved out of this status entirely into 'Validation/measurement reported'. Three more turned out to report a genuine real-world demonstration without a formal quantitative comparison, and are now described as such rather than left blank. Two were confirmed to report no validation of any kind (pure review/design papers with no results section) and are coded as such below, rather than as a blank/unrecorded field.")
def no_validation_subcategory(value):
    t=norm(value).lower()
    if t in {"no","none","-","[missing]"}:return "No validation performed (confirmed: no results/evaluation section)"
    return "Explicitly demonstration-only (no formal comparison stated)"
no_validation=studies[studies.validation_status=="No validation reported"].copy()
no_validation["subcategory"]=no_validation.validation.map(no_validation_subcategory)
show_table("RQ3.1 'No validation reported' papers",no_validation[["source_csv_row","paper_title","subcategory","validation"]],"rq3_1_no_validation_papers.csv",max_rows=None,force=True)
display(HTML("<h3>RQ3.2 — What are the execution levels of the validation reference?</h3>"));explain("The same two-granularity taxonomy used for RQ2.2 — three functional groups, then all six raw components individually — is applied here to the validation or measurement reference, defined again below before each batch of charts.")
reference_group=state_frame("reference",GROUP_SCHEME)
reference_component=state_frame("reference",COMPONENT_SCHEME)
display(HTML("<h4>Three functional groups</h4>"))
state_legend(GROUP_SCHEME,"three functional groups")
display(HTML("<h5>Approach vs. reference — how real is each group, side by side</h5>"));explain("A direct cross-cut between RQ2.2 and RQ3.2: for each group, the share of studies that are real-bearing (Real, Partially real, Mixed, or Both) in the proposed approach versus in the validation reference. Neither section shows this comparison on its own — a large gap between the two bars for a group means that group is tested against reality far more (or less) than it is actually built on it.")
approach_vs_reference_group_table=approach_vs_reference_chart(approach_group,reference_group,GROUP_SCHEME,"RQ3.2 Approach vs. reference — % real-bearing, by group","rq3_2_approach_vs_reference_group");show_table("RQ3.2 Approach vs. reference real-bearing share (groups)",approach_vs_reference_group_table,"rq3_2_approach_vs_reference_group.csv")
display(HTML("<h5>Group state shares</h5>"))
state_proportion_chart("reference",GROUP_SCHEME,"RQ3.2 Reference group state shares","rq3_2_reference_group_shares")
display(HTML("<h5>Group × state counts</h5>"))
state_heatmap(visible(state_long("reference",GROUP_SCHEME)),GROUP_SCHEME,"RQ3.2 Reference group execution states","rq3_2_reference_group_heatmap")
display(HTML("<h5>Every Scene/Sensing/Control combination actually used</h5>"));explain("Every distinct combination of the three groups' states on the reference side, ranked by how many studies use it, with the literal state of each group spelled out directly below it.")
reference_group_combo_table=state_combination_chart("reference",GROUP_SCHEME,"RQ3.2 Reference — every group combination","rq3_2_reference_group_combinations");show_table("RQ3.2 Reference group combinations",reference_group_combo_table,"rq3_2_reference_group_combinations.csv",max_rows=None)
display(HTML("<h4>All six raw components individually</h4>"));explain("The same charts, now at full component resolution instead of paired into groups.")
state_legend(COMPONENT_SCHEME,"six raw components")
display(HTML("<h5>Approach vs. reference by component</h5>"))
approach_vs_reference_component_table=approach_vs_reference_chart(approach_component,reference_component,COMPONENT_SCHEME,"RQ3.2 Approach vs. reference — % real-bearing, by component","rq3_2_approach_vs_reference_component");show_table("RQ3.2 Approach vs. reference real-bearing share (components)",approach_vs_reference_component_table,"rq3_2_approach_vs_reference_component.csv")
display(HTML("<h5>Component state shares</h5>"))
state_proportion_chart("reference",COMPONENT_SCHEME,"RQ3.2 Reference component state shares","rq3_2_reference_component_shares")
display(HTML("<h5>Component × state counts</h5>"))
state_heatmap(visible(state_long("reference",COMPONENT_SCHEME)),COMPONENT_SCHEME,"RQ3.2 Reference component execution states","rq3_2_reference_component_heatmap")
display(HTML("<h5>Every six-component combination actually used</h5>"));explain("Every distinct combination of all six components' states on the reference side, ranked by how many studies use it — the full, un-grouped detail behind the group-level combination chart above.")
reference_component_combo_table=state_combination_chart("reference",COMPONENT_SCHEME,"RQ3.2 Reference — every six-component combination","rq3_2_reference_component_combinations");show_table("RQ3.2 Reference six-component combinations",reference_component_combo_table,"rq3_2_reference_component_combinations.csv",max_rows=None)
display(HTML("<h3>RQ3.3 — Which metrics are used?</h3>"));explain("The extracted metric text is coded into a reproducible multi-label taxonomy organised entirely by what is actually being compared, not by which statistical formula happens to be used: Sensor/perception comparison (image/feature quality and perception-task metrics — how well simulated sensor data or a perception model's output matches reality), ADS behaviour/trajectory comparison (trajectory, planning, and multi-object-tracking metrics — how the ego vehicle or tracked agents behave/move), Vehicle dynamics/actuation comparison (steering/braking/acceleration/jerk/control-tracking), Safety/event outcome, Runtime/computational evidence, Qualitative assessment (human/expert/Likert-based judgement, including qualitative behaviour comparisons), or no metric. There is deliberately no separate 'Statistical/distributional' category: RMSE, correlation, variance, distribution-distance tests, and composite/coverage scores are formula types, not domains, so every occurrence of one is filed under whichever domain(s) its own study's text actually measures instead. A study contributes once to every matching category, and every category is broken down into the specific named metric it covers below — no residual 'other' or 'unspecified' bucket survives at the top level.")
metric_long=studies.assign(metric_category=studies.metric_categories.str.split(" | ",regex=False)).explode("metric_category")
metric_counts=metric_long.metric_category.value_counts().rename("studies").rename_axis("metric_category").reset_index();metric_counts["percentage"]=(100*metric_counts.studies/len(studies)).round(1)
bar_report(metric_long,"metric_category","RQ3.3 Metric categories","rq3_3_metrics");show_table("RQ3.3 Metric-category counts",visible(metric_counts),"rq3_3_metric_categories.csv")
explain("Each category above is broken into the specific named metrics it covers. Sensor/perception comparison covers hardware/perception-model output fidelity (image quality, detection, point clouds, sensor field-of-view/detection-range, sensor-model qualification); ADS behaviour/trajectory comparison covers how the ego vehicle or tracked agents move and behave (trajectory/path error and distance, tracking, planning, policy-training success, ADAS-function pass/fail, control-tracking — moved here from vehicle dynamics since these describe functional/behavioural outcomes, not raw physical dynamics values); Vehicle dynamics/actuation comparison is now limited to the physical dynamics quantities themselves (acceleration/steering/braking/yaw/jerk/energy, named model-plant dynamics-mismatch metrics). 'Statistical/distributional' no longer exists as a category: every study that previously landed there only (RMSE/MAE, correlation, variance, distribution-distance tests, composite robustness/coverage scores, reported with no domain named in the metric text itself) was individually re-read against its detailed WHAT/HOW description and assigned to the real domain its study measures — e.g. a bare 'RMSE' from a vehicle-dynamics parameter-estimation paper is now Vehicle dynamics/actuation comparison, an LLM-as-judge robustness score is now ADS behaviour/trajectory comparison. For studies that already had a domain category from other wording in the same text, the statistical formula is simply added as a further subtag inside that existing category rather than a separate bucket. A study contributes once to every specific metric its text mentions within that category; every study previously landing in a generic 'other specified metric' or 'no specific metric named' bucket was individually re-read and given a real, specific subtag — none remain. Sub-metric charts follow the same order and objective color-split as the category chart above.")
METRIC_LABELS=["Sensor/perception comparison","ADS behaviour/trajectory comparison","Vehicle dynamics/actuation comparison","Safety/event outcome","Runtime/computational","Qualitative assessment"]
metric_long["metric_subtags"]=metric_long.apply(lambda r:subcategory_of(r.metric_subcategories,r.metric_category),axis=1)
metric_sub_long=metric_long.assign(metric_subtag=metric_long.metric_subtags.str.split(" | ",regex=False)).explode("metric_subtag")
metric_order=[c for c in visible(metric_counts).metric_category if c in METRIC_LABELS]
for metric_cat in metric_order:
    sub=(metric_sub_long[metric_sub_long.metric_category==metric_cat].groupby("metric_subtag")["source_csv_row"].nunique()
         .rename("studies").reset_index().sort_values("studies",ascending=False))
    if sub.empty:continue
    sub["percentage"]=(100*sub.studies/len(studies)).round(1)
    prefix="rq3_3_"+re.sub(r"[^a-z0-9]+","_",metric_cat.lower()).strip("_")+"_sub"
    single_bar(metric_sub_long[metric_sub_long.metric_category==metric_cat],"metric_subtag",f"RQ3.3 {metric_cat} — specific metrics",prefix,order=sub.metric_subtag.tolist())
    show_table(f"RQ3.3 {metric_cat} specific-metric counts",sub,f"{prefix}_counts.csv",max_rows=None)
study_metric_subtags=metric_sub_long.groupby("source_csv_row").metric_subtag.apply(lambda s:" | ".join(dict.fromkeys(s))).rename("metric_subtags").reset_index()
studies=studies.merge(study_metric_subtags,on="source_csv_row",how="left")
if not FILTER_AGGREGATES:show_table("RQ3.3 Metrics and assigned categories by study",studies[["source_csv_row","paper_title","metric_categories","metric_subtags","metrics"]],"rq3_3_metrics_by_study.csv")
explain("The metric taxonomy above is coded purely from the reported evaluation metrics, independent of what the study's gap targets (RQ1.2). This heatmap crosses the two, showing which concrete WHAT targets sit behind each metric category — for example whether Sensor/perception comparison metrics concentrate on camera/LiDAR perception targets as expected, or also appear against scenario/behaviour targets.")
metric_what_long=metric_long.merge(motivation_long[["source_csv_row","atomic_what"]],on="source_csv_row",how="left")
metric_what_counts=metric_what_long.groupby(["metric_category","atomic_what"])["source_csv_row"].nunique().rename("studies").reset_index()
metric_what_order=[c for c in metric_counts.sort_values("studies",ascending=False).metric_category if c in metric_what_long.metric_category.unique()]
metric_what_matrix=metric_what_counts.pivot(index="metric_category",columns="atomic_what",values="studies").fillna(0).reindex(index=metric_what_order,columns=what_order,fill_value=0)
fig,ax=plt.subplots(figsize=(max(11,.55*len(what_order)),max(3.2,.5*len(metric_what_order)+1)));im=ax.imshow(metric_what_matrix,cmap="Blues",aspect="auto",vmin=0)
ax.set_xticks(range(len(metric_what_matrix.columns)),metric_what_matrix.columns,rotation=45,ha="right",fontsize=8)
ax.set_yticks(range(len(metric_what_matrix.index)),metric_what_matrix.index)
metric_what_vmax=metric_what_matrix.values.max() or 1
for i in range(len(metric_what_matrix.index)):
  for j in range(len(metric_what_matrix.columns)):
    v=int(metric_what_matrix.iloc[i,j])
    if v:ax.text(j,i,str(v),ha="center",va="center",fontsize=7.5,color="white" if v>metric_what_vmax*0.55 else "#222222")
ax.set_title("RQ3.3 Metric category × WHAT target");fig.colorbar(im,ax=ax,label="Studies",shrink=.8);fig.tight_layout()
fig.savefig(OUTPUT_DIR/"rq3_3_metrics_by_what.png",dpi=220,bbox_inches="tight");plt.show();plt.close(fig)
show_table("RQ3.3 Metric category × WHAT target counts",metric_what_counts.sort_values(["metric_category","studies"],ascending=[True,False]),"rq3_3_metrics_by_what.csv",max_rows=None)
'''

RQ4 = r'''
SUT_NON_TECHNOLOGY_CATEGORIES={"SUT not specified","No SUT (none, or simulator/testbed fidelity only)","SUT function not recorded (field left blank)","Unspecified SUT function (described, but no function named)"}
SUT_SUBTAG_RULES={
  "Camera perception":[
    (r"yolo","YOLO-family detector"),(r"faster r-?cnn","Faster R-CNN detector"),
    (r"bevformer|\bbev\b","BEV-based detector/feature extractor"),
    (r"deeplab|segment","Segmentation model"),(r"depth estimation","Depth estimation"),
    (r"lane detection|lane keeping","Lane detection"),(r"classif","Classification model"),
    (r"temporal object tracking|object tracking","Object detection + tracking"),
    (r"bounding box|recurrent rolling convolution|\brrc\b|automotive camera perception and dnn vehicle detectors|^object detecor$|dnn vehicle detector|vehicle detectors?|object detector|object detection|^camera perception system$|^autonomous driving perception model$","Object detector"),
    (r"vru detection|driver warning system|pedestrian detect","VRU (pedestrian/cyclist) detection & driver-warning system"),
    (r"ai 3d human pose estimator","3D human pose estimation"),
  ],
  "LiDAR perception":[
    (r"pointpillars","PointPillars"),(r"pv.?rcnn","PV-RCNN"),(r"voxelnet","VoxelNet"),
    (r"pointnet","PointNet(++)"),(r"rangenet","RangeNet++"),(r"polarnet","PolarNet"),
    (r"prediction and tracking of lidar objects|object.based lidar model","LiDAR object detection + tracking"),
    (r"lidar intensity simulator|guide intensity simulation","LiDAR intensity simulation model"),
    (r"lidar sensor simulation model|data.driven.{0,15}sensor model","Sensor simulation/generative model"),
    (r"bev feature extractor|\bbev\b","BEV feature extractor (LiDAR/camera fusion)"),
    (r"segment","Point-cloud segmentation"),(r"detect","3D object detection"),
    (r"lidar perception system\b","3D object detection"),
  ],
  "Ego-vehicle control/actuation":[
    (r"acc\b|adaptive cruise|cruise control(?:ler)?","Adaptive cruise control"),(r"\baeb\b|emergency braking","Automatic emergency braking"),
    (r"lane keeping|lane change","Lane keeping/changing control"),(r"nmpc|model predictive|\bmpc\b","Model predictive control"),
    (r"\bpid\b","PID trajectory-tracking controller"),
    (r"obstacle avoidance","Obstacle-avoidance planner"),
    (r"rule.based reactive","Rule-based reactive control policy"),
    (r"trajectory planner and vehicle controller","Integrated trajectory-planner + vehicle controller"),
    (r"steering|\beps\b|deepdriving","Steering control"),
    (r"waypoint follower","Waypoint-following control"),
    (r"ground robot dynamics","Ground-robot dynamics control system"),
    (r"\becu\b|control stack","Vehicle control stack/ECU"),
  ],
  "Planning/trajectory prediction":[
    (r"trajector prediction|motion forecasting|chauffeurnet|chauffernet|chaffeurnet","Trajectory/motion prediction"),
    (r"valet parking|\bavp\b","Automated valet-parking planner"),
    (r"decision.?making","Decision-making/planning policy"),
    (r"path planning|planner|planning","Path/motion planning"),
  ],
  "Full ADS":[
    (r"black.box","Black-box AV (architecture undisclosed)"),
    (r"end.?to.?end|\be2e\b|uniad|\bvad\b|latent.transfuser|transfuser|deepdriving|\broh\b.{0,15}(driving policy|autonomous)|imitation learning|sensors to control","End-to-end"),
    (r"modular|autoware|apollo|open.?cav|\bstack\b|(decision|planning).{0,25}control|control.{0,25}(decision|planning)|custom adas|longitudinal and lateral control","Modular"),
  ],
  "Localization":[
    (r"slam","SLAM-based localization"),(r"gnss|gps","GNSS/GPS-based localization"),
    (r"ndt mapping","NDT-based mapping/localization"),
    (r"custom modular ads.{0,40}localization","Modular-ADS localization component"),
  ],
  "Multi-sensor fusion perception/tracking":[
    (r"virtrack|jmodt|dfmot|yontd|bevfusion","BevFusion/VirTrack/JMODT/DFMOT/YONTD (fusion-tracking algorithm)"),
    (r"sensor fusion camera.?lidar.?radar","Camera/LiDAR/radar sensor-fusion perception"),
    (r"^not specified$","Camera/LiDAR/radar sensor-fusion perception (mixed-reality test setup)"),
  ],
}
SUT_SUBTAG_ROW_OVERRIDES={
  ("Full ADS",157): "Modular | End-to-end",
  ("Full ADS",188): "Modular | End-to-end",
  # Rows that matched their SUT category but no technique rule, which used to fall through to an
  # "Other/unspecified <category> function" label. The papers name the system:
  ("Camera perception",209): "ADAS smart front camera (sensor and embedded software)",  # CHASE: production ADAS "smart camera" tested in HiL
  ("Ego-vehicle control/actuation",227): "Automatic emergency braking | Lane keeping/changing control",  # real AEB and LKAS controllers of the test vehicle
  ("Full ADS",213): "Simplified driving function with PID vehicle controller (path following, TTC-based braking)",  # DynViL ROS prototype: ADF + longitudinal/lateral controller
}
def sut_subtags(category,value,source_csv_row=None):
    if (category,source_csv_row) in SUT_SUBTAG_ROW_OVERRIDES:return SUT_SUBTAG_ROW_OVERRIDES[(category,source_csv_row)]
    if category not in SUT_SUBTAG_RULES:return category
    t=norm(value).lower();labels=[label for pattern,label in SUT_SUBTAG_RULES[category] if re.search(pattern,t)]
    return " | ".join(dict.fromkeys(labels)) if labels else f"Other/unspecified {category.lower()} function"


for number,column,raw_column,title,question,method in [("4.1","simulator_tags","simulator_dataset","Simulation platforms","Which simulation platforms are employed?","Counts simplified platform/resource tags; multi-platform studies can contribute to several categories."),("4.2","hardware_tags","hardware","Hardware platforms","Which hardware platforms are used?","Counts simplified hardware-role tags; the complete report retains the exact equipment descriptions."),("4.3","sut_tags","sut","Systems under test","Which systems are evaluated?","Counts simplified SUT-function tags; multi-function systems can contribute to several categories. Every category here is a vehicle/ADS-side subsystem (Camera/LiDAR/Radar/multi-sensor perception, Localization, Planning, Ego-vehicle control, Full ADS, LLM-based reasoning agent) whose own sim-to-real gap is what the study measures — an end-to-end learned driving policy (e.g. a camera-to-steering network) is coded under whichever concrete function it performs (Ego-vehicle control if it is a single control axis, Full ADS if it is a complete sensors-to-actuation pipeline) rather than a separate 'learned policy' category, since 'learned' describes how the function was built, not what it does. Any study whose SUT is not itself an ADS/ADAS/driving function — a traffic-agent behaviour model, a scenario/data-generation tool, a bare sensor-fidelity/rendering model, a human driver, or a test methodology/framework paper with no specific system named — falls under 'No SUT (none, or simulator/testbed fidelity only)' rather than being force-fit into one of these categories.")]:
    display(HTML(f"<h3>RQ{number} — {question}</h3>"));explain(method+" Every bar is split by research objective like every other chart in this report.")
    long=studies.assign(category=studies[column].str.split(" | ",regex=False)).explode("category")
    table=long.category.value_counts().rename("studies").rename_axis("category").reset_index();table["percentage"]=(100*table.studies/len(studies)).round(1)
    stem="rq"+number.replace(".","_")
    if number=="4.1":
        order=category_order(long,"category")
        if FILTER_AGGREGATES:
            counts_now=long.groupby("category")["source_csv_row"].nunique()
            order=[c for c in order if counts_now.get(c,0)>MIN_COUNT]
        if not order:
            display(HTML(f"<p><b>RQ{number} {title}:</b> no category reaches MIN_COUNT={MIN_COUNT}.</p>"))
        else:
            single_bar(long,"category",f"RQ{number} {title}",stem,order=order)
    else:
        bar_report(long,"category",f"RQ{number} {title}",stem)
    show_table(f"RQ{number} Reported counts",visible(table),f"{stem}_counts.csv")
    if not FILTER_AGGREGATES:show_table(f"RQ{number} Complete exact values by study",studies[["source_csv_row","paper_title",raw_column,column]],f"{stem}_by_study.csv")
    if number=="4.1" and "Dataset/reference data" in table.category.values:
        explain("'Dataset/reference data' is the single largest simulation-platform category, but as a bucket it hides very different kinds of resources — perception benchmarks (KITTI/nuScenes/Waymo/Cityscapes/BDD100K), traffic-trajectory datasets (NGSIM/HighD/rounD/INTERACTION), and named or custom study-collected data. This breaks that bucket down into the specific named datasets its members actually use; every member is given a real, specific dataset name — none fall into a residual 'Other' bucket.")
        ds_members=long[long.category=="Dataset/reference data"].copy()
        ds_members["dataset_subtag"]=ds_members.dataset_tags
        ds_sub_long=ds_members.assign(dataset_subtag=ds_members.dataset_subtag.str.split(" | ",regex=False)).explode("dataset_subtag")
        ds_sub=ds_sub_long.groupby("dataset_subtag")["source_csv_row"].nunique().rename("studies").reset_index().sort_values("studies",ascending=False)
        ds_sub["percentage"]=(100*ds_sub.studies/len(studies)).round(1)
        single_bar(ds_sub_long,"dataset_subtag","RQ4.1 Dataset/reference data — specific datasets","rq4_1_dataset_sub",order=ds_sub.dataset_subtag.tolist())
        show_table("RQ4.1 Dataset/reference data specific-dataset counts",ds_sub,"rq4_1_dataset_sub_counts.csv",max_rows=None)
    if number=="4.3":
        explain("For every system-under-test category reported by 5 or more studies, the studies within it are broken down further into the specific named model/architecture/technique their text mentions (e.g. YOLO-family detector, PointPillars, adaptive cruise control). Every study in every category is given a real, specific technique label read off its own SUT text — none fall into a residual 'Other/unspecified' bucket. The handful of purely count-of-studies meta-categories (SUT not recorded, no SUT, generic/unnamed ADS) are not broken down further since they carry no technology detail to split. Each sub-technology chart follows the same category order and objective color-split as the chart above.")
        for _,row in table[(table.studies>=5)&(~table.category.isin(SUT_NON_TECHNOLOGY_CATEGORIES))].sort_values("studies",ascending=False).iterrows():
            cat=row.category
            members=long[long.category==cat].copy()
            members["sut_subtag"]=members.apply(lambda r:subcategory_of(r.sut_subcategories,cat),axis=1)
            sub_long=members.assign(sut_subtag=members.sut_subtag.str.split(" | ",regex=False)).explode("sut_subtag")
            sub=sub_long.groupby("sut_subtag")["source_csv_row"].nunique().rename("studies").reset_index().sort_values("studies",ascending=False)
            if sub.empty:continue
            sub["percentage"]=(100*sub.studies/len(studies)).round(1)
            prefix="rq4_3_"+re.sub(r"[^a-z0-9]+","_",cat.lower()).strip("_")+"_sub"
            single_bar(sub_long,"sut_subtag",f"RQ4.3 {cat} — specific technology",prefix,order=sub.sut_subtag.tolist())
            show_table(f"RQ4.3 {cat} specific-technology counts",sub,f"{prefix}_counts.csv",max_rows=None)
display(HTML("<h4>RQ4.1/4.2 × WHAT target</h4>"));explain("The simulation-platform and hardware categories above are coded purely from what equipment/resources a study reports, independent of what the study's gap targets (RQ1.2). These two heatmaps cross each against the atomic WHAT targets, showing which concrete gap targets sit behind each platform/hardware category — for example whether CARLA usage concentrates on scenario/actor-realism targets as expected, or also appears against perception or control targets. Simulation platforms named by only 1-2 studies are omitted from the heatmap to keep it readable; the full per-study crossing is in the CSV.")
for cat_col,what_stem,heat_title,min_studies in [("simulator_tags","rq4_1_simulator_by_what","RQ4.1 Simulation platform × WHAT target",3),("hardware_tags","rq4_2_hardware_by_what","RQ4.2 Hardware platform × WHAT target",1)]:
    cw_long=studies.assign(category=studies[cat_col].str.split(" | ",regex=False)).explode("category")
    cw_long=cw_long.merge(motivation_long[["source_csv_row","atomic_what"]],on="source_csv_row",how="left")
    cw_counts_all=cw_long.groupby(["category","atomic_what"])["source_csv_row"].nunique().rename("studies").reset_index()
    cw_totals=cw_long.groupby("category").source_csv_row.nunique().sort_values(ascending=False)
    cw_order=cw_totals[cw_totals>=min_studies].index.tolist()
    cw_counts=cw_counts_all[cw_counts_all.category.isin(cw_order)]
    cw_matrix=cw_counts.pivot(index="category",columns="atomic_what",values="studies").fillna(0).reindex(index=cw_order,columns=what_order,fill_value=0)
    fig,ax=plt.subplots(figsize=(max(11,.55*len(what_order)),max(3.2,.4*len(cw_order)+1)));im=ax.imshow(cw_matrix,cmap="Blues",aspect="auto",vmin=0)
    ax.set_xticks(range(len(cw_matrix.columns)),cw_matrix.columns,rotation=45,ha="right",fontsize=8)
    ax.set_yticks(range(len(cw_matrix.index)),cw_matrix.index,fontsize=7.5)
    cw_vmax=cw_matrix.values.max() or 1
    for i in range(len(cw_matrix.index)):
      for j in range(len(cw_matrix.columns)):
        v=int(cw_matrix.iloc[i,j])
        if v:ax.text(j,i,str(v),ha="center",va="center",fontsize=7,color="white" if v>cw_vmax*0.55 else "#222222")
    ax.set_title(heat_title);fig.colorbar(im,ax=ax,label="Studies",shrink=.8);fig.tight_layout()
    fig.savefig(OUTPUT_DIR/f"{what_stem}.png",dpi=220,bbox_inches="tight");plt.show();plt.close(fig)
    show_table(f"{heat_title} counts",cw_counts_all.sort_values(["category","studies"],ascending=[True,False]),f"{what_stem}.csv",max_rows=None)
display(HTML("<h3>RQ4.4 — How are evaluation scenarios or test cases generated?</h3>"));explain("Rule-based, multi-label reporting of the extracted data-creation text as manual/predefined, simulation-generated, real-world reconstruction, recorded/replayed, real-world executed, hybrid/mixed, or unclear, split by research objective like every other chart. The complete report includes the original text for verification.")
generation_long=studies.assign(category=studies.test_generation_role.str.split(" | ",regex=False)).explode("category")
generation_counts=generation_long.category.value_counts().rename("studies").rename_axis("generation_role").reset_index();generation_counts["percentage"]=(100*generation_counts.studies/len(studies)).round(1)
bar_report(generation_long,"category","RQ4.4 Test-case generation","rq4_4_generation");show_table("RQ4.4 Reported counts",visible(generation_counts),"rq4_4_generation_counts.csv")
if not FILTER_AGGREGATES:show_table("RQ4.4 Complete generation descriptions by study",studies[["source_csv_row","paper_title","test_generation_role","data_usage"]],"rq4_4_generation_by_study.csv")
display(HTML("<h3>RQ4.5 — Which testing artefacts are publicly available?</h3>"));explain("Classifies the extracted artefact field into code, dataset, both, other artefact, or none/unspecified, split by research objective like every other chart. The complete report retains the original entry.")
artefacts=counts(studies.artefact_status,"artefact_status")
artefacts_long=long_for(studies.artefact_status,"artefact_status")
bar_report(artefacts_long,"artefact_status","RQ4.5 Public testing artefacts","rq4_5_artefacts");show_table("RQ4.5 Reported counts",visible(artefacts),"rq4_5_artefacts.csv")
if not FILTER_AGGREGATES:show_table("RQ4.5 Complete artefacts by study",studies[["source_csv_row","paper_title","artefact_status","artefacts"]],"rq4_5_artefacts_by_study.csv")
'''

