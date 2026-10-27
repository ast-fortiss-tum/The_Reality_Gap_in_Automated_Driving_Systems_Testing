"""Shared, auditable helpers for the deep systematic-mapping notebooks."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = RESULTS_DIR
PACKAGE_ROOT = RESULTS_DIR.parent
DATA_DIR = PACKAGE_ROOT / "data"
CSV_PATH = DATA_DIR / "primary_studies_coding.csv"
COMPONENTS = ["scenario", "actors", "perception", "localization", "actuation", "ecu_control"]
APPROACH_POSITIONS = [9, 10, 11, 12, 13, 14]
REFERENCE_POSITIONS = [18, 19, 20, 21, 22, 23]
REAL_BEARING = {"REAL", "MIXED", "BOTH"}
SIM_BEARING = {"SIM", "PSEUDO SIM"}

# Execution sources are multi-label full-text audit findings, not YES/NO proxies.
from execution_levels import BASIS_ORDER, BASIS_COLORS, basis_tags
BASIS_POSITIONS = {"approach": 8, "reference": 17}


def basis_of(value) -> str:
    return " + ".join(basis_tags(value))


def norm(value) -> str:
    value = "" if pd.isna(value) else str(value).strip()
    return value if value else "[MISSING]"


def norm_level(value) -> str:
    value = norm(value).upper()
    return "[MISSING]" if value in {"NAN", "NONE"} else value


def objective_group(value) -> str:
    text = norm(value).lower()
    has_measure = "measure" in text
    has_mitigate = "mitigat" in text
    if has_measure and has_mitigate:
        return "Measure and mitigate"
    if has_measure:
        return "Measure only"
    if has_mitigate:
        return "Mitigate only"
    return "Other/unclear"


def integration_level(values) -> str:
    values = [norm_level(value) for value in values]
    by = dict(zip(COMPONENTS, values))
    inactive = {"NO", "[MISSING]"}
    if all(value in inactive for value in values):
        return "L0 — Not specified"
    if not any(value in REAL_BEARING for value in values):
        return "L2 — Recorded/pseudo-real grounding" if "PSEUDO REAL" in values else "L1 — Virtual-only"
    if all(by[name] in REAL_BEARING for name in ["scenario", "actors", "actuation"]):
        return "L5 — Real-world-dominant benchmark"
    if by["actuation"] in REAL_BEARING or (
        by["ecu_control"] in REAL_BEARING and any(value in SIM_BEARING | {"MIXED"} for value in values)
    ):
        return "L4 — Vehicle/system-level physical integration"
    if any(by[name] in REAL_BEARING for name in ["perception", "localization", "ecu_control"]):
        return "L3 — Component-level physical integration"
    return "L6 — Environment/actor-level real integration"


LEVEL_RANK = {
    "L0 — Not specified": 0,
    "L1 — Virtual-only": 1,
    "L2 — Recorded/pseudo-real grounding": 2,
    "L3 — Component-level physical integration": 3,
    "L4 — Vehicle/system-level physical integration": 4,
    "L5 — Real-world-dominant benchmark": 5,
    "L6 — Environment/actor-level real integration": 6,
}

MISSING_SETUP = {"", "-", "no", "na", "n/a", "not specified", "not mentioned",
                 "no information", "not disclosed", "unclear", "[missing]", "nan", "none"}


def _setup_tags(value, rules, missing_label, fallback):
    """fallback=None means: fall back to the literal cleaned text instead of
    a generic bucket label, so every distinctly named/described resource
    that matches no rule is shown by its own specific name rather than
    disappearing into an uninformative 'Other' count."""
    text = str(value).strip().lower()
    if text in MISSING_SETUP or (missing_label.startswith("No hardware") and text.startswith("no (")):
        return [missing_label]
    tags = [label for pattern, label in rules if re.search(pattern, text, re.I)]
    if tags:
        return list(dict.fromkeys(tags))
    if fallback is None:
        raw = re.sub(r"\s+", " ", str(value).strip())
        return [raw] if raw else [missing_label]
    return [fallback]


# The fallback below is applied to every study whose simulator/dataset field is blank, a
# placeholder, or unclear ("[MISSING]", "Not specified", "Unclear", "NO", ...), so it records the
# absence of a *reported* named platform or dataset, not a verified absence of use.
NO_NAMED_SIMULATOR = "No simulation platform or reference dataset reported"


def simulator_tags(value):
    rules = [
        (r"\bmonti\b", "MonTI (agent-based simulator)"),
        (r"sim.on.wheels", "Sim-on-Wheels (custom physical-world-in-the-loop simulator)"),
        (r"avl driving ?cube", "AVL DrivingCube (VIL chassis-dynamometer testbench)"),
        (r"ground robot simulator", "Custom differential-drive ground-robot simulator"),
        (r"integrated av simulation framework", "Custom multi-model AV simulation framework"),
        (r"sim.to.real rehearsal platform", "Custom RL sim-to-real rehearsal platform (racing testbed)"),
        (r"\bcarla\b", "CARLA"), (r"\bsumo\b", "SUMO"), (r"carmaker", "IPG CarMaker"),
        (r"\b(?:vires )?vtd\b|virtual test drive", "VIRES VTD"), (r"\bunity(?:3d| engine)?\b", "Unity"),
        (r"\bgazebo\b", "Gazebo"), (r"matlab|simulink", "MATLAB/Simulink"),
        (r"\blgsvl\b|\bsvl\b.{0,20}simulator", "SVL/LGSVL"), (r"apollo dreamland|baidu apollo simulator", "Apollo simulator"),
        (r"autodrive", "AutoDRIVE"), (r"beamng", "BeamNG"), (r"prescan", "PreScan"),
        (r"\bvissim\b", "VISSIM"), (r"roadrunner", "RoadRunner"), (r"\besmini\b", "esmini"),
        (r"\bvista(?: 2\.0)?\b", "VISTA"), (r"\bcarsim\b", "CarSim"), (r"\bverosim\b", "VEROSIM"),
        (r"\bmorai\b", "MORAI"), (r"\bnuplan\b", "nuPlan"), (r"\bscenic\b", "Scenic scenario language"),
        (r"\bblender\b|\bcycles\b", "Blender/rendering engine"), (r"\bdspace\b|modeldesk", "dSPACE tooling"),
        (r"applied intuition", "Applied Intuition (commercial simulation platform)"), (r"donkey", "Donkeycar"),
        (r"simcenter", "Siemens Simcenter"), (r"\bfmu\b|functional mock.?up", "FMU co-simulation"),
        (r"lidarsim|custom.{0,25}lidar|lidar.{0,25}custom", "LidarSim"),
        (r"opensceneg", "OpenSceneGraph"), (r"\bavl\b", "AVL simulator"), (r"\bautoware\b", "Autoware simulator"),
        (r"hardware.?in.?the.?loop simulator|\bhils?\b.{0,10}simulator", "Custom HiL simulator"),
        (r"rain simulation|ray tracing|mie scattering|physics.?based.{0,15}weather", "Custom physics-based weather/sensor-effect simulator"),
        (r"dataset|kitti|cityscapes|nuscenes|waymo|ngsim|highd|argoverse|bdd100k|pandaset|ms-?coco\b|naturalistic driving data|benchmark",
         "Dataset/reference data"),
        (r"custom.{0,20}lidar sensor model|not mentioned.custom made", "Custom LiDAR sensor-model simulator"),
        (r"custom|in.house|contribution|novel|d2c2.?auks", "Custom/research simulator"),
    ]
    tags = _setup_tags(value, rules, NO_NAMED_SIMULATOR, None)
    # A more specifically-named simulator/resource makes the generic
    # "Custom/research simulator" bucket redundant for that study; keep only
    # the specific name(s).
    if "Custom/research simulator" in tags and len(tags) > 1:
        tags = [t for t in tags if t != "Custom/research simulator"]
    # The AVL DrivingCube is a specific named AVL product; do not also list
    # the generic "AVL simulator" bucket for the same study.
    if "AVL DrivingCube (VIL chassis-dynamometer testbench)" in tags and "AVL simulator" in tags:
        tags = [t for t in tags if t != "AVL simulator"]
    return tags


# Row-level corrections from reading the primary studies (2026-10-01). The free-text field was
# blank or "Not disclosed" although the paper itself describes the physical setup.

# RQ4.5 row-level corrections from reading the primary studies (2026-10-01); the keyword rules in
# artefact_status() mis-read these free-text values.


def hardware_tags(value):
    text_check = str(value).strip().lower()
    if text_check in MISSING_SETUP or text_check.startswith("no ("):
        return ["No hardware / not specified"]
    rules = [
        (r"\bvehicles?\b|\bfleet\b|ford mondeo|hyundai|kia soul|smart fortwo|lincoln mkz|changan|lexus rx|polaris", "Full-size vehicle"),
        (r"donkey|jetson nano vehicle|jetracer|micro.?car|small.?scale|ground robot|zen-wheels|skid.?steer|differential.?drive", "Small-scale vehicle/robot"),
        (r"camera|dashcam|imx|blackfly|basler|zed 2i", "Camera"), (r"lidar|velodyne|ouster|livox|robosense", "LiDAR"),
        (r"radar|fmcw|ars548", "Radar"), (r"gnss|gps|imu|inertial|rtk|locali[sz]", "Localization/positioning hardware"),
        (r"ecu|microautobox|autobox|xpack|nvidia drive|stm32|embedded|can bus|domain controller", "ECU/embedded controller"),
        (r"gpu|rtx|gtx|tesla v100|a100|a6000|computer|workstation|cpu|xeon|core i[3579]|raspberry pi", "Compute hardware"),
        (r"chassis dynamometer|test bench|testbench|drivingcube|hil platform|hils platform", "Testbench/HIL equipment"),
        (r"projector|projection|screen|radar target stimulator|antenna|brdf|network analyzer|goniometer|reflectometer", "Sensor-stimulation equipment"),
        (r"fog chamber|artificial fog|rain (?:and fog )?platform|weather platform|water film|spray water|rain simulation test facility", "Atmospheric test chamber"),
        (r"vr headset|vive vr|oculus|hololens|motion capture|mocopi|vicon", "VR/MR or motion-capture equipment"),
        (r"roadside|5g base|v2x|c-v2x|infrastructure|surveillance camera network", "Roadside/communication infrastructure"),
        (r"drone|uav|aerial", "Drone/aerial platform"), (r"logitech|steering|pedal|driving rig|driver input", "Human driving controls"),
    ]
    tags = _setup_tags(value, rules, "No hardware / not specified", None)
    # A bare "vehicle" mention is a role word, not proof of a full-size
    # vehicle; do not let it override an explicitly small-scale platform
    # (e.g. "Ego vehicle: Donkeycar") unless a real make/model or an
    # "instrumented"/"proving ground" qualifier confirms an actual full-size
    # car. Absent any of that, a bare "vehicle" defaults to full-size.
    text = str(value).strip().lower()
    generic_role_word_only = bool(re.search(r"\bvehicles?\b", text)) and not re.search(
        r"ford mondeo|hyundai|kia soul|smart fortwo|lincoln mkz|changan|lexus rx|polaris|instrumented|proving ground", text)
    if "Full-size vehicle" in tags and "Small-scale vehicle/robot" in tags and generic_role_word_only:
        tags = [t for t in tags if t != "Full-size vehicle"]
    return tags


SUT_ADS_GATE = r"modular ads|ads stack|autonomous driving system|autoware|apollo|open.?cav|custom adas|\bads\b|full.?stack"
SUT_NO_VALUES = {"no", "no.", "none"}
SUT_UNSPECIFIED_PATTERN = r"not (?:further |explicitly )?specified|non ?specified|no specifics|unclear|unspecified|not (?:further )?defined|not disclosed"

# Manually reviewed reclassifications for SUT descriptions whose function is
# clear from the paper's title/detailed WHAT-HOW text but was not recoverable
# from the SUT field's wording by keyword rules alone (e.g. named models such
# as "DeepDriving"/"TransFuser, LAV", or a bare "ADS" whose surrounding paper
# context reveals a more specific role). Keyed by the visible source CSV row
# so they remain auditable and stable across runs, matching WHAT_OVERRIDES in
# fill_detailed_what_how.py.
# "No SUT (explicitly none)" and the former "No downstream SUT" label both
# meant the same thing to a reader -- there is no separately evaluated
# downstream system, whether because the field literally said "no" or
# because the field named the simulator/sensor-renderer itself (the paper's
# real subject is that simulator's fidelity, not a system running on top of
# it) -- so both now share one label.
NO_DOWNSTREAM_SUT = "No SUT (none, or simulator/testbed fidelity only)"

SUT_ROW_OVERRIDES = {
    5: ["Multi-sensor fusion perception/tracking"],
    8: [NO_DOWNSTREAM_SUT],
    9: [NO_DOWNSTREAM_SUT],
    15: ["Radar perception"],
    17: [NO_DOWNSTREAM_SUT],
    19: ["Planning/trajectory prediction"],
    22: ["Camera perception"],
    28: [NO_DOWNSTREAM_SUT],
    29: ["Planning/trajectory prediction"],
    112: ["Full ADS"],
    162: ["Planning/trajectory prediction", "Ego-vehicle control/actuation"],
    38: [NO_DOWNSTREAM_SUT],
    40: [NO_DOWNSTREAM_SUT],
    41: [NO_DOWNSTREAM_SUT],
    44: [NO_DOWNSTREAM_SUT],
    46: ["Ego-vehicle control/actuation"],
    48: [NO_DOWNSTREAM_SUT],
    49: [NO_DOWNSTREAM_SUT],
    50: ["Planning/trajectory prediction"],
    54: ["Camera perception"],
    55: ["Full ADS"],
    56: ["Full ADS"],
    57: [NO_DOWNSTREAM_SUT],
    58: [NO_DOWNSTREAM_SUT],
    59: [NO_DOWNSTREAM_SUT],
    60: [NO_DOWNSTREAM_SUT],
    61: [NO_DOWNSTREAM_SUT],
    62: [NO_DOWNSTREAM_SUT],
    63: [NO_DOWNSTREAM_SUT],
    64: [NO_DOWNSTREAM_SUT],
    66: [NO_DOWNSTREAM_SUT],
    67: ["Camera perception"],
    68: [NO_DOWNSTREAM_SUT],
    69: [NO_DOWNSTREAM_SUT],
    70: [NO_DOWNSTREAM_SUT],
    71: [NO_DOWNSTREAM_SUT],
    72: [NO_DOWNSTREAM_SUT],
    73: [NO_DOWNSTREAM_SUT],
    74: [NO_DOWNSTREAM_SUT],
    75: ["Radar perception"],
    76: [NO_DOWNSTREAM_SUT],
    77: [NO_DOWNSTREAM_SUT],
    78: [NO_DOWNSTREAM_SUT],
    79: [NO_DOWNSTREAM_SUT],
    81: [NO_DOWNSTREAM_SUT],
    82: [NO_DOWNSTREAM_SUT],
    83: [NO_DOWNSTREAM_SUT],
    84: ["LLM/foundation-model reasoning agent"],
    85: [NO_DOWNSTREAM_SUT],
    92: [NO_DOWNSTREAM_SUT],
    103: ["Full ADS"],
    108: [NO_DOWNSTREAM_SUT],
    119: ["Ego-vehicle control/actuation"],
    120: ["Camera perception"],
    125: ["Ego-vehicle control/actuation"],
    126: [NO_DOWNSTREAM_SUT],
    128: [NO_DOWNSTREAM_SUT],
    131: ["Full ADS"],
    133: [NO_DOWNSTREAM_SUT],
    135: ["Full ADS"],
    139: [NO_DOWNSTREAM_SUT],
    140: [NO_DOWNSTREAM_SUT],
    141: [NO_DOWNSTREAM_SUT],
    142: [NO_DOWNSTREAM_SUT],
    144: ["Full ADS"],
    145: [NO_DOWNSTREAM_SUT],
    148: ["LiDAR perception"],
    150: ["Full ADS", "Ego-vehicle control/actuation"],
    151: [NO_DOWNSTREAM_SUT],
    152: ["Camera perception"],
    153: ["Localization"],
    158: [NO_DOWNSTREAM_SUT],
    159: [NO_DOWNSTREAM_SUT],
    160: ["LLM/foundation-model reasoning agent"],
    163: [NO_DOWNSTREAM_SUT],
    164: ["Full ADS"],
    165: [NO_DOWNSTREAM_SUT],
    166: [NO_DOWNSTREAM_SUT],
    167: [NO_DOWNSTREAM_SUT],
    170: ["Camera perception"],
    171: [NO_DOWNSTREAM_SUT],
    172: [NO_DOWNSTREAM_SUT],
    173: [NO_DOWNSTREAM_SUT],
    161: [NO_DOWNSTREAM_SUT],
    176: ["LiDAR perception"],
    177: [NO_DOWNSTREAM_SUT],
    178: [NO_DOWNSTREAM_SUT],
    179: [NO_DOWNSTREAM_SUT],
    180: [NO_DOWNSTREAM_SUT],
    182: [NO_DOWNSTREAM_SUT],
    183: [NO_DOWNSTREAM_SUT],
    184: [NO_DOWNSTREAM_SUT],
    185: [NO_DOWNSTREAM_SUT],
    186: ["Radar perception"],
    181: ["Camera perception"],
    188: ["Full ADS"],
    189: ["Ego-vehicle control/actuation"],
    190: ["Full ADS"],
    198: [NO_DOWNSTREAM_SUT],
    199: ["Full ADS"],
    196: [NO_DOWNSTREAM_SUT],
    197: [NO_DOWNSTREAM_SUT],

    # RQ4.3 SUT-classification audit (2026-09-29): these rows' free-text SUT
    # field named a specific system/framework in language the automatic
    # rules did not recognise, so each fell through as its own literal-text
    # pseudo-category instead of one of the ten established SUT categories.
    200: [NO_DOWNSTREAM_SUT],
    # SUT field literally says "NA (image-level perception fidelity metric,
    # not a running SUT)" -- the regex matched the word "image" and filed it
    # under Camera perception despite the field explicitly denying it is a SUT.
    203: [NO_DOWNSTREAM_SUT],
    # "Benchmark autonomous driving algorithm (simulation-only)" evaluates
    # scenario-library quality, not a specific vehicle-side driving function.
    205: [NO_DOWNSTREAM_SUT],
    # "Traffic/NPC behaviour model" is an environment/simulation component,
    # not an ego-vehicle or ADS-side function.
    206: [NO_DOWNSTREAM_SUT],
    # The "Hybrid-reality AD test system" is the paper's own testbed
    # infrastructure, evaluated for its own technical performance, not a
    # downstream driving function riding on it.
    207: [NO_DOWNSTREAM_SUT],
    # "Human pedestrian behaviour" is a human-subject study; no vehicle or
    # ADS-side function is exercised.
    210: ["Full ADS"],
    # "Full autonomous driving stack (navigation)" is an explicit full-ADS
    # description.
    211: ["Planning/trajectory prediction"],
    # "AV decision-making in traffic interactions" is a decision/planning
    # function.
    212: [NO_DOWNSTREAM_SUT],
    # The "Scenario-in-the-Loop control model" is the co-simulation test
    # infrastructure itself, not a vehicle-side driving function.
    213: ["Full ADS"],
    # "Automated driving function (vehicle driving function)" names a
    # generic complete driving function with no lower-level control detail.
    215: ["Camera perception"],
    # "Object detection model (Faster RCNN) for evaluation" is a camera-based
    # detector.
    216: [NO_DOWNSTREAM_SUT],
    # The "AV test scenario generator" is a scenario-generation tool, not a
    # vehicle-side driving function.
    217: [NO_DOWNSTREAM_SUT],
    # The "HGSim scenario/scene generation framework" is a scene-generation
    # tool, not a vehicle-side driving function.
    218: [NO_DOWNSTREAM_SUT],
    # The Hu&ViL "test method/vehicle" entry names the test method being
    # compared across fidelity levels, not a specific driving function under
    # test.
    220: [NO_DOWNSTREAM_SUT],
    # "Background/NPC vehicle behaviour model" is an environment/simulation
    # component, not an ego-vehicle or ADS-side function.
    221: ["LiDAR perception"],
    # "Data-driven ADAS sensor model" is confirmed (full-text) to be a LiDAR
    # sensor model specifically.
    223: ["Ego-vehicle control/actuation"],
    # "Automated driving functions (e.g., emergency braking, platooning)"
    # names AEB explicitly, an existing Ego-vehicle control/actuation
    # subcategory member.
    224: ["Camera perception"],
    # "AI-based pedestrian detector" is a camera-based detection model.
    228: ["LLM/foundation-model reasoning agent"],
    # "VLM-based scene understanding + decision-making stack" is explicitly
    # a vision-language (foundation) model reasoning agent.
    229: ["Full ADS"],
    # NAVSIM benchmarks complete end-to-end driving architectures (TransFuser,
    # Latent TransFuser, UniAD, PARA-Drive) rather than one named SUT; the raw
    # SUT text is a list of these architectures, which the generic sut_tags()
    # keyword rules do not match, so it falls back to the unclassified raw
    # string instead of Full ADS.
}


def _sut_ads_architecture(text):
    if re.search(r"end.?to.?end|\be2e\b", text):
        return "Full ADS"
    if re.search(r"modular|autoware|apollo|open.?cav|\bstack\b|(decision|planning).{0,25}control|control.{0,25}(decision|planning)", text):
        return "Full ADS"
    return None


def sut_tags(value, source_csv_row=None):
    if source_csv_row in SUT_ROW_OVERRIDES:
        return SUT_ROW_OVERRIDES[source_csv_row]
    text = str(value).strip().lower()
    if text in SUT_NO_VALUES:
        return [NO_DOWNSTREAM_SUT]
    if text in MISSING_SETUP:
        return ["SUT function not recorded (field left blank)"]
    rules = [
        (r"camera|image|vision|yolo|segment|depth estimation|faster r-cnn|deeplab|lane detection", "Camera perception"),
        (r"lidar|pointpillars|pv.?rcnn|voxelnet|pointnet|rangenet|polarnet", "LiDAR perception"), (r"radar", "Radar perception"),
        (r"sensor fusion|multimodal|multi.?sensor|bevfusion", "Multi-sensor fusion perception/tracking"),
        (r"locali[sz]|slam|gnss|gps.imu|pose synchronization|ndt mapping", "Localization"),
        (r"planner|planning|trajectory prediction|motion forecasting|path planning", "Planning/trajectory prediction"),
        (r"acc\b|adaptive cruise|aeb|lane keeping|lane change|controller|control stack|vehicle control|nmpc|steering|actuation|waypoint follower|\becu\b", "Ego-vehicle control/actuation"),
        (r"\bgpt-?\d|llama|mistral|claude|gemini|large language model|\bllm\b", "LLM/foundation-model reasoning agent"),
    ]
    tags = []
    if re.search(SUT_ADS_GATE, text):
        architecture = _sut_ads_architecture(text)
        if architecture:
            tags.append(architecture)
    tags += [label for pattern, label in rules if re.search(pattern, text, re.I)]
    tags = list(dict.fromkeys(tags))
    if tags:
        return tags
    if re.search(SUT_UNSPECIFIED_PATTERN, text):
        return ["Unspecified SUT function (described, but no function named)"]
    raw = re.sub(r"\s+", " ", str(value).strip())
    return [raw] if raw else ["SUT function not recorded (field left blank)"]


BIB_PATH = DATA_DIR / "primary_studies.bib"


def _norm_title(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def bib_key_to_title(bib_path: Path = BIB_PATH) -> dict[str, str]:
    """Map every bib key to its title: the '% Title' comment above each @entry,
    or, where an entry has no such comment, the entry's own `title = {...}` field."""
    text = bib_path.read_text(encoding="utf-8")
    mapping = {
        key.strip(): title.strip()
        for title, key in re.findall(r"^%\s*(.+?)\s*\n@\w+\{\s*([^,\s]+)\s*,", text, re.MULTILINE)
    }
    for block in re.split(r"\n(?=@\w+\{)", text):
        head = re.match(r"@\w+\{\s*([^,\s]+)\s*,", block)
        field = re.search(r"\n\s*title\s*=\s*\{", block)
        if not head or not field or head.group(1) in mapping:
            continue
        depth, end = 1, field.end()
        while end < len(block) and depth:
            depth += {"{": 1, "}": -1}.get(block[end], 0)
            end += 1
        mapping[head.group(1)] = re.sub(r"[{}]", "", block[field.end():end - 1])
    return mapping


def row_to_bib_key(studies: pd.DataFrame, bib_path: Path = BIB_PATH) -> tuple[dict[int, str], list[int], list[str]]:
    """Match every study's source_csv_row to its bib citation key by
    normalized title, so tables can be built from (row, field) data without
    ever hand-typing a \\cite key. Returns (row_to_key, unmatched_rows,
    unused_bib_keys) -- callers must surface the last two rather than
    silently dropping rows, per the audit finding that hand-picked
    citations can silently drift from the data they are meant to cite."""
    key_to_title = bib_key_to_title(bib_path)
    title_to_key: dict[str, str] = {}
    duplicate_titles = []
    for key, title in key_to_title.items():
        norm_t = _norm_title(title)
        if norm_t in title_to_key and title_to_key[norm_t] != key:
            duplicate_titles.append((title, title_to_key[norm_t], key))
        title_to_key[norm_t] = key

    row_to_key: dict[int, str] = {}
    unmatched_rows: list[int] = []
    for _, r in studies[["source_csv_row", "paper_title"]].drop_duplicates("source_csv_row").iterrows():
        k = title_to_key.get(_norm_title(r.paper_title))
        if k:
            row_to_key[int(r.source_csv_row)] = k
        else:
            unmatched_rows.append(int(r.source_csv_row))
    used_keys = set(row_to_key.values())
    unused_bib_keys = sorted(set(key_to_title) - used_keys)
    if duplicate_titles:
        print("WARNING duplicate bib titles:", duplicate_titles)
    return row_to_key, unmatched_rows, unused_bib_keys


# Final categories of the free-text fields, written into the table (columns of data/primary_studies_coding.csv).
FROZEN_COLUMNS = {
    "simulator_tags": "category_simulator", "dataset_tags": "category_dataset", "hardware_tags": "category_hardware",
    "sut_tags": "category_sut", "validation_status": "category_validation_status",
    "validation_mode": "category_validation_technique", "metric_status": "category_metric_status",
    "metric_categories": "category_metrics", "test_generation_role": "category_test_generation",
    "artefact_status": "category_artefacts", "metric_subcategories": "subcategory_metrics",
    "sut_subcategories": "subcategory_sut", "what_tags": "category_what", "how_tags": "category_how",
}


def subcategory_of(cell: str, category: str) -> str:
    """Sub-categories of one category from a cell written as 'Category::sub | sub ;; Category::sub'."""
    return dict(part.split("::", 1) for part in cell.split(" ;; "))[category]


def validation_technique(frame: pd.DataFrame) -> pd.Series:
    """Final evaluation-technique label of each study in `frame` (column category_validation_technique)."""
    table = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False,
                        usecols=["study_id", "category_validation_technique"]).set_index("study_id")["category_validation_technique"]
    return frame["source_csv_row"].astype(int).astype(str).map(table)


CODED_COLUMNS = [
    "paper_title", "link", "notes", "objective", "gap_direction", "mitigated_measured_goal_and_setup",
    "simplified_what", "simplified_how", "approach_execution_basis",
    "approach_level_scenario", "approach_level_actors", "approach_level_perception",
    "approach_level_localization", "approach_level_actuation", "approach_level_ecu_control",
    "validation_procedure", "gap_type", "reference_execution_basis",
    "reference_level_scenario", "reference_level_actors", "reference_level_perception",
    "reference_level_localization", "reference_level_actuation", "reference_level_ecu_control",
    "metrics", "simulator_or_dataset", "hardware_setup", "sut", "data_creation_usage", "available_artefacts",
]


def load_mapping() -> tuple[pd.DataFrame, pd.DataFrame]:
    # The two execution-basis columns hold the literal source 'None', so they are read without NA conversion.
    raw = pd.read_csv(CSV_PATH, dtype=object, keep_default_na=True)
    basis_columns = ["approach_execution_basis", "reference_execution_basis"]
    raw[basis_columns] = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False, usecols=basis_columns)[basis_columns]
    df = raw[CODED_COLUMNS].copy().reset_index(drop=True)
    study_ids = raw["study_id"].astype(int).reset_index(drop=True)
    frozen = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False, usecols=list(FROZEN_COLUMNS.values()))
    what_tags = frozen["category_what"].map(lambda value: value.split(" | "))
    how_tags = frozen["category_how"].map(lambda value: value.split(" | "))
    what = pd.DataFrame({
        "what_tags": what_tags.map(lambda tags: " | ".join(tags)),
        "number_of_what_tags": what_tags.map(len),
    })
    how = pd.DataFrame({
        "how_tags": how_tags.map(lambda tags: " | ".join(tags)),
        "number_of_how_tags": how_tags.map(len),
    })
    assert len(df) == len(what) == len(how) and study_ids.is_unique
    studies = pd.DataFrame({
        "source_csv_row": study_ids.values,
        "paper_title": df.iloc[:, 0].map(norm),
        "objective_raw": df.iloc[:, 3].map(norm),
        "direction": df.iloc[:, 4].map(norm),
        "detailed_what_how": df.iloc[:, 5].map(norm),
        "simplified_what": df.iloc[:, 6].map(norm),
        "simplified_how": df.iloc[:, 7].map(norm),
        "validation": df.iloc[:, 15].map(norm),
        "gap_type": df.iloc[:, 16].map(norm),
        "metrics": df.iloc[:, 24].map(norm),
        "simulator_dataset": df.iloc[:, 25].map(norm),
        "hardware": df.iloc[:, 26].map(norm),
        "sut": df.iloc[:, 27].map(norm),
        "data_usage": df.iloc[:, 28].map(norm),
        "artefacts": df.iloc[:, 29].map(norm),
        "approach_basis": df.iloc[:, BASIS_POSITIONS["approach"]].map(basis_of),
        "reference_basis": df.iloc[:, BASIS_POSITIONS["reference"]].map(basis_of),
        "what_tags": what["what_tags"].map(norm),
        "how_tags": how["how_tags"].map(norm),
        "number_of_what_tags": pd.to_numeric(what["number_of_what_tags"]),
        "number_of_how_tags": pd.to_numeric(how["number_of_how_tags"]),
    })
    studies["objective_group"] = studies["objective_raw"].map(objective_group)
    approach = df.iloc[:, APPROACH_POSITIONS].apply(lambda col: col.map(norm_level))
    reference = df.iloc[:, REFERENCE_POSITIONS].apply(lambda col: col.map(norm_level))
    approach.columns = [f"approach_{name}" for name in COMPONENTS]
    reference.columns = [f"reference_{name}" for name in COMPONENTS]
    studies = pd.concat([studies, approach, reference], axis=1)
    studies["approach_level"] = approach.apply(lambda row: integration_level(row.tolist()), axis=1)
    studies["reference_level"] = reference.apply(lambda row: integration_level(row.tolist()), axis=1)
    studies["approach_rank"] = studies["approach_level"].map(LEVEL_RANK)
    studies["reference_rank"] = studies["reference_level"].map(LEVEL_RANK)
    for name, column in FROZEN_COLUMNS.items():
        if name not in studies.columns:
            studies[name] = frozen[column].values
    return df, studies












# One fixed order for the " + "-joined technique labels. The automatic classifier and the manual
# overrides joined the same techniques in different orders ("A + B" vs "B + A"), which split one
# combination into two rows of the exact-combination ranking.








