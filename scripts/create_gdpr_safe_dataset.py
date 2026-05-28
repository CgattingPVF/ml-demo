from __future__ import annotations

import json
import re
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from app.data_loader import load_dataset, normalize_columns

DEFAULT_SOURCE_PATH = BASE_DIR / "projects_export.csv"
XLSX_OUTPUT_PATH = BASE_DIR / "data" / "projects_export_gdpr_safe.xlsx"
MANIFEST_PATH = BASE_DIR / "data" / "projects_export_gdpr_safe_manifest.json"

TARGET_ROWS = 12000
TOP_TEAMS = 250
MAX_ROWS_PER_TEAM = 60
RANDOM_STATE = 42

KEEP_COLUMNS = [
    "project_no",
    "project_name",
    "contact_name",
    "date",
    "due_date",
    "completed_date",
    "status",
    "estimated_duration",
    "total_to_do",
    "total_done",
    "total_to_do_billable",
    "done_billable",
    "budget_revenue",
    "actual_revenue",
    "budget_cost",
    "actual_cost",
    "budget_labor_cost",
    "actual_labor_cost",
    "related_contacts",
    "c_homeownerpostcode",
    "c_creationdate",
    "c_erectdate",
    "c_installdate",
    "c_dismantledate",
    "c_locationofscaffold",
    "c_scotlandonlyloadnumeric",
    "c_scotlandonlyliftsnumeric",
    "c_scaffoldpurpose",
    "c_frontscaffoldheight",
    "c_frontscaffoldlength",
    "c_rearscaffoldheight",
    "c_rearscaffoldlength",
    "c_leftscaffoldheight",
    "c_leftscaffoldlength",
    "c_rightscaffoldheight",
    "c_rightscaffoldlength",
    "c_edgeprotection",
    "c_hirecharges",
    "c_hsincidentdate",
    "c_hscategory",
    "c_riddorreportable",
    "c_damagereporting",
    "c_damagestatus",
    "c_damageliability",
    "c_damagecategory",
    "c_damageresolutiondate",
    "c_damagecauseidentified",
]

REQUIRED_COLUMNS = [
    "related_contacts",
    "c_homeownerpostcode",
    "c_scaffoldpurpose",
    "status",
    "date",
    "due_date",
    "budget_revenue",
]

DATE_COLUMNS = [
    "date",
    "due_date",
    "completed_date",
    "c_creationdate",
    "c_erectdate",
    "c_installdate",
    "c_dismantledate",
    "c_hsincidentdate",
    "c_damageresolutiondate",
]

NUMERIC_COLUMNS = [
    "estimated_duration",
    "total_to_do",
    "total_done",
    "total_to_do_billable",
    "done_billable",
    "budget_revenue",
    "actual_revenue",
    "budget_cost",
    "actual_cost",
    "budget_labor_cost",
    "actual_labor_cost",
    "c_scotlandonlyloadnumeric",
    "c_scotlandonlyliftsnumeric",
    "c_frontscaffoldheight",
    "c_frontscaffoldlength",
    "c_rearscaffoldheight",
    "c_rearscaffoldlength",
    "c_leftscaffoldheight",
    "c_leftscaffoldlength",
    "c_rightscaffoldheight",
    "c_rightscaffoldlength",
    "c_hirecharges",
    "c_riddorreportable",
]

DROPPED_GDPR_COLUMNS = [
    "id",
    "entity_id",
    "project_manager",
    "project_members",
    "description",
    "tags",
    "c_zendeskid",
    "c_clientreference",
    "c_shroudingref",
    "c_propertyownerinformation",
    "c_homeownername",
    "c_homeowneraddress",
    "c_w3w",
    "c_homepropertyphone",
    "c_homepropertyemail",
    "c_projectdatestimes",
    "c_scaffoldspecifics",
    "c_scaffoldspecification",
    "c_inspectedby",
    "c_siteaccess",
    "c_msadditionalcomments",
    "c_siterequirementsquote",
    "c_photo1",
    "c_photo2",
    "c_elevation",
    "c_siteaccessimage",
    "c_hscategorynotes",
    "c_damageticketid",
    "c_damagepartnerid",
    "c_damagedescription",
    "c_damagesreviewsummary",
    "c_damagefindings",
]


def text_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def non_empty_mask(series: pd.Series) -> pd.Series:
    return text_series(series).ne("")


def outward_postcode(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().upper())
    if not text:
        return ""
    if " " in text:
        return text.split(" ", 1)[0]
    match = re.match(r"^[A-Z]{1,2}\d[A-Z\d]?", text)
    return match.group(0) if match else text[:4]


def stable_aliases(values: pd.Series, prefix: str) -> dict[str, str]:
    counts = text_series(values)
    ordered = counts[counts.ne("")].value_counts().index.tolist()
    return {value: f"{prefix} {index:04d}" for index, value in enumerate(ordered, start=1)}


def clean_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.fillna("")
        .astype(str)
        .str.replace(r"[£$,]", "", regex=True)
        .str.replace(r"\(([^)]+)\)", r"-\1", regex=True)
        .str.strip()
    )
    numeric = pd.to_numeric(cleaned, errors="coerce").round(2)
    return numeric.mask(numeric.abs() > 1_000_000)


def clean_dates(frame: pd.DataFrame) -> None:
    for column in DATE_COLUMNS:
        if column not in frame.columns:
            continue
        parsed = pd.to_datetime(frame[column], errors="coerce", dayfirst=True, format="mixed")
        frame[column] = parsed.dt.strftime("%Y-%m-%d").fillna("")


def status_group(value: Any) -> str:
    text = str(value or "").lower()
    if any(token in text for token in ("cancel", "postponed")):
        return "cancelled"
    if any(token in text for token in ("down", "completed", "complete", "closed", "handed over")):
        return "completed"
    if any(token in text for token in ("booked", "planned", "awaiting", "rts", "request", "sent")):
        return "active"
    return "other"


def select_rows(df: pd.DataFrame) -> pd.DataFrame:
    present_required = [column for column in REQUIRED_COLUMNS if column in df.columns]
    eligible = df.copy()
    for column in present_required:
        eligible = eligible[non_empty_mask(eligible[column])]

    team_counts = text_series(eligible["related_contacts"]).value_counts()
    top_teams = team_counts.head(TOP_TEAMS).index
    eligible = eligible[text_series(eligible["related_contacts"]).isin(top_teams)].copy()

    sampled = (
        eligible.groupby(text_series(eligible["related_contacts"]), group_keys=False)
        .apply(lambda group: group.sample(min(len(group), MAX_ROWS_PER_TEAM), random_state=RANDOM_STATE))
        .copy()
    )

    status_text = text_series(eligible.get("status", pd.Series(dtype=str))).str.lower()
    priority_mask = (
        status_text.str.contains("cancel|postponed|down|completed|complete|closed", regex=True)
        | non_empty_mask(eligible.get("c_hscategory", pd.Series(index=eligible.index, dtype=str)))
        | non_empty_mask(eligible.get("c_hsincidentdate", pd.Series(index=eligible.index, dtype=str)))
        | non_empty_mask(eligible.get("c_damagestatus", pd.Series(index=eligible.index, dtype=str)))
        | non_empty_mask(eligible.get("c_damagecategory", pd.Series(index=eligible.index, dtype=str)))
    )
    priority = eligible[priority_mask].sample(
        min(int(priority_mask.sum()), TARGET_ROWS // 3),
        random_state=RANDOM_STATE,
    )

    combined = pd.concat([sampled, priority]).drop_duplicates()
    if len(combined) > TARGET_ROWS:
        priority_index = priority.index.intersection(combined.index)
        remaining_slots = max(TARGET_ROWS - len(priority_index), 0)
        remainder = combined.drop(index=priority_index, errors="ignore")
        remainder = remainder.sample(min(len(remainder), remaining_slots), random_state=RANDOM_STATE)
        combined = pd.concat([combined.loc[priority_index], remainder]).drop_duplicates()

    return combined.sort_values(["related_contacts", "date"], kind="mergesort").reset_index(drop=True)


def build_safe_dataset(source_path: Path = DEFAULT_SOURCE_PATH) -> dict[str, Any]:
    source_path = source_path.expanduser().resolve()
    source = normalize_columns(load_dataset(source_path))
    selected = select_rows(source)

    keep_columns = [column for column in KEEP_COLUMNS if column in selected.columns]
    safe = selected[keep_columns].copy()

    client_aliases = stable_aliases(safe["contact_name"], "Client") if "contact_name" in safe.columns else {}

    safe["related_contacts"] = text_series(safe["related_contacts"])
    if "contact_name" in safe.columns:
        safe["contact_name"] = text_series(safe["contact_name"]).map(client_aliases).fillna("Client Unknown")
    if "status" in safe.columns:
        insert_at = safe.columns.get_loc("status") + 1
        safe.insert(insert_at, "status_group", safe["status"].map(status_group))
    if "project_no" in safe.columns:
        safe["project_no"] = [f"JOB-{index:06d}" for index in range(1, len(safe) + 1)]
    if "project_name" in safe.columns:
        safe["project_name"] = [f"Project {index:06d}" for index in range(1, len(safe) + 1)]
    if "c_homeownerpostcode" in safe.columns:
        safe["c_homeownerpostcode"] = safe["c_homeownerpostcode"].map(outward_postcode)

    clean_dates(safe)
    for column in NUMERIC_COLUMNS:
        if column in safe.columns:
            safe[column] = clean_numeric(safe[column])

    XLSX_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    safe.to_excel(XLSX_OUTPUT_PATH, index=False, sheet_name="GDPR Safe Data")

    manifest = {
        "source_file": source_path.name,
        "raw_source_retained": False,
        "excel_output_file": str(XLSX_OUTPUT_PATH.relative_to(BASE_DIR)),
        "source_rows": int(source.shape[0]),
        "source_columns": int(source.shape[1]),
        "output_rows": int(safe.shape[0]),
        "output_columns": int(safe.shape[1]),
        "team_names_preserved": True,
        "client_aliases": int(len(client_aliases)),
        "postcodes": "reduced to outward postcode areas only",
        "row_policy": {
            "top_teams": TOP_TEAMS,
            "max_rows_per_team": MAX_ROWS_PER_TEAM,
            "target_rows": TARGET_ROWS,
            "priority_rows_preserved": "cancelled/completed, HSE, and damage signal rows",
        },
        "dropped_gdpr_columns": [column for column in DROPPED_GDPR_COLUMNS if column in source.columns],
        "kept_columns": keep_columns,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = ArgumentParser(description="Build the GDPR-safe PVF training workbook from a raw export.")
    parser.add_argument(
        "source",
        nargs="?",
        default=str(DEFAULT_SOURCE_PATH),
        help="Path to the raw PVF export. Defaults to ./projects_export.csv when present.",
    )
    args = parser.parse_args()
    print(json.dumps(build_safe_dataset(Path(args.source)), indent=2))
