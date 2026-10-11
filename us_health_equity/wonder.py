"""Client for the CDC WONDER Underlying Cause of Death API.

The API only serves national-level vital statistics (no state grouping), so it is
used for the national mortality-by-race series. Two databases are stitched together:

* D76  - Underlying Cause of Death 1999-2020, bridged race (used for 2015-2017)
* D158 - Underlying Cause of Death 2018-latest, single race (used for 2018+)

CDC requires at least 15 seconds between API requests; responses are cached on disk.
"""

from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET

from loguru import logger
import pandas as pd
import requests

from us_health_equity.config import WONDER_URL

# Variable codes differ by database only for race.
RACE_VAR = {"D76": "V8", "D158": "V42"}
YEAR, HISPANIC, CAUSE_113 = "V1-level1", "V17", "V4"

REQUEST_GAP_SECONDS = 16
_last_request = 0.0

VALUE_COLUMNS = ["deaths", "population", "crude_rate", "crude_rate_se", "aadr", "aadr_se"]


def _build_params(db: str, group_by: list[str], years: list[int]) -> dict:
    """Assemble the WONDER request parameters for an age-adjusted rate query."""
    race = RACE_VAR[db]
    params = {
        **{f"B_{i + 1}": f"{db}.{g}" for i, g in enumerate(group_by)},
        **{f"B_{i + 1}": "*None*" for i in range(len(group_by), 5)},
        "M_1": f"{db}.M1",  # deaths
        "M_2": f"{db}.M2",  # population
        "M_3": f"{db}.M3",  # crude rate
        "M_31": f"{db}.M31",  # crude rate standard error
        "M_32": f"{db}.M32",  # crude rate CI
        "M_41": f"{db}.M41",  # age-adjusted rate
        "M_42": f"{db}.M42",  # age-adjusted rate standard error
        "O_aar": "aar_std",
        "O_aar_pop": "0000",
        "O_age": f"{db}.V5",
        "O_javascript": "on",
        "O_location": f"{db}.V9",
        "O_precision": "1",
        "O_rate_per": "100000",
        "O_show_totals": "false",
        "O_timeout": "600",
        "O_title": "",
        "O_ucd": f"{db}.V2",
        "O_urban": f"{db}.V19",
        "action-Send": "Send",
        "stage": "request",
    }
    if db != "D76":
        params["O_race"] = f"{db}.{race}"
    for var in ["V1", "V10", "V2", "V27", "V9"]:
        params[f"O_{var}_fmode"] = "freg"
        params[f"finder-stage-{db}.{var}"] = "codeset"
        params[f"F_{db}.{var}"] = [str(y) for y in years] if var == "V1" else "*All*"
        params[f"I_{db}.{var}"] = "*All*"
        params[f"V_{db}.{var}"] = ""
    for var in ["V4", "V5", "V7", "V17", "V19", race]:
        params[f"V_{db}.{var}"] = "*All*"
    params[f"V_{db}.V6"] = "00"
    return params


def _to_xml(params: dict) -> str:
    parts = ["<request-parameters>"]
    for name, value in params.items():
        values = value if isinstance(value, list) else [value]
        parts.append(
            f"<parameter><name>{name}</name>"
            + "".join(f"<value>{v}</value>" for v in values)
            + "</parameter>"
        )
    parts.append("</request-parameters>")
    return "".join(parts)


def fetch(db: str, group_by: list[str], years: list[int], cache_path: Path) -> str:
    """POST a query to WONDER (or reuse the cached response) and return the XML text."""
    global _last_request
    if cache_path.exists():
        return cache_path.read_text()

    wait = REQUEST_GAP_SECONDS - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    logger.info(f"WONDER {db} query grouped by {group_by} for {years[0]}-{years[-1]}")
    resp = requests.post(
        WONDER_URL.format(db=db),
        data={
            "request_xml": _to_xml(_build_params(db, group_by, years)),
            "accept_datause_restrictions": "true",
        },
        timeout=900,
    )
    _last_request = time.time()
    if resp.status_code != 200 or "<data-table" not in resp.text:
        messages = re.findall(r"<message>(.*?)</message>", resp.text, flags=re.DOTALL)
        raise RuntimeError(f"WONDER {db} request failed ({resp.status_code}): {messages}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(resp.text)
    return resp.text


def _number(text: str | None) -> float:
    if text is None:
        return float("nan")
    try:
        return float(text.replace(",", ""))
    except ValueError:  # "Suppressed", "Unreliable", "Not Applicable"
        return float("nan")


def parse(xml_text: str, label_columns: list[str]) -> pd.DataFrame:
    """Parse a WONDER <data-table> response into a tidy DataFrame."""
    table = ET.fromstring(
        xml_text[xml_text.find("<data-table") :].split("</data-table>")[0] + "</data-table>"
    )
    records = []
    # A label cell with r="n" spans n rows; the following rows omit that cell.
    carried: list[str | None] = [None] * len(label_columns)
    span_left = [0] * len(label_columns)
    for row in table.iter("r"):
        cells = list(row)
        labels = []
        for i in range(len(label_columns)):
            if span_left[i] > 0:
                span_left[i] -= 1
            else:
                cell = cells.pop(0)
                carried[i] = cell.get("l")
                span_left[i] = int(cell.get("r", 1)) - 1
            labels.append(carried[i])
        values = cells
        rec = dict(zip(label_columns, labels))
        for col, cell in zip(VALUE_COLUMNS, values):
            rec[col] = _number(cell.get("v"))
            if col == "aadr":
                rec["aadr_flag"] = cell.get("v") if rec[col] != rec[col] else ""
                ci = cell.find("l")
                lo, hi = (float("nan"), float("nan"))
                if ci is not None and "-" in (ci.get("v") or ""):
                    lo, hi = (_number(x.strip(" ()")) for x in ci.get("v").split("-"))
                rec["aadr_lci"], rec["aadr_uci"] = lo, hi
        records.append(rec)
    df = pd.DataFrame.from_records(records)
    if "year" in df:
        df["year"] = df["year"].astype(int)
    return df
