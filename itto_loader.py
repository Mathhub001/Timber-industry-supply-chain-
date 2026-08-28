"""
Loads and reshapes the manually-exported ITTO statistics file.

ITTO has no public API (see README.md Step 3 for the manual export procedure
at https://www.itto.int/biennal_review_statistics/?mode=searchdata).

This script is defensive about column names because ITTO's export format has
varied slightly across years/exports — inspect data/raw/itto_export_raw.csv
yourself first and adjust COLUMN_MAP below if your export's headers differ.

Run: python itto_loader.py
Output: data/raw/itto_clean.csv  (long format: country, year, product, flow, value)
"""

import os
import pandas as pd

from config import RAW_DIR

RAW_FILE = os.path.join(RAW_DIR, "itto_export_raw.csv")
OUT_FILE = os.path.join(RAW_DIR, "itto_clean.csv")

# Adjust this if your exported column headers differ from ITTO's default export.
# Left side = what to look for (case-insensitive substring match),
# right side = standardised name used in the rest of the pipeline.
COLUMN_MAP_HINTS = {
    "country": "country",
    "product": "product",
    "flow": "flow",       # e.g. Export / Production / Import
    "year": "year",
    "value": "value",
}

# Map ITTO's country name strings to ISO3 — extend as needed based on what's in your export.
# ITTO's web export returns accented names ("Bénin", "Côte d'Ivoire") and we sanitise
# HTML entities (e.g. "C&#039;Ivoire") before lookup.
import html as _html

ITTO_COUNTRY_TO_ISO3 = {
    "Ghana": "GHA",
    "Cote d'Ivoire": "CIV",
    "Côte d'Ivoire": "CIV",
    "C\u00f4te d'Ivoire": "CIV",
    "Côte d'Ivoire": "CIV",
    "C\u00f4te d\u2019Ivoire": "CIV",
    "Nigeria": "NGA",
    "Liberia": "LBR",
    "Senegal": "SEN",
    "Guinea": "GIN",
    "Sierra Leone": "SLE",
    "Togo": "TGO",
    "Benin": "BEN",
    "Bénin": "BEN",
    "B\u00e9nin": "BEN",
    "Gambia": "GMB",
    "Burkina Faso": "BFA",
    "Guinea-Bissau": "GNB",
}


def _normalise_country(name: str) -> str:
    """Strip HTML entities then collapse whitespace, so Burkina Faso and
    "Côte  d'Ivoire" (variants appear in ITTO exports) match consistently."""
    if not isinstance(name, str):
        return name
    decoded = _html.unescape(name)
    return " ".join(decoded.split()).strip()


def guess_columns(df: pd.DataFrame) -> dict:
    """Try to auto-match ITTO's actual column headers to our standard names."""
    resolved = {}
    for hint, standard_name in COLUMN_MAP_HINTS.items():
        match = next((c for c in df.columns if hint in c.lower()), None)
        if match:
            resolved[match] = standard_name
    return resolved


def main():
    if not os.path.exists(RAW_FILE):
        print(f"ERROR: {RAW_FILE} not found.")
        print("Follow README.md Step 3 to manually export data from ITTO first,")
        print(f"and save it as {RAW_FILE}")
        return

    df = pd.read_csv(RAW_FILE)
    print("Columns found in your ITTO export:", list(df.columns))

    col_map = guess_columns(df)
    print("Auto-matched columns:", col_map)

    missing = set(COLUMN_MAP_HINTS.values()) - set(col_map.values())
    if missing:
        print(f"\n[WARNING] Could not auto-match: {missing}")
        print("Open the CSV, check the real header names, and edit COLUMN_MAP_HINTS")
        print("in this script accordingly, then re-run.")
        return

    df = df.rename(columns=col_map)
    df["country_norm"] = df["country"].map(_normalise_country)
    df["country_code"] = df["country_norm"].map(ITTO_COUNTRY_TO_ISO3)

    unmatched = df[df["country_code"].isna()]["country_norm"].unique()
    if len(unmatched) > 0:
        print(f"[WARNING] Unmatched country names (add to ITTO_COUNTRY_TO_ISO3): {unmatched}")

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    clean = df[["country_code", "country", "year", "product", "flow", "value"]].dropna(
        subset=["country_code", "year"]
    )

    clean.to_csv(OUT_FILE, index=False)
    print(f"\nSaved {len(clean)} rows to {OUT_FILE}")
    print(clean.head(10))


if __name__ == "__main__":
    main()


