"""
Phase 1 ingestion: pull target 10-K filings from SEC EDGAR, extract clean
sections, and save them to disk with a manifest.

Run once. edgartools caches filings locally, so re-runs are cheap.
TIP: test on ONE company first (comment out the others in CORPUS) before
running the full loop.
"""

import os
import json
from pathlib import Path

from dotenv import load_dotenv
from edgar import Company, set_identity

# --- 1. Identity (REQUIRED by SEC; missing/generic = 403 + ~10-min IP block) ---
# Put EDGAR_IDENTITY="Your Name your.email@example.com" in your environment/.env,
# or edit the fallback string below.
load_dotenv()
set_identity(f"Srikar {os.environ['EDGAR_CONTACT_EMAIL']}")

# --- 2. Your corpus: ticker -> fiscal years (from FinanceBench coverage) ---
# Edit freely. Add FY2024/2025 later for a "recent, not-memorized" eval slice.
CORPUS = {
    "MSFT": [2016, 2023],             # Microsoft (fiscal year ends ~June 30)
    "JPM":  [2022],                   # JPMorgan Chase (bank -> retrieval stressor)
    "ADBE": [2015, 2016, 2017, 2022], # Adobe (fiscal year ends ~early Dec)
    "AWK":  [2020, 2021, 2022],       # American Water Works (utility, 3 consecutive)
    "BA":   [2018, 2022],             # Boeing (aerospace/defense)
    "PFE":  [2021]   # Pfizer (pharma; FinanceBench gold = FY2021 only)
}

# --- 3. Sections we care about (section-aware chunking depends on this) ---
# (label, convenience_property_or_None, bracket_key)
SECTIONS = [
    ("item1_business",      "business",              "Item 1"),
    ("item1a_risk_factors", "risk_factors",          "Item 1A"),
    ("item7_mdna",          "management_discussion", "Item 7"),
    ("item8_financials",    None,                    "Item 8"),
]

OUT = Path("./data/raw")
OUT.mkdir(parents=True, exist_ok=True)


def section_text(tenk, prop, key):
    """Return clean section text, trying the convenience property then [] access."""
    if prop:
        try:
            val = getattr(tenk, prop, None)
            if val:
                return str(val).strip()
        except Exception:
            pass
    try:
        sec = tenk[key]
        if sec is not None:
            text = getattr(sec, "text", None) or str(sec)
            text = text.strip()
            if text:
                return text
    except Exception:
        pass
    return None


def fiscal_year(filing):
    """Fiscal year = the year of the reporting-period END, not the filing date.
    (Microsoft/Adobe have off-calendar fiscal years, so filing_date would be wrong.)"""
    for attr in ("period_of_report", "report_date"):
        val = getattr(filing, attr, None)
        if val:
            s = str(val)
            if len(s) >= 4 and s[:4].isdigit():
                return int(s[:4])
    return None


manifest = []

for ticker, years in CORPUS.items():
    print(f"\n=== {ticker} ===")
    company = Company(ticker)

    # all 10-K filings, excluding 10-K/A amendments
    filings = [f for f in company.get_filings(form="10-K") if f.form == "10-K"]

    # map fiscal year -> filing (keep the first/most-recent match per year)
    by_year = {}
    for f in filings:
        fy = fiscal_year(f)
        if fy is not None and fy not in by_year:
            by_year[fy] = f

    for year in years:
        filing = by_year.get(year)
        if filing is None:
            print(f"  [MISS] no 10-K for FY{year} (available: {sorted(by_year)})")
            continue

        try:
            tenk = filing.obj()
        except Exception as e:
            print(f"  [ERR] FY{year} obj() failed: {e}")
            continue

        folder = OUT / ticker / str(year)
        folder.mkdir(parents=True, exist_ok=True)

        saved = []
        for label, prop, key in SECTIONS:
            text = section_text(tenk, prop, key)
            if text:
                (folder / f"{label}.txt").write_text(text, encoding="utf-8")
                saved.append(label)
            else:
                print(f"  [warn] FY{year}: section '{label}' missing "
                      f"-> inspect tenk.sections for this filing")

        # best-effort financials for later numeric ground truth (optional now)
        try:
            inc = tenk.financials.income_statement()
            df = inc.to_dataframe() if hasattr(inc, "to_dataframe") else None
            if df is not None:
                df.to_json(folder / "income_statement.json")
        except Exception:
            pass  # fine to skip; we pull XBRL facts properly in Phase 5

        manifest.append({
            "ticker": ticker,
            "company": str(getattr(filing, "company", ticker)),
            "cik": getattr(filing, "cik", None),
            "fiscal_year": year,
            "form": filing.form,
            "filing_date": str(getattr(filing, "filing_date", "")),
            "period_of_report": str(getattr(filing, "period_of_report", "")),
            "accession_number": getattr(filing, "accession_number", None),
            "sections": saved,
            "path": str(folder),
        })
        print(f"  [ok] FY{year}: saved {len(saved)} sections -> {folder}")

Path("data").mkdir(exist_ok=True)
Path("data/manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"\nDone. {len(manifest)} filings in manifest -> data/manifest.json")