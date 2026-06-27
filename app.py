from __future__ import annotations

import functools
import hashlib
import io
import math
import re
from collections import Counter
import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="HS & Shipment Pre-Check", layout="wide")

_CONFECTIONERY_WORDS = (
    "chocolate", "chocolates", "biscuit", "biscuits",
    "candy", "candies", "confection", "confections",
    "snack", "snacks", "cookie", "cookies",
)

# Common ISO 4217 currency text codes used to build the value-strip pattern.
_ISO_CODES = (
    'AED|AFN|ALL|ARS|AUD|BDT|BGN|BHD|BRL|CAD|CHF|CLP|CNY|COP'
    '|CZK|DKK|EGP|ETB|EUR|GBP|GEL|GHS|HKD|HRK|HUF|IDR|ILS|INR|IQD|IRR'
    '|JOD|JPY|KES|KRW|KWD|LBP|LKR|MAD|MXN|MYR|NGN|NOK|NZD|OMR|PHP|PKR'
    '|PLN|QAR|RON|RSD|RUB|SAR|SEK|SGD|THB|TRY|TWD|TZS|UAH|UGX|USD|UZS'
    '|VND|XAF|XOF|ZAR|ZMW'
)
# Strips currency symbols (£$€¥₹) and ISO 4217 text codes that appear as a
# prefix ("GBP 250", "USD1250") or suffix ("250 EUR", "250USD") in value
# fields exported from ERP/accounting systems.  Start/end anchors are used
# instead of \b so no-space variants like "USD1250" are handled correctly
# (there is no word boundary between a letter and a digit in \b semantics).
_VALUE_STRIP_RE = re.compile(
    r'[£$€¥₹]'
    r'|^(?:' + _ISO_CODES + r')\s*'
    r'|\s*(?:' + _ISO_CODES + r')$',
    re.IGNORECASE,
)
_FASHION_WORDS = (
    "belt", "belts", "wallet", "wallets", "glove", "gloves",
    "hat", "hats", "cap", "caps", "tie", "ties",
    "brooch", "brooches", "headband", "headbands",
)
_BAG_WORDS = (
    "bag", "bags", "handbag", "handbags", "purse", "purses",
    "tote", "totes", "clutch", "clutches", "satchel", "satchels",
    "backpack", "backpacks", "rucksack", "rucksacks",
    "briefcase", "briefcases",
)

# Pre-compiled alternation patterns for keyword groups used in classification.
_CONFECTIONERY_RE = re.compile(
    r'\b(?:' + '|'.join(re.escape(w) for w in _CONFECTIONERY_WORDS) + r')\b'
)
_FASHION_RE = re.compile(
    r'\b(?:' + '|'.join(re.escape(w) for w in _FASHION_WORDS) + r')\b'
)
_BAG_RE = re.compile(
    r'\b(?:' + '|'.join(re.escape(w) for w in _BAG_WORDS) + r')\b'
)
_FREE_MARKER_RE = re.compile(
    r'\b(?:fragrance|perfume)[-–— ]free\b'   # fragrance-free, perfume free, etc.
    r'|\b(?:no|without)\s+(?:fragrance|perfume)\b'  # no fragrance, without perfume
    r'|\bunscented\b'                                # unscented
)
# Preceding qualifiers that indicate a synthetic imitation rather than the genuine
# material.  These complement the negative-lookahead approach used in _SILK_RE and
# _LEATHER_RE (which only catch following modifiers like "silk-effect", "leather-look")
# by also catching constructions like "faux leather", "vegan leather", "PU leather",
# "synthetic silk" — all of which would otherwise pass through the duty-code upgrade
# branches and attract the wrong (higher) duty rates.
_FAUX_SILK_RE = re.compile(
    r'\b(?:faux|synthetic|artificial|imitation|fake)\s+silk\b'
)
_FAUX_LEATHER_RE = re.compile(
    r'\b(?:faux|vegan|synthetic|artificial|imitation|fake|pu|polyurethane)\s+leather\b'
)
_EURO_DECIMAL_RE = re.compile(r',\d{1,2}$')
_PERFUME_RE = re.compile(
    r'\b(?:perfumes?|fragrances?|colognes?|aftershaves?'
    r'|eau[ -]de[ -](?:parfum|toilette|cologne))\b'
)
_SCARF_RE = re.compile(r'\b(?:scarf|scarves)\b')
# Negative-lookahead excludes compound modifiers such as "silk-effect", "silk-like",
# "leather-look", "leather-feel", etc. which describe synthetic imitations rather
# than the genuine material, preventing false duty-code upgrades for polyester/PU goods.
_SILK_RE = re.compile(r'\bsilk\b(?![-\s](?:effect|like|look|feel|finish|touch)\b)')
_LEATHER_RE = re.compile(r'\bleather\b(?![-\s](?:look|like|effect|feel|finish|touch)\b)')

# Threshold above which items attract additional customs scrutiny
HIGH_VALUE_THRESHOLD = 1000.00

# Valid risk levels
RISK_GREEN = "GREEN"
RISK_AMBER = "AMBER"
RISK_RED = "RED"

# Review queue status values
STATUS_PENDING = "Pending review"
STATUS_APPROVED = "Approved"
STATUS_OVERRIDDEN = "Overridden — pending analyst"

# Columns produced by classify_product — used to drop conflicts before bulk concat
RESULT_COLUMNS = frozenset({"hs6", "uk_code", "confidence", "risk", "duty", "vat", "explanation"})

# Sentinel values used in result rows
ERROR_CODE = "ERROR"
UNCLASSIFIED_CODE = "UNCLASSIFIED"

# Maximum number of distinct (desc, material, category, high_value) tuples held in
# the classification cache.  Origin is excluded from the key because classification
# logic is identical regardless of origin — only the explanation note differs, and
# that is appended in the classify_product wrapper outside the cache.  Covers the
# vast majority of real-world SKU catalogues while keeping memory bounded to roughly
# 4 MB worst-case.
_CACHE_MAX_SIZE = 4096


def _parse_value(raw) -> tuple[float, str]:
    """Convert raw value to (normalised_float, warning_message).

    The warning is non-empty only when the raw input was absent or invalid
    and has been defaulted to 0.0.
    Handles common CSV formats: "£1,250.00", "$500", "1,000.50",
    "GBP 250", "250 USD", "EUR1250,00".
    """
    if isinstance(raw, str):
        # Strip currency symbols and ISO 4217 text codes in one pass; strip()
        # afterward removes any whitespace left between the code and the number
        # (e.g. "GBP 250" → "GBP 250" → sub → " 250" → strip → "250").
        s = _VALUE_STRIP_RE.sub('', raw.strip()).strip()
        if not s:
            return 0.0, " Warning: declared value was missing; defaulted to £0 for risk assessment."
        # Detect European decimal format: comma followed by 1–2 digits at end,
        # with exactly one comma (e.g. "1.250,00" → "1250.00"). The single-comma
        # guard prevents "1,250,00" (two commas, a common typo) from matching the
        # Euro branch and producing the unparseable "1.250.00". Otherwise treat
        # commas as UK/US thousands separators (e.g. "1,250.00" → "1250.00").
        comma_count = s.count(',')
        euro_tail = _EURO_DECIMAL_RE.search(s)
        if comma_count > 1 and euro_tail:
            # Two+ commas with a decimal-like tail (e.g. "1,250,00") is ambiguous;
            # the value cannot be reliably parsed so default to zero with a warning.
            return 0.0, " Warning: declared value format is ambiguous (multiple commas); defaulted to £0 for risk assessment."
        if euro_tail and comma_count == 1:
            s = s.replace('.', '').replace(',', '.')
        else:
            s = s.replace(',', '')
        cleaned = s
    else:
        cleaned = raw
    try:
        v = float(cleaned)
    except (TypeError, ValueError):
        try:
            is_missing = pd.isna(raw)
        except (TypeError, ValueError):
            is_missing = raw is None
        msg = (
            " Warning: declared value was missing; defaulted to £0 for risk assessment."
            if is_missing
            else " Warning: declared value could not be parsed; defaulted to £0 for risk assessment."
        )
        return 0.0, msg
    if math.isnan(v):
        return 0.0, " Warning: declared value was missing; defaulted to £0 for risk assessment."
    if math.isinf(v):
        return 0.0, " Warning: declared value was non-finite; defaulted to £0 for risk assessment."
    if v < 0.0:
        return 0.0, " Warning: declared value was negative; defaulted to £0 for risk assessment."
    return round(v, 2), ""


def _normalise_value(value) -> float:
    """Convert value to a finite, non-negative float rounded to pence."""
    v, _ = _parse_value(value)
    return v


def classify_product(description, material, origin, category, value):
    """Normalise inputs then delegate to the cached implementation."""
    # Skip full normalisation when the caller has already parsed the value to a
    # finite non-negative float (e.g. classify_row passes the result of _parse_value
    # directly).  This avoids a redundant _parse_value round-trip on bulk uploads.
    if isinstance(value, float) and not math.isnan(value) and not math.isinf(value) and value >= 0:
        v = round(value, 2)
    else:
        v = _normalise_value(value)
    origin_upper = (origin or "").strip().upper()
    origin_note = (
        f" Country of origin: {origin_upper}."
        if origin_upper
        else " Warning: country of origin not declared — required for customs clearance."
    )
    # Return a shallow copy so callers cannot mutate the lru_cache entry.
    # Origin is handled here (outside the cache) so products from different countries
    # with identical descriptions/materials/categories share the same cache entry.
    result = dict(_classify_product_cached(
        (description or "").strip().lower(),
        (material or "").strip().lower(),
        (category or "").strip().lower(),
        v >= HIGH_VALUE_THRESHOLD,
    ))
    result["explanation"] = result["explanation"] + origin_note
    return result


@functools.lru_cache(maxsize=_CACHE_MAX_SIZE)
def _classify_product_cached(desc, material_lower, category_lower, high_value):
    # high_value is a bool; using it instead of the raw value means products that
    # share the same description/material/category and the same high-value status
    # hit the same cache entry regardless of exact declared price.  Origin is NOT
    # part of the key — classification logic is identical across origins; only the
    # explanation note differs and that is appended by classify_product after the
    # cache lookup.
    hv_note = " High declared value flagged for additional customs scrutiny." if high_value else ""

    # Pre-compute all keyword flags once to avoid redundant regex evaluation.
    is_scarf = bool(_SCARF_RE.search(desc))
    # Material is the authoritative source for composition.  Only fall back to
    # description when the material field was not supplied, so that terms like
    # "silk-effect polyester" or "leather-look PU" in a description do not
    # trigger the silk/leather duty codes when the actual material differs.
    # _FAUX_SILK_RE / _FAUX_LEATHER_RE guard against preceding qualifiers
    # ("faux leather", "vegan leather", "synthetic silk", "PU leather", etc.)
    # which would otherwise pass through _SILK_RE / _LEATHER_RE unchanged and
    # attract the wrong (higher) duty-code branches.
    is_silk = (
        bool(_SILK_RE.search(material_lower)) and not bool(_FAUX_SILK_RE.search(material_lower))
    ) or (
        not material_lower
        and bool(_SILK_RE.search(desc))
        and not bool(_FAUX_SILK_RE.search(desc))
    )
    is_leather = (
        bool(_LEATHER_RE.search(material_lower)) and not bool(_FAUX_LEATHER_RE.search(material_lower))
    ) or (
        not material_lower
        and bool(_LEATHER_RE.search(desc))
        and not bool(_FAUX_LEATHER_RE.search(desc))
    )
    # Either "fragrance-free" or "perfume-free" in description or material negates
    # the product being a fragrance/perfume; both flags suppress ALL perfume signals
    # (including cologne, aftershave, eau-de) not just the keyword they name.
    # Fragrance is checked in both desc and material_lower for consistency with how
    # is_silk and is_leather inspect both fields (e.g. "alcohol base and fragrance
    # compounds" in material correctly triggers perfume classification).
    # category_lower == "beauty" is intentionally NOT included: it is too broad and
    # would misclassify all cosmetics (face creams, lipstick, etc.) as perfumes.
    _free_marker = bool(
        _FREE_MARKER_RE.search(desc) or _FREE_MARKER_RE.search(material_lower)
    )
    is_perfume = not _free_marker and bool(
        _PERFUME_RE.search(desc) or _PERFUME_RE.search(material_lower)
    )
    # Non-fragrance beauty products (skincare, make-up, etc.) fall here.
    is_cosmetics = category_lower == "beauty" and not is_perfume
    is_confectionery = bool(_CONFECTIONERY_RE.search(desc))
    # Confectionery keywords drive food classification only when the category is
    # blank (no signal) or explicitly "food".  Any non-empty category — whether a
    # known type like "bags"/"beauty" or an unknown bulk-CSV value like "electronics"
    # — is treated as a contradicting signal and suppresses the keyword override.
    # This prevents "chocolate-coloured sofa" (category: furniture) and
    # "chocolate gift bag" (category: bags) from being misclassified as food.
    is_food = category_lower == "food" or (
        is_confectionery and not category_lower
    )
    is_fashion = category_lower == "fashion_accessories" or bool(_FASHION_RE.search(desc))
    # Bag detection: fashion_accessories and food categories override bag keywords.
    # fashion_accessories: "handbag charm" is an accessory, not a bag.
    # food: "chocolate gift bag" is food, not a handbag — without this guard the
    # is_bag branch fires before is_food and produces an incorrect HS 4202 code.
    # The is_food guard covers both an explicit category="food" and the case where
    # confectionery keywords trigger food with no category (e.g. "chocolate gift bag"
    # with blank category), since is_bag is checked before is_food in the decision tree.
    # category="bags" only fires when description keywords do not indicate a fashion
    # accessory, preventing items like belts or scarves from being misrouted to bag
    # HS codes due to a miscategorised or imprecise category field.
    _bag_keyword = bool(_BAG_RE.search(desc))
    # Keyword path does NOT exclude is_fashion: "belt bag" / "clutch bag" descriptions
    # explicitly name a bag and should be classified as such even when a fashion keyword
    # ("belt", "clutch") is also present.  The category path uses the stricter guard
    # (not is_fashion) because category="bags" on an item whose description says only
    # "belt" is likely a data-entry error; the description is the authoritative signal.
    _bag_by_keyword = _bag_keyword and category_lower not in {"fashion_accessories", "food"} and not is_food
    _bag_by_category = category_lower == "bags" and not is_fashion
    is_bag = _bag_by_keyword or _bag_by_category

    if is_scarf and is_silk:
        return {
            "hs6": "621410",
            "uk_code": "6214100090",
            "confidence": 0.94,
            "risk": RISK_RED if high_value else RISK_GREEN,
            "duty": "8%",
            "vat": "20%",
            "explanation": "Classified under silk scarves based on material composition and accessory type." + hv_note,
        }
    elif is_bag and is_leather:
        return {
            "hs6": "420221",
            "uk_code": "4202210000",
            "confidence": 0.88,
            "risk": RISK_RED if high_value else RISK_AMBER,
            "duty": "16%",
            "vat": "20%",
            "explanation": "Classified under handbags with outer surface of leather." + hv_note,
        }
    elif is_bag:
        return {
            "hs6": "420229",
            "uk_code": "4202290000",
            "confidence": 0.65,
            "risk": RISK_RED if high_value else RISK_AMBER,
            "duty": "3.7%",
            "vat": "20%",
            "explanation": "Classified under handbags with other outer surface; verify material composition for precise subheading." + hv_note,
        }
    elif is_scarf:
        return {
            "hs6": "621490",
            "uk_code": "6214900000",
            "confidence": 0.72,
            "risk": RISK_RED if high_value else RISK_GREEN,
            "duty": "12%",
            "vat": "20%",
            "explanation": "Classified under scarves and similar articles (non-silk); verify fibre composition for precise subheading (wool: 621420, synthetic fibres: 621430, other fibres: 621490)." + hv_note,
        }
    elif is_perfume:
        return {
            "hs6": "330300",
            "uk_code": "3303001000",
            "confidence": 0.81,
            "risk": RISK_RED if high_value else RISK_AMBER,
            "duty": "6.5%",
            "vat": "20%",
            "explanation": "Classified under perfumes and toilet waters; regulated cosmetics handling required." + hv_note,
        }
    elif is_cosmetics:
        return {
            "hs6": "330499",
            "uk_code": "3304990000",
            "confidence": 0.68,
            "risk": RISK_RED if high_value else RISK_AMBER,
            "duty": "6.5%",
            "vat": "20%",
            "explanation": "Classified under beauty and make-up preparations; verify specific subheading for product type (e.g. lip, eye, skin care)." + hv_note,
        }
    elif is_food:
        food_vat = "20%" if is_confectionery else "0%"
        vat_note = (
            " Note: confectionery and snack products (e.g. chocolate, biscuits, candy, confections, snacks)"
            " are standard-rated at 20% VAT in the UK."
            if is_confectionery
            else " Note: most food is zero-rated for VAT in the UK; verify the applicable rate."
        )
        return {
            "hs6": "210690",
            "uk_code": "2106909900",
            "confidence": 0.65,
            "risk": RISK_RED if high_value else RISK_AMBER,
            "duty": "varies",
            "vat": food_vat,
            "explanation": (
                "Classified under miscellaneous food preparations; phytosanitary and food safety checks required."
                + vat_note + hv_note
            ),
        }
    elif is_fashion:
        return {
            "hs6": "621790",
            "uk_code": "6217900000",
            "confidence": 0.70,
            "risk": RISK_RED if high_value else RISK_GREEN,
            "duty": "12%",
            "vat": "20%",
            "explanation": "Classified under other made-up clothing accessories; verify composition for precise subheading." + hv_note,
        }
    else:
        return {
            "hs6": UNCLASSIFIED_CODE,
            "uk_code": UNCLASSIFIED_CODE,
            "confidence": 0.0,
            "risk": RISK_RED if high_value else RISK_AMBER,
            "duty": "TBD",
            "vat": "TBD",
            "explanation": "Insufficient structured data; manual review recommended." + hv_note,
        }


def _format_confidence(conf) -> str:
    """Return confidence as a clamped percentage string, e.g. '94%'."""
    try:
        return f"{min(100, max(0, round(float(conf) * 100)))}%"
    except (TypeError, ValueError):
        return "0%"


def _safe_str(v) -> str:
    """Convert a value to string, returning empty string for NaN/None."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v)


def classify_row(row):
    """Apply classify_product to a DataFrame row; safe for use with df.apply()."""
    val, val_warning = 0.0, ""
    try:
        val, val_warning = _parse_value(row.get("value"))
        result = classify_product(
            _safe_str(row.get("description", "")),
            _safe_str(row.get("material", "")),
            _safe_str(row.get("origin", "")),
            _safe_str(row.get("category", "")),
            val,
        )
        if val_warning:
            result["explanation"] += val_warning
        return pd.Series(result)
    except Exception as e:
        row_idx = getattr(row, "name", None)
        # hasattr(__index__) covers both Python int and numpy integer scalars.
        display_idx = (row_idx + 1) if hasattr(row_idx, "__index__") else row_idx
        prefix = f"Row {display_idx}: " if display_idx is not None else ""
        msg = f"{prefix}Classification failed: {type(e).__name__}: {str(e)}"
        suffix = val_warning
        msg_budget = max(10, 250 - len(suffix))
        truncated = (msg[:msg_budget - 3] + "...") if len(msg) > msg_budget else msg
        explanation = truncated + suffix
        return pd.Series({
            "hs6": ERROR_CODE,
            "uk_code": ERROR_CODE,
            "confidence": 0.0,
            "risk": RISK_RED if val >= HIGH_VALUE_THRESHOLD else RISK_AMBER,
            "duty": "TBD",
            "vat": "TBD",
            "explanation": explanation,
        })


def _add_to_review_queue(result: dict):
    """Add a classified item to the review queue if not already present.

    Deduplicates on (description, high_value_flag, uk_code) so that re-clicking
    the button for the same product does not create duplicate queue entries, but
    a genuine reclassification that produces a different code or crosses the
    high-value threshold (and thus a different risk rating) is still added.
    Silently ignores ERROR and UNCLASSIFIED items — callers filter these, but
    this guard prevents accidental queue corruption if called directly.
    """
    if result.get("uk_code") in {ERROR_CODE, UNCLASSIFIED_CODE}:
        return
    safe_val = _normalise_value(result.get("value", 0.0))
    # Use the high-value boolean rather than the raw amount: classification only
    # distinguishes values by whether they meet HIGH_VALUE_THRESHOLD, so two
    # sub-threshold prices for the same product produce the same classification
    # and should map to the same dedup key.
    key = (
        _safe_str(result.get("description", "")).strip().lower(),
        safe_val >= HIGH_VALUE_THRESHOLD,
        _safe_str(result.get("uk_code", "")),
    )
    if key not in st.session_state["review_keys"]:
        st.session_state["review_keys"].add(key)
        st.session_state["review_items"].append({
            "Product": result.get("description", ""),
            "Suggested Code": result.get("uk_code", UNCLASSIFIED_CODE),
            "Confidence": _format_confidence(result.get("confidence", 0.0)),
            "Explanation": result.get("explanation", ""),
            "Risk": result.get("risk", RISK_AMBER),
            "Status": STATUS_PENDING,
        })


def _apply_bulk_review(new_status: str, audit_event: str, toast_msg: str, toast_icon: str) -> None:
    """Set all pending review-queue items to new_status and log the action."""
    ts = datetime.now().isoformat(timespec="microseconds")
    changed = 0
    skipped_unclassified = 0
    for item in st.session_state["review_items"]:
        if item["Status"] == STATUS_PENDING:
            # Never bulk-action items with no assigned code; they require manual
            # code entry before either approval or override.
            if item.get("Suggested Code") in {UNCLASSIFIED_CODE, ERROR_CODE}:
                skipped_unclassified += 1
                continue
            item["Status"] = new_status
            changed += 1
    if changed > 0:
        skipped_note = (
            f"; {skipped_unclassified} item(s) skipped (unclassified or errored — require manual code assignment)"
            if skipped_unclassified
            else ""
        )
        st.session_state["audit_log"].append({"Timestamp": ts, "Event": audit_event.format(count=changed) + skipped_note})
        st.toast(toast_msg.format(count=changed), icon=toast_icon)
        st.rerun()
    elif skipped_unclassified:
        st.session_state["audit_log"].append({
            "Timestamp": ts,
            "Event": (
                f"Bulk action attempted: {skipped_unclassified} pending item(s) skipped — "
                "all are unclassified or errored and require manual code assignment before approval."
            ),
        })
        st.toast("No pending items to action — unclassified or errored items require manual code assignment.", icon="ℹ️")
        # No st.rerun() — the review queue display is unchanged, so a rerun would
        # only reset the data_editor widget state unnecessarily.  The audit log
        # entry is persisted in session state and visible on the Audit Trail page.
    else:
        st.toast("No pending items in the review queue.", icon="ℹ️")


def _process_bulk_upload(file_bytes: bytes, filename: str, file_id: tuple[str, str]) -> None:
    """Validate, classify, and store results for a newly uploaded CSV.

    Accepts raw bytes so the function is independent of the UploadedFile
    cursor position and can be called without side-effects on the file object.
    Uses return-on-error instead of st.stop() so the caller can still render
    any previously stored bulk results after a failed upload attempt.
    """
    st.session_state["_bulk_messages"] = []
    # Reset stale results so a failed upload never shows the previous run's data.
    st.session_state["bulk_result"] = None
    try:
        # Read one extra row so len(df) > 5000 can detect oversized files.
        df = pd.read_csv(io.BytesIO(file_bytes), nrows=5001, encoding="utf-8-sig", encoding_errors="replace")
        df.columns = df.columns.str.strip().str.lower()
        # Warn if any cell contains U+FFFD (the Unicode replacement character),
        # which indicates bytes that could not be decoded from the file's encoding.
        str_cols = df.select_dtypes(include=["object", "string"])
        if not str_cols.empty and str_cols.apply(
            lambda col: col.str.contains("\ufffd", regex=False, na=False).any()
        ).any():
            st.session_state["_bulk_messages"].append(("warning", (
                "Some characters in the CSV could not be decoded and have been "
                "replaced with \ufffd. Re-save the file as UTF-8 to ensure accurate "
                "classification."
            )))
    except pd.errors.ParserError:
        st.session_state["_bulk_messages"].append(("error", "CSV format is invalid — check that columns are comma-separated and the file is UTF-8 encoded."))
        return
    except Exception as e:
        st.session_state["_bulk_messages"].append(("error", f"Failed to read file: {e}"))
        return

    if len(df) > 5000:
        st.session_state["_bulk_messages"].append(("error", "CSV exceeds the 5,000-row limit (more than 5,000 rows detected). Split the file and re-upload."))
        return

    if df.empty:
        st.session_state["_bulk_messages"].append(("error", "The uploaded CSV contains no data rows."))
        return

    required = {"description", "material", "origin", "category", "value"}
    missing = required - set(df.columns)
    if missing:
        st.session_state["_bulk_messages"].append(("error", f"Missing required columns: {', '.join(sorted(missing))}"))
        return

    # Warn if pre-existing result columns will be overwritten.
    overlapping = sorted(col for col in RESULT_COLUMNS if col in df.columns)
    if overlapping:
        st.session_state["_bulk_messages"].append(("warning", f"The following columns from your CSV will be overwritten by classification results: {', '.join(overlapping)}"))
    # Drop any pre-existing result columns to avoid duplicate columns after concat.
    input_df = df.drop(columns=overlapping).reset_index(drop=True)
    try:
        with st.spinner(f"Classifying {len(input_df)} rows…"):
            classified = input_df.apply(classify_row, axis=1)
            result_df = pd.concat([input_df, classified], axis=1)
    except Exception as e:
        st.session_state["_bulk_messages"].append(("error", f"Classification failed: {e}"))
        return

    error_count = (result_df["hs6"] == ERROR_CODE).sum()
    unclassified_count = (result_df["hs6"] == UNCLASSIFIED_CODE).sum()
    detail_parts = []
    if unclassified_count:
        detail_parts.append(f"{unclassified_count} unclassified")
    if error_count:
        detail_parts.append(f"{error_count} error{'s' if error_count != 1 else ''}")
    row_word = "row" if len(result_df) == 1 else "rows"
    summary = f"Processed {len(result_df)} {row_word}"
    if detail_parts:
        summary += f" ({', '.join(detail_parts)})"
    st.session_state["audit_log"].append({
        "Timestamp": datetime.now().isoformat(timespec="microseconds"),
        "Event": f"Bulk upload: {summary} from '{filename}'",
    })
    st.session_state["bulk_result"] = {
        "df": result_df,
        "summary": summary,
        "filename": filename,
    }

    queueable_df = result_df[~result_df["hs6"].isin({ERROR_CODE, UNCLASSIFIED_CODE})]
    _queue_cols = ["description", "value", "uk_code", "confidence", "explanation", "risk"]
    try:
        for row in queueable_df[_queue_cols].to_dict("records"):
            _add_to_review_queue({
                "description": _safe_str(row.get("description", "")),
                "value": row.get("value", 0.0),
                "uk_code": _safe_str(row.get("uk_code", "")),
                "confidence": row.get("confidence", 0.0),
                "explanation": _safe_str(row.get("explanation", "")),
                "risk": _safe_str(row.get("risk")) or RISK_AMBER,
            })
    except Exception as e:
        st.session_state["_bulk_messages"].append(("warning", f"Review queue could not be fully populated: {e}"))
    # Mark the file as processed after state is updated. Set unconditionally so a
    # partial queue failure does not trigger an infinite re-classification loop on
    # subsequent reruns.
    st.session_state["_bulk_file_id"] = file_id


# Initialise session state keys once so all pages can rely on them existing.
st.session_state.setdefault("review_items", [])
st.session_state.setdefault("review_keys", set())
st.session_state.setdefault("audit_log", [])
st.session_state.setdefault("bulk_result", None)
st.session_state.setdefault("_bulk_file_id", None)
st.session_state.setdefault("_bulk_messages", [])
st.session_state.setdefault("last_result", None)
if "seed_logs" not in st.session_state:
    _seed_date = datetime.now().strftime("%Y-%m-%d")
    st.session_state["seed_logs"] = [
        {"Timestamp": f"{_seed_date}T09:12:00.000000", "Event": "SKU123 classified as 6214100090 by system"},
        {"Timestamp": f"{_seed_date}T09:17:00.000000", "Event": "Reviewed by compliance_officer_01"},
        {"Timestamp": f"{_seed_date}T09:18:00.000000", "Event": "Approved and published to product master"},
    ]
    del _seed_date

st.sidebar.title("HS & Shipment Pre-Check")
page = st.sidebar.radio("Navigate", ["Dashboard", "Classify", "Bulk Upload", "Review Queue", "Audit Trail"])

if page == "Dashboard":
    st.title("HS & Shipment Pre-Check Dashboard")

    session_items = st.session_state["review_items"]
    session_total = len(session_items)
    status_counts = Counter(i["Status"] for i in session_items)
    session_pending = status_counts[STATUS_PENDING]
    session_approved = status_counts[STATUS_APPROVED]
    session_overridden = status_counts[STATUS_OVERRIDDEN]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Session SKUs", session_total if session_total else "—")
    c2.metric("Pending Review", session_pending if session_total else "—")
    c3.metric("Approved", session_approved if session_total else "—")
    c4.metric("Overridden", session_overridden if session_total else "—")

    st.caption("Metrics reflect classifications performed in this session.")

    if session_items:
        st.subheader("Session Risk Distribution")
        counted = Counter(i["Risk"] for i in session_items)
        risk_df = pd.DataFrame(
            {"Risk": [RISK_GREEN, RISK_AMBER, RISK_RED],
             "Count": [counted[RISK_GREEN], counted[RISK_AMBER], counted[RISK_RED]]},
        )
    else:
        st.subheader("Session Risk Distribution (Demo)")
        st.info("No classifications yet this session. The chart below shows illustrative demo data.")
        risk_df = pd.DataFrame({"Risk": [RISK_GREEN, RISK_AMBER, RISK_RED], "Count": [9710, 2140, 600]})
    st.bar_chart(risk_df.set_index("Risk"))

elif page == "Classify":
    st.title("Classify Product")

    left, right = st.columns([2, 1])

    with left:
        description = st.text_input("Product Description", "Luxury silk scarf with hand-rolled edges", max_chars=500)
        material = st.text_input("Material Composition", "100% silk", max_chars=200)
        origin = st.text_input("Country of Origin", "IT", max_chars=50)
        category = st.selectbox("Category", ["fashion_accessories", "bags", "beauty", "food", "other"])
        value = st.number_input("Declared Value (£)", min_value=0.0, value=250.0, step=10.0)

        if st.button("Run Classification"):
            if not description.strip():
                st.warning("Please enter a product description before classifying.")
            else:
                desc_clean = description.strip()
                mat_clean = material.strip()
                orig_clean = origin.strip().upper()
                ts = datetime.now().isoformat(timespec="microseconds")
                try:
                    result = classify_product(desc_clean, mat_clean, orig_clean, category, value)
                except Exception as exc:
                    st.session_state["audit_log"].append({
                        "Timestamp": ts,
                        "Event": f'"{desc_clean}" classification error — {type(exc).__name__}: {exc}',
                    })
                    st.error(f"Classification failed: {exc}")
                else:
                    entry = {
                        "description": desc_clean,
                        "material": mat_clean,
                        "origin": orig_clean,
                        "category": category,
                        "value": value,
                        "timestamp": ts,
                        **result,
                    }
                    st.session_state["last_result"] = entry
                    if result.get("hs6") == UNCLASSIFIED_CODE:
                        audit_event = f'"{entry["description"]}" could not be classified — manual code assignment required'
                    else:
                        _add_to_review_queue(entry)
                        audit_event = f'"{entry["description"]}" classified as {entry["uk_code"]} (risk: {entry["risk"]})'
                    st.session_state["audit_log"].append({
                        "Timestamp": entry["timestamp"],
                        "Event": audit_event,
                    })

    with right:
        st.info(
            "Check if your product description is customs-ready before shipment. "
            "Detect missing data, improve descriptions, and reduce shipment rejection risk."
        )

    if st.session_state["last_result"] is not None:
        r = st.session_state["last_result"]
        st.subheader("Classification Result")
        if r["hs6"] == UNCLASSIFIED_CODE:
            st.warning(
                "Could not assign an HS code from the information provided. "
                "Refine the product description or manually assign a commodity code before shipment."
            )
        a, b, c = st.columns(3)
        a.metric("HS6", r["hs6"])
        b.metric("UK Commodity Code", r["uk_code"])
        c.metric("Confidence", _format_confidence(r["confidence"]))

        d, e, f = st.columns(3)
        d.metric("Risk", r["risk"])
        e.metric("Duty", r["duty"])
        f.metric("VAT", r["vat"])

        st.write("**Explanation**")
        st.write(r["explanation"])

        st.write("**Audit Snapshot**")
        st.json({
            "product_description": r["description"],
            "material_composition": r["material"],
            "country_of_origin": r["origin"],
            "category": r["category"],
            "value_gbp": r["value"],
            "hs6": r["hs6"],
            "uk_commodity_code": r["uk_code"],
            "confidence": r["confidence"],
            "risk": r["risk"],
            "duty": r["duty"],
            "vat": r["vat"],
            "explanation": r["explanation"],
            "decision_timestamp": r["timestamp"],
        })

elif page == "Bulk Upload":
    st.title("Bulk Upload")
    uploaded = st.file_uploader(
        "Upload CSV with columns: description, material, origin, category, value",
        type=["csv"],
    )

    if uploaded:
        # Only re-process when the file actually changes; guards against
        # re-classifying (and adding duplicate audit entries) on every rerun.
        # MD5 of file contents is used as the dedup key so two different files
        # with the same name and byte size are still treated as distinct.
        raw_bytes = uploaded.getvalue()
        # usedforsecurity=False is required on FIPS-enabled Python 3.9+ systems;
        # the TypeError fallback keeps compatibility with Python 3.8.
        try:
            _hex = hashlib.md5(raw_bytes, usedforsecurity=False).hexdigest()
        except TypeError:
            _hex = hashlib.md5(raw_bytes).hexdigest()
        file_id = (uploaded.name, _hex)
        if st.session_state["_bulk_file_id"] != file_id:
            _process_bulk_upload(raw_bytes, uploaded.name, file_id)
    elif st.session_state["_bulk_file_id"] is not None:
        # File was removed — clear messages and results so previous state
        # does not bleed into a fresh upload attempt.
        st.session_state["_bulk_messages"] = []
        st.session_state["bulk_result"] = None
        st.session_state["_bulk_file_id"] = None

    for _level, _msg in st.session_state["_bulk_messages"]:
        if _level == "error":
            st.error(_msg)
        else:
            st.warning(_msg)

    bulk = st.session_state["bulk_result"]
    if bulk is not None:
        result_df = bulk["df"]
        error_rows = int((result_df["hs6"] == ERROR_CODE).sum())
        unclassified_rows = int((result_df["hs6"] == UNCLASSIFIED_CODE).sum())
        if error_rows == len(result_df):
            st.error(bulk["summary"])
        elif error_rows + unclassified_rows > 0:
            st.warning(bulk["summary"])
        else:
            st.success(bulk["summary"])
        st.dataframe(result_df, use_container_width=True)
        st.download_button(
            "Download Results CSV",
            data=result_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="hs_classification_results.csv",
            mime="text/csv",
        )
    elif not uploaded:
        st.caption("Use the sample CSV in the deployment bundle to test bulk processing.")

elif page == "Review Queue":
    st.title("Review Queue")

    items = st.session_state["review_items"]

    if items:
        display_cols = ["Product", "Suggested Code", "Confidence", "Risk", "Status", "Explanation"]
        review_df = pd.DataFrame(items, columns=display_cols)

        # Editable table: Status column is a dropdown; all other columns are read-only.
        edited_df = st.data_editor(
            review_df,
            column_config={
                "Status": st.column_config.SelectboxColumn(
                    "Status",
                    options=[STATUS_PENDING, STATUS_APPROVED, STATUS_OVERRIDDEN],
                    required=True,
                ),
            },
            disabled=["Product", "Suggested Code", "Confidence", "Risk", "Explanation"],
            hide_index=True,
            use_container_width=True,
            key="review_queue_editor",
        )

        # Detect per-row status changes made directly in the table and persist them.
        original_statuses = review_df["Status"].tolist()
        edited_statuses = edited_df["Status"].tolist()
        if original_statuses != edited_statuses:
            ts = datetime.now().isoformat(timespec="microseconds")
            for i, (orig_status, new_status) in enumerate(zip(original_statuses, edited_statuses)):
                if orig_status != new_status and i < len(items):
                    items[i]["Status"] = new_status
                    st.session_state["audit_log"].append({
                        "Timestamp": ts,
                        "Event": (
                            f"Review Queue: '{items[i]['Product']}' "
                            f"status changed from {orig_status} to {new_status}"
                        ),
                    })
            st.rerun()

        st.write("**Bulk review actions**")
        col1, col2 = st.columns(2)

        if col1.button("Approve All"):
            _apply_bulk_review(
                STATUS_APPROVED,
                "Review Queue: {count} pending item(s) approved in bulk",
                "{count} pending item(s) marked as approved.",
                "✅",
            )

        if col2.button("Override All"):
            _apply_bulk_review(
                STATUS_OVERRIDDEN,
                "Review Queue: {count} pending item(s) flagged for analyst override in bulk",
                "{count} pending item(s) flagged for analyst override.",
                "⚠️",
            )
    else:
        st.info("No items in the review queue. Classify a product first or use Bulk Upload.")

elif page == "Audit Trail":
    st.title("Audit Trail")

    seed_logs = st.session_state["seed_logs"]

    session_logs = st.session_state["audit_log"]
    all_logs = seed_logs + session_logs
    if all_logs:
        logs = (
            pd.DataFrame(all_logs)
            .sort_values("Timestamp")
            .reset_index(drop=True)
        )
    else:
        logs = pd.DataFrame(columns=["Timestamp", "Event"])
    st.dataframe(logs, use_container_width=True)
    st.download_button(
        "Download Audit Log CSV",
        data=logs.to_csv(index=False).encode("utf-8-sig"),
        file_name="audit_log.csv",
        mime="text/csv",
    )
