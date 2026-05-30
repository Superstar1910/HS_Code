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

_CONFECTIONERY_WORDS = ("chocolate", "chocolates", "biscuit", "biscuits", "candy", "candies", "confection", "confections", "snack", "snacks")
_FASHION_WORDS = (
    "belt", "belts", "wallet", "wallets", "glove", "gloves",
    "hat", "hats", "cap", "caps", "tie", "ties",
    "brooch", "brooches", "scarf", "scarves",
)
_BAG_WORDS = (
    "bag", "bags", "handbag", "handbags", "purse", "purses",
    "tote", "totes", "clutch", "satchel",
    "backpack", "backpacks", "rucksack", "rucksacks",
    "briefcase", "briefcases",
)


@functools.lru_cache(maxsize=None)
def _word_pattern(word: str) -> re.Pattern:
    return re.compile(r'\b' + re.escape(word) + r'\b')


def _word_in_text(word: str, text: str) -> bool:
    """Return True if word appears as a whole word in text."""
    return bool(_word_pattern(word).search(text))

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

# Categories that suppress food classification even when confectionery keywords match.
# "other" is included because it signals an unrecognised category, not food.
_NON_FOOD_CATEGORIES = frozenset({"bags", "beauty", "fashion_accessories", "other"})


def _parse_value(raw) -> tuple[float, str]:
    """Convert raw value to (normalised_float, warning_message).

    The warning is non-empty only when the raw input was absent or invalid
    and has been defaulted to 0.0.
    """
    try:
        v = float(raw)
    except (TypeError, ValueError):
        try:
            is_missing = pd.isna(raw)
        except Exception:
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
    v = _normalise_value(value)
    # Return a shallow copy so callers cannot mutate the lru_cache entry.
    return dict(_classify_product_cached(
        (description or "").strip().lower(),
        (material or "").strip().lower(),
        (origin or "").strip().upper(),
        (category or "").strip().lower(),
        v >= HIGH_VALUE_THRESHOLD,
    ))


@functools.lru_cache(maxsize=4096)
def _classify_product_cached(desc, material_lower, origin_upper, category_lower, high_value):
    # high_value is a bool; using it instead of the raw value means products that
    # share the same description/material/origin/category and the same high-value
    # status hit the same cache entry regardless of exact declared price.
    hv_note = " High declared value flagged for additional customs scrutiny." if high_value else ""
    origin_note = (
        f" Country of origin: {origin_upper}."
        if origin_upper
        else " Warning: country of origin not declared — required for customs clearance."
    )

    # Pre-compute all keyword flags once to avoid redundant regex evaluation.
    is_scarf = _word_in_text("scarf", desc) or _word_in_text("scarves", desc)
    is_silk = _word_in_text("silk", material_lower) or _word_in_text("silk", desc)
    is_leather = _word_in_text("leather", material_lower) or _word_in_text("leather", desc)
    # "fragrance-free" explicitly negates the product being a fragrance;
    # guard against \bfragrance\b matching that compound adjective.
    is_perfume = (
        _word_in_text("perfume", desc) or _word_in_text("perfumes", desc)
        or ("fragrance-free" not in desc
            and (_word_in_text("fragrance", desc) or _word_in_text("fragrances", desc)))
        or _word_in_text("cologne", desc) or _word_in_text("colognes", desc)
        or _word_in_text("aftershave", desc)
        or "eau de parfum" in desc
        or "eau de toilette" in desc
        or "eau de cologne" in desc
        or category_lower == "beauty"
    )
    is_confectionery = any(_word_in_text(w, desc) for w in _CONFECTIONERY_WORDS)
    # Confectionery keywords only drive food classification when the category does
    # not indicate a different product type; prevents "chocolate leather wallet"
    # from being misclassified as food when category == "fashion_accessories".
    is_food = category_lower == "food" or (
        is_confectionery and category_lower not in _NON_FOOD_CATEGORIES
    )
    is_fashion = category_lower == "fashion_accessories" or any(_word_in_text(w, desc) for w in _FASHION_WORDS)
    # Bag detection: an explicit fashion_accessories category overrides bag keywords
    # (a "handbag charm" is an accessory, not a bag); category="bags" only fires
    # when description keywords do not indicate a fashion accessory, preventing
    # items like belts or scarves from being misrouted to bag HS codes due to a
    # miscategorised or imprecise category field.
    _bag_keyword = any(_word_in_text(w, desc) for w in _BAG_WORDS)
    is_bag = (_bag_keyword and category_lower != "fashion_accessories") or (category_lower == "bags" and not is_fashion)

    if is_scarf and is_silk:
        return {
            "hs6": "621410",
            "uk_code": "6214100090",
            "confidence": 0.94,
            "risk": RISK_RED if high_value else RISK_GREEN,
            "duty": "8%",
            "vat": "20%",
            "explanation": "Classified under silk scarves based on material composition and accessory type." + origin_note + hv_note,
        }
    elif is_bag and is_leather:
        return {
            "hs6": "420221",
            "uk_code": "4202210000",
            "confidence": 0.88,
            "risk": RISK_RED if high_value else RISK_AMBER,
            "duty": "16%",
            "vat": "20%",
            "explanation": "Classified under handbags with outer surface of leather." + origin_note + hv_note,
        }
    elif is_bag:
        return {
            "hs6": "420229",
            "uk_code": "4202290000",
            "confidence": 0.65,
            "risk": RISK_RED if high_value else RISK_AMBER,
            "duty": "3.7%",
            "vat": "20%",
            "explanation": "Classified under handbags with other outer surface; verify material composition for precise subheading." + origin_note + hv_note,
        }
    elif is_perfume:
        return {
            "hs6": "330300",
            "uk_code": "3303001000",
            "confidence": 0.81,
            "risk": RISK_RED if high_value else RISK_AMBER,
            "duty": "6.5%",
            "vat": "20%",
            "explanation": "Classified under perfumes and toilet waters; regulated cosmetics handling required." + origin_note + hv_note,
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
                + vat_note + origin_note + hv_note
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
            "explanation": "Classified under other made-up clothing accessories; verify composition for precise subheading." + origin_note + hv_note,
        }
    else:
        return {
            "hs6": UNCLASSIFIED_CODE,
            "uk_code": UNCLASSIFIED_CODE,
            "confidence": 0.0,
            "risk": RISK_RED if high_value else RISK_AMBER,
            "duty": "TBD",
            "vat": "TBD",
            "explanation": "Insufficient structured data; manual review recommended." + origin_note + hv_note,
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
    val, val_warning = _parse_value(row.get("value"))
    try:
        result = classify_product(
            _safe_str(row.get("description", "")),
            _safe_str(row.get("material", "")),
            _safe_str(row.get("origin", "")),
            _safe_str(row.get("category", "")),
            val,
        )
        if val_warning:
            result = {**result, "explanation": result["explanation"] + val_warning}
        return pd.Series(result)
    except Exception as e:
        row_idx = getattr(row, "name", None)
        # hasattr(__index__) covers both Python int and numpy integer scalars.
        display_idx = (row_idx + 1) if hasattr(row_idx, "__index__") else row_idx
        prefix = f"Row {display_idx}: " if display_idx is not None else ""
        msg = f"{prefix}Classification failed: {type(e).__name__}: {str(e)}"
        suffix = val_warning or ""
        msg_budget = 250 - len(suffix)
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
    """
    safe_val = _normalise_value(result.get("value", 0.0))
    # Use the high-value boolean rather than the raw amount: classification only
    # distinguishes values by whether they meet HIGH_VALUE_THRESHOLD, so two
    # sub-threshold prices for the same product produce the same classification
    # and should map to the same dedup key.
    key = (result.get("description", ""), safe_val >= HIGH_VALUE_THRESHOLD, result.get("uk_code", ""))
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
    count = 0
    for item in st.session_state["review_items"]:
        if item["Status"] == STATUS_PENDING:
            # Never auto-approve items with no assigned code; they require manual
            # code entry, not a sign-off.
            if new_status == STATUS_APPROVED and item.get("Suggested Code") == UNCLASSIFIED_CODE:
                continue
            item["Status"] = new_status
            count += 1
    st.session_state["audit_log"].append({"Timestamp": ts, "Event": audit_event.format(count=count)})
    st.toast(toast_msg.format(count=count), icon=toast_icon)
    st.rerun()


def _process_bulk_upload(file_bytes: bytes, filename: str, file_id: tuple[str, str]) -> None:
    """Validate, classify, and store results for a newly uploaded CSV.

    Accepts raw bytes so the function is independent of the UploadedFile
    cursor position and can be called without side-effects on the file object.
    Uses return-on-error instead of st.stop() so the caller can still render
    any previously stored bulk results after a failed upload attempt.
    """
    # Mark the file as seen immediately so Streamlit reruns (triggered by
    # widgets elsewhere on the page) don't re-process the same file.
    st.session_state["_bulk_file_id"] = file_id
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
            lambda col: col.str.contains("�", regex=False, na=False).any()
        ).any():
            st.session_state["_bulk_messages"].append(("warning", (
                "Some characters in the CSV could not be decoded and have been "
                "replaced with �. Re-save the file as UTF-8 to ensure accurate "
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
            classified = input_df.apply(classify_row, axis=1).reset_index(drop=True)
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
        detail_parts.append(f"{error_count} errors")
    summary = f"Processed {len(result_df)} rows"
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

    for row in result_df.to_dict("records"):
        if row.get("hs6") != ERROR_CODE:
            _add_to_review_queue({
                "description": str(row.get("description", "")),
                "value": row.get("value", 0.0),
                "uk_code": str(row.get("uk_code", "")),
                "confidence": row.get("confidence", 0.0),
                "explanation": str(row.get("explanation", "")),
                "risk": str(row.get("risk", RISK_AMBER)),
            })


# Initialise session state keys once so all pages can rely on them existing.
st.session_state.setdefault("review_items", [])
st.session_state.setdefault("review_keys", set())
st.session_state.setdefault("audit_log", [])
st.session_state.setdefault("bulk_result", None)
st.session_state.setdefault("_bulk_file_id", None)
st.session_state.setdefault("_bulk_messages", [])
st.session_state.setdefault("last_result", None)
_today = datetime.now().strftime("%Y-%m-%d")
st.session_state.setdefault("seed_logs", [
    {"Timestamp": f"{_today}T09:12:00.000000", "Event": "SKU123 classified as 6214100090 by system"},
    {"Timestamp": f"{_today}T09:17:00.000000", "Event": "Reviewed by compliance_officer_01"},
    {"Timestamp": f"{_today}T09:18:00.000000", "Event": "Approved and published to product master"},
])

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
                result = classify_product(desc_clean, mat_clean, orig_clean, category, value)
                entry = {
                    "description": desc_clean,
                    "material": mat_clean,
                    "origin": orig_clean,
                    "category": category,
                    "value": value,
                    "timestamp": datetime.now().isoformat(timespec="microseconds"),
                    **result,
                }
                st.session_state["last_result"] = entry
                _add_to_review_queue(entry)
                st.session_state["audit_log"].append({
                    "Timestamp": entry["timestamp"],
                    "Event": f'"{entry["description"]}" classified as {entry["uk_code"]} (risk: {entry["risk"]})',
                })

    with right:
        st.info(
            "Check if your product description is customs-ready before shipment. "
            "Detect missing data, improve descriptions, and reduce shipment rejection risk."
        )

    if st.session_state["last_result"] is not None:
        r = st.session_state["last_result"]
        st.subheader("Classification Result")
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
        file_id = (uploaded.name, hashlib.md5(raw_bytes, usedforsecurity=False).hexdigest())
        if st.session_state["_bulk_file_id"] != file_id:
            _process_bulk_upload(raw_bytes, uploaded.name, file_id)
    else:
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
        st.dataframe(review_df, use_container_width=True)

        st.write("**Manual review actions**")
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
        st.dataframe(logs, use_container_width=True)
    else:
        st.info("No audit events recorded yet.")
