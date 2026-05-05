import math
from collections import Counter
import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="HS & Shipment Pre-Check", layout="wide")

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


def _normalise_value(value) -> float:
    """Convert value to a finite, non-negative float rounded to pence."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v) or v < 0.0:
        return 0.0
    return round(v, 2)


def classify_product(description, material, origin, category, value):
    """Normalise inputs then delegate to the cached implementation."""
    return _classify_product_cached(
        (description or "").strip().lower(),
        (material or "").strip().lower(),
        (origin or "").strip().upper(),
        (category or "").strip().lower(),
        _normalise_value(value),
    )


@st.cache_data
def _classify_product_cached(desc, material_lower, origin_upper, category_lower, value):
    # value is already rounded to pence and guaranteed finite by _normalise_value
    high_value = value >= HIGH_VALUE_THRESHOLD
    hv_note = " High declared value flagged for additional customs scrutiny." if high_value else ""

    if ("scarf" in desc or "scarves" in desc) and ("silk" in material_lower or "silk" in desc):
        return {
            "hs6": "621410",
            "uk_code": "6214100090",
            "confidence": 0.94,
            "risk": RISK_RED if high_value else RISK_GREEN,
            "duty": "8%",
            "vat": "20%",
            "explanation": "Classified under silk scarves based on material composition and accessory type." + hv_note,
        }
    elif (
        "bag" in desc or "purse" in desc
        or category_lower == "bags"
    ) and ("leather" in material_lower or "leather" in desc):
        return {
            "hs6": "420221",
            "uk_code": "4202210000",
            "confidence": 0.88,
            "risk": RISK_RED if high_value else RISK_AMBER,
            "duty": "16%",
            "vat": "20%",
            "explanation": "Classified under handbags with outer surface of leather." + hv_note,
        }
    elif "perfume" in desc or "eau de parfum" in desc or category_lower == "beauty":
        return {
            "hs6": "330300",
            "uk_code": "3303001000",
            "confidence": 0.81,
            "risk": RISK_RED if high_value else RISK_AMBER,
            "duty": "6.5%",
            "vat": "20%",
            "explanation": "Classified under perfumes and toilet waters; regulated cosmetics handling required." + hv_note,
        }
    elif category_lower == "food" or any(
        w in desc for w in ("chocolate", "biscuit", "candy", "confection", "snack")
    ):
        return {
            "hs6": "210690",
            "uk_code": "2106909900",
            "confidence": 0.65,
            "risk": RISK_RED if high_value else RISK_AMBER,
            "duty": "varies",
            "vat": "20%",
            "explanation": (
                "Classified under miscellaneous food preparations; phytosanitary and food safety checks required."
                " Note: confectionery (chocolate, biscuits, candy) is standard-rated at 20% VAT in the UK." + hv_note
            ),
        }
    elif category_lower == "fashion_accessories" or any(
        w in desc for w in ("belt", "wallet", "glove", "hat", "cap", "tie", "brooch")
    ):
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
            "vat": "20%",
            "explanation": "Insufficient structured data; manual review recommended." + hv_note,
        }


def _safe_str(v) -> str:
    """Convert a value to string, returning empty string for NaN/None."""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v)


def classify_row(row):
    """Apply classify_product to a DataFrame row; safe for use with df.apply()."""
    raw_val = row.get("value")
    val = _normalise_value(raw_val)
    val_warning = ""
    if val == 0.0:
        if raw_val is None:
            val_warning = " Warning: declared value was missing; defaulted to £0 for risk assessment."
        else:
            try:
                parsed = float(raw_val)
                if not math.isfinite(parsed) or parsed < 0.0:
                    val_warning = " Warning: declared value was missing or invalid; defaulted to £0 for risk assessment."
            except (ValueError, TypeError):
                val_warning = " Warning: declared value could not be parsed; defaulted to £0 for risk assessment."
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
        return pd.Series({
            "hs6": ERROR_CODE,
            "uk_code": ERROR_CODE,
            "confidence": 0.0,
            "risk": RISK_AMBER,
            "duty": "TBD",
            "vat": "20%",
            "explanation": f"Classification failed: {str(e)[:200]}",
        })


def _add_to_review_queue(result: dict):
    """Add a classified item to the review queue if not already present.

    Deduplicates on (description, value, uk_code) so that re-clicking the
    button for the same product does not create duplicate queue entries, but
    a genuine reclassification that produces a different code is still added.
    """
    safe_val = _normalise_value(result.get("value", 0.0))
    key = (result["description"], safe_val, result["uk_code"])
    if key not in st.session_state["review_keys"]:
        st.session_state["review_keys"].add(key)
        st.session_state["review_items"].append({
            "Product": result["description"],
            "Suggested Code": result["uk_code"],
            "Confidence": f'{min(100, max(0, round(result["confidence"] * 100)))}%',
            "Explanation": result["explanation"],
            "Risk": result["risk"],
            "Status": STATUS_PENDING,
        })


# Initialise session state keys once so all pages can rely on them existing
st.session_state.setdefault("review_items", [])
st.session_state.setdefault("review_keys", set())
st.session_state.setdefault("audit_log", [])
st.session_state.setdefault("bulk_result", None)
st.session_state.setdefault("_bulk_file_id", None)
st.session_state.setdefault("last_result", None)

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
        risk_df = pd.DataFrame({"Risk": ["GREEN", "AMBER", "RED"], "Count": [9710, 2140, 600]})
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
                orig_clean = origin.strip()
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
        c.metric("Confidence", f'{round(r["confidence"] * 100)}%')

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
        file_id = uploaded.file_id
        if st.session_state["_bulk_file_id"] != file_id:
            try:
                # Read one extra row so len(df) > 5000 can detect oversized files
                df = pd.read_csv(uploaded, nrows=5001, encoding="utf-8-sig", encoding_errors="replace")
                df.columns = df.columns.str.strip().str.lower()
                # Warn if any cell contains the Unicode replacement character,
                # which indicates bytes that could not be decoded from the file's encoding.
                str_cols = df.select_dtypes(include=["object"])
                if not str_cols.empty and str_cols.apply(
                    lambda col: col.astype(str).str.contains("�", regex=False).any()
                ).any():
                    st.warning(
                        "Some characters in the CSV could not be decoded and have been "
                        "replaced with �. Re-save the file as UTF-8 to ensure accurate "
                        "classification."
                    )
            except pd.errors.ParserError:
                st.error("CSV format is invalid — check that columns are comma-separated and the file is UTF-8 encoded.")
                st.stop()
            except Exception as e:
                st.error(f"Failed to read file: {e}")
                st.stop()

            if len(df) > 5000:
                st.error(f"CSV exceeds the 5,000-row limit (at least {len(df):,} rows found). Split the file and re-upload.")
                st.stop()

            if df.empty:
                st.warning("The uploaded CSV contains no data rows.")
                st.stop()

            required = {"description", "material", "origin", "category", "value"}
            missing = required - set(df.columns)
            if missing:
                st.error(f"Missing required columns: {', '.join(sorted(missing))}")
                st.stop()

            # Warn if pre-existing result columns will be overwritten
            overlapping = sorted(col for col in RESULT_COLUMNS if col in df.columns)
            if overlapping:
                st.warning(f"The following columns from your CSV will be overwritten by classification results: {', '.join(overlapping)}")
            # Drop any pre-existing result columns to avoid duplicate columns after concat
            input_df = df.drop(columns=overlapping).reset_index(drop=True)
            try:
                with st.spinner(f"Classifying {len(input_df)} rows…"):
                    result_df = pd.concat(
                        [input_df, input_df.apply(classify_row, axis=1)],
                        axis=1,
                    )
            except Exception as e:
                st.error(f"Classification failed: {e}")
                st.stop()

            error_count = int((result_df["hs6"] == ERROR_CODE).sum())
            unclassified_count = int((result_df["hs6"] == UNCLASSIFIED_CODE).sum())
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
                "Event": f"Bulk upload: {summary} from '{uploaded.name}'",
            })
            st.session_state["bulk_result"] = {
                "df": result_df,
                "summary": summary,
                "filename": uploaded.name,
            }

            for row in result_df.to_dict("records"):
                if row.get("hs6") not in (ERROR_CODE, UNCLASSIFIED_CODE):
                    raw_conf = row.get("confidence", 0.0)
                    try:
                        conf = float(raw_conf)
                        conf = max(0.0, min(1.0, conf)) if math.isfinite(conf) else 0.0
                    except (ValueError, TypeError):
                        conf = 0.0
                    _add_to_review_queue({
                        "description": str(row.get("description", "")),
                        "value": _normalise_value(row.get("value", 0.0)),
                        "uk_code": str(row.get("uk_code", "")),
                        "confidence": conf,
                        "explanation": str(row.get("explanation", "")),
                        "risk": str(row.get("risk", RISK_AMBER)),
                    })

            st.session_state["_bulk_file_id"] = file_id

    bulk = st.session_state["bulk_result"]
    if bulk is not None:
        st.success(bulk["summary"])
        st.dataframe(bulk["df"], use_container_width=True)
        st.download_button(
            "Download Results CSV",
            data=bulk["df"].to_csv(index=False).encode("utf-8"),
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
            ts = datetime.now().isoformat(timespec="microseconds")
            count = len(items)
            st.session_state["review_items"] = [
                {**item, "Status": STATUS_APPROVED} for item in items
            ]
            st.session_state["audit_log"].append({
                "Timestamp": ts,
                "Event": f"Review Queue: {count} item(s) approved in bulk",
            })
            st.toast("All items marked as approved.", icon="✅")
            st.rerun()

        if col2.button("Override All"):
            ts = datetime.now().isoformat(timespec="microseconds")
            count = len(items)
            st.session_state["review_items"] = [
                {**item, "Status": STATUS_OVERRIDDEN} for item in items
            ]
            st.session_state["audit_log"].append({
                "Timestamp": ts,
                "Event": f"Review Queue: {count} item(s) flagged for analyst override in bulk",
            })
            st.toast("All items flagged for analyst override.", icon="⚠️")
            st.rerun()
    else:
        st.info("No items in the review queue. Classify a product first or use Bulk Upload.")

elif page == "Audit Trail":
    st.title("Audit Trail")

    # Initialise seed logs once per session so they don't regenerate on every rerun
    if "seed_logs" not in st.session_state:
        today = datetime.now().strftime("%Y-%m-%d")
        st.session_state["seed_logs"] = [
            {"Timestamp": f"{today}T09:12:00.000000", "Event": "SKU123 classified as 6214100090 by system"},
            {"Timestamp": f"{today}T09:17:00.000000", "Event": "Reviewed by compliance_officer_01"},
            {"Timestamp": f"{today}T09:18:00.000000", "Event": "Approved and published to product master"},
        ]
    seed_logs = st.session_state["seed_logs"]

    session_logs = st.session_state["audit_log"]
    logs = (
        pd.DataFrame(seed_logs + session_logs)
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )
    st.dataframe(logs, use_container_width=True)
