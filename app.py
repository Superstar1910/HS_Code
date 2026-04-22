
import math
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

# Columns produced by classify_product — used to drop conflicts before bulk concat
RESULT_COLUMNS = frozenset({"hs6", "uk_code", "confidence", "risk", "duty", "vat", "explanation"})

# Sentinel values used in result rows
ERROR_CODE = "ERROR"
UNCLASSIFIED_CODE = "UNCLASSIFIED"


@st.cache_data
def classify_product(description, material, origin, category, value):
    desc = (description or "").strip().lower()
    material_lower = (material or "").strip().lower()
    category_lower = (category or "").strip().lower()

    # High-value items attract additional customs scrutiny
    # Round to pence to avoid floating-point edge cases near the threshold
    high_value = round(value, 2) >= HIGH_VALUE_THRESHOLD
    hv_note = " High declared value flagged for additional customs scrutiny." if high_value else ""

    if ("scarf" in desc or "scarves" in desc) and "silk" in material_lower:
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
        "bag" in desc or "handbag" in desc or "purse" in desc
        or category_lower == "bags"
    ) and "leather" in material_lower:
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
        w in desc for w in ("chocolate", "biscuit", "candy", "confection", "food", "snack")
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
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v)


def classify_row(row):
    """Apply classify_product to a DataFrame row; safe for use with df.apply()."""
    val_warning = ""
    try:
        val = float(row["value"])
        if not math.isfinite(val) or val < 0.0:
            val = 0.0
            val_warning = " Warning: declared value was missing or invalid; defaulted to £0 for risk assessment."
    except (ValueError, TypeError, KeyError):
        val = 0.0
        val_warning = " Warning: declared value could not be parsed; defaulted to £0 for risk assessment."
    try:
        result = classify_product(
            _safe_str(row["description"]),
            _safe_str(row["material"]),
            _safe_str(row["origin"]),
            _safe_str(row["category"]),
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
            "explanation": f"Classification failed: {str(e)}",
        })


def _add_to_review_queue(result: dict):
    """Add a classified item to the review queue if not already present.

    Deduplicates on (description, value, uk_code) so that re-clicking the
    button for the same product does not create duplicate queue entries, but
    a genuine reclassification that produces a different code is still added.
    """
    key = (result["description"], result.get("value", ""), result["uk_code"])
    if key not in st.session_state["review_keys"]:
        st.session_state["review_keys"].add(key)
        st.session_state["review_items"].append({
            "Product": result["description"],
            "Suggested Code": result["uk_code"],
            "Confidence": f'{round(result["confidence"] * 100)}%',
            "Explanation": result["explanation"],
            "Risk": result["risk"],
            "Status": "Pending review",
        })


# Initialise session state keys once so all pages can rely on them existing
st.session_state.setdefault("review_items", [])
st.session_state.setdefault("review_keys", set())
st.session_state.setdefault("audit_log", [])
st.session_state.setdefault("bulk_result", None)

st.sidebar.title("HS & Shipment Pre-Check")
page = st.sidebar.radio("Navigate", ["Dashboard", "Classify", "Bulk Upload", "Review Queue", "Audit Trail"])

if page == "Dashboard":
    st.title("HS & Shipment Pre-Check Dashboard")

    session_items = st.session_state["review_items"]
    session_total = len(session_items)
    session_pending = sum(1 for i in session_items if i["Status"] == "Pending review")
    session_approved = sum(1 for i in session_items if i["Status"] == "Approved")
    session_overridden = sum(1 for i in session_items if "Overridden" in i["Status"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Session SKUs", session_total if session_total else "—")
    c2.metric("Pending Review", session_pending if session_total else "—")
    c3.metric("Approved", session_approved if session_total else "—")
    c4.metric("Overridden", session_overridden if session_total else "—")

    st.caption("Metrics reflect classifications performed in this session.")

    if session_items:
        st.subheader("Session Risk Distribution")
        risk_counts = {RISK_GREEN: 0, RISK_AMBER: 0, RISK_RED: 0}
        for i in session_items:
            risk = i["Risk"]
            if risk in risk_counts:
                risk_counts[risk] += 1
        risk_df = pd.DataFrame({"Risk": list(risk_counts.keys()), "Count": list(risk_counts.values())})
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

    if "last_result" in st.session_state:
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
        try:
            # Read one extra row so len(df) > 5000 can detect oversized files
            df = pd.read_csv(uploaded, nrows=5001, encoding_errors="replace")
            df.columns = df.columns.str.strip().str.lower()
            # Warn if any string column contains the Unicode replacement character,
            # which indicates bytes that could not be decoded from the file's encoding.
            str_cols = df.select_dtypes(include="object").columns
            if any(
                df[col].astype(str).str.contains("\ufffd", regex=False).any()
                for col in str_cols
            ):
                st.warning(
                    "Some characters in the CSV could not be decoded and have been "
                    "replaced with \ufffd. Re-save the file as UTF-8 to ensure accurate "
                    "classification."
                )
        except pd.errors.ParserError:
            st.error("CSV format is invalid — check that columns are comma-separated and the file is UTF-8 encoded.")
            st.stop()
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            st.stop()

        if len(df) > 5000:
            st.error("CSV exceeds the 5,000-row limit. Split the file and re-upload.")
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
        input_df = df.drop(columns=overlapping)
        try:
            with st.spinner(f"Classifying {len(input_df)} rows…"):
                result_df = pd.concat(
                    [input_df.reset_index(drop=True), input_df.apply(classify_row, axis=1)],
                    axis=1,
                )
        except Exception as e:
            st.error(f"Classification failed: {e}")
            st.stop()

        error_count = int((result_df["hs6"] == ERROR_CODE).sum())
        unclassified_count = int((result_df["hs6"] == UNCLASSIFIED_CODE).sum())
        parts = [f"Processed {len(result_df)} rows"]
        if unclassified_count:
            parts.append(f"{unclassified_count} unclassified")
        if error_count:
            parts.append(f"{error_count} errors")
        st.session_state["audit_log"].append({
            "Timestamp": datetime.now().isoformat(timespec="microseconds"),
            "Event": f"Bulk upload processed {len(result_df)} rows from '{uploaded.name}'" + (f" ({error_count} errors)" if error_count else ""),
        })
        st.session_state["bulk_result"] = {
            "df": result_df,
            "summary": " — ".join(parts),
            "filename": uploaded.name,
        }

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
        review_df = pd.DataFrame(items)[display_cols]
        st.dataframe(review_df, use_container_width=True)

        st.write("**Manual review actions**")
        col1, col2 = st.columns(2)

        if col1.button("Approve All"):
            ts = datetime.now().isoformat(timespec="microseconds")
            count = len(items)
            st.session_state["review_items"] = [
                {**item, "Status": "Approved"} for item in items
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
                {**item, "Status": "Overridden — pending analyst"} for item in items
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
            {"Timestamp": f"{today}T09:12:00", "Event": "SKU123 classified as 6214100090 by system"},
            {"Timestamp": f"{today}T09:17:00", "Event": "Reviewed by compliance_officer_01"},
            {"Timestamp": f"{today}T09:18:00", "Event": "Approved and published to product master"},
        ]
    seed_logs = st.session_state["seed_logs"]

    session_logs = st.session_state["audit_log"]
    logs = (
        pd.DataFrame(seed_logs + session_logs)
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )
    st.dataframe(logs, use_container_width=True)
