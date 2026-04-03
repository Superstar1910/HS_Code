
import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="HS & Shipment Pre-Check", layout="wide")

# Columns produced by classify_product — used to drop conflicts before bulk concat
RESULT_COLUMNS = {"hs6", "uk_code", "confidence", "risk", "duty", "vat", "explanation"}


def classify_product(description, material, origin, category, value):
    desc = (description or "").strip().lower()
    material_lower = (material or "").strip().lower()
    category_lower = (category or "").strip().lower()

    if "scarf" in desc and "silk" in material_lower:
        return {
            "hs6": "621410",
            "uk_code": "6214100090",
            "confidence": 0.94,
            "risk": "GREEN",
            "duty": "8%",
            "vat": "20%",
            "explanation": "Classified under silk scarves based on material composition and accessory type.",
        }
    elif "bag" in desc and "leather" in material_lower:
        return {
            "hs6": "420221",
            "uk_code": "4202210000",
            "confidence": 0.88,
            "risk": "AMBER",
            "duty": "16%",
            "vat": "20%",
            "explanation": "Classified under handbags with outer surface of leather.",
        }
    elif "perfume" in desc or "eau de parfum" in desc or category_lower == "beauty":
        return {
            "hs6": "330300",
            "uk_code": "3303001000",
            "confidence": 0.81,
            "risk": "RED",
            "duty": "6.5%",
            "vat": "20%",
            "explanation": "Classified under perfumes and toilet waters; flagged red due to regulated cosmetics handling.",
        }
    else:
        return {
            "hs6": "UNCLASSIFIED",
            "uk_code": "UNCLASSIFIED",
            "confidence": 0.52,
            "risk": "AMBER",
            "duty": "TBD",
            "vat": "20%",
            "explanation": "Insufficient structured data; manual review recommended.",
        }


def classify_row(row):
    """Apply classify_product to a DataFrame row; safe for use with df.apply()."""
    try:
        val = float(row["value"])
    except (ValueError, TypeError):
        val = 0.0
    return pd.Series(classify_product(
        str(row["description"]),
        str(row["material"]),
        str(row["origin"]),
        str(row["category"]),
        val,
    ))


st.sidebar.title("HS & Shipment Pre-Check")
page = st.sidebar.radio("Navigate", ["Dashboard", "Classify", "Bulk Upload", "Review Queue", "Audit Trail"])

if page == "Dashboard":
    st.title("HS & Shipment Pre-Check Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total SKUs", "12,450")
    c2.metric("Auto-Classified", "78%")
    c3.metric("Pending Review", "1,120")
    c4.metric("Accuracy", "95.4%")

    st.subheader("Risk Distribution")
    risk_df = pd.DataFrame({
        "Risk": ["GREEN", "AMBER", "RED"],
        "Count": [9710, 2140, 600],
    })
    st.bar_chart(risk_df.set_index("Risk"))

elif page == "Classify":
    st.title("Classify Product")

    left, right = st.columns([2, 1])

    with left:
        description = st.text_input("Product Description", "Luxury silk scarf with hand-rolled edges")
        material = st.text_input("Material Composition", "100% silk")
        origin = st.text_input("Country of Origin", "IT")
        category = st.selectbox("Category", ["fashion_accessories", "bags", "beauty", "food", "other"])
        value = st.number_input("Declared Value (£)", min_value=0.0, value=250.0, step=10.0)

        if st.button("Run Classification"):
            if not description.strip():
                st.warning("Please enter a product description before classifying.")
            else:
                result = classify_product(description, material, origin, category, value)
                st.session_state["last_result"] = {
                    "description": description.strip(),
                    "material": material.strip(),
                    "origin": origin.strip(),
                    "category": category,
                    "value": value,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    **result,
                }

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
        c.metric("Confidence", f'{int(r["confidence"] * 100)}%')

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
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to parse CSV: {e}")
            st.stop()

        required = {"description", "material", "origin", "category", "value"}
        missing = required - set(df.columns)
        if missing:
            st.error(f"Missing required columns: {', '.join(sorted(missing))}")
        else:
            # Drop any pre-existing result columns to avoid duplicate columns after concat
            input_df = df.drop(columns=[c for c in RESULT_COLUMNS if c in df.columns])
            result_df = pd.concat(
                [input_df.reset_index(drop=True), input_df.apply(classify_row, axis=1)],
                axis=1,
            )
            st.success(f"Processed {len(result_df)} rows")
            st.dataframe(result_df, use_container_width=True)
            st.download_button(
                "Download Results CSV",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="hs_classification_results.csv",
                mime="text/csv",
            )
    else:
        st.caption("Use the sample CSV in the deployment bundle to test bulk processing.")

elif page == "Review Queue":
    st.title("Review Queue")

    if "review_df" not in st.session_state:
        st.session_state["review_df"] = pd.DataFrame({
            "Product": ["Silk Scarf", "Leather Bag", "Perfume"],
            "Suggested Code": ["6214100090", "4202210000", "3303001000"],
            "Confidence": ["94%", "88%", "81%"],
            "Risk": ["GREEN", "AMBER", "RED"],
            "Status": ["Auto-approved", "Needs analyst review", "Compliance sign-off"],
        })

    st.dataframe(st.session_state["review_df"], use_container_width=True)
    st.write("**Manual review actions**")
    col1, col2 = st.columns(2)

    if col1.button("Approve All"):
        st.session_state["review_df"]["Status"] = "Approved"
        st.success("All items marked as approved.")
        st.rerun()

    if col2.button("Override All"):
        st.session_state["review_df"]["Status"] = "Overridden — pending analyst"
        st.warning("All items flagged for analyst override.")
        st.rerun()

elif page == "Audit Trail":
    st.title("Audit Trail")

    # Use fixed timestamps so they don't shift on every rerun
    today = datetime.now().strftime("%Y-%m-%d")
    logs = pd.DataFrame({
        "Timestamp": [
            f"{today} 09:12:00",
            f"{today} 09:17:00",
            f"{today} 09:18:00",
        ],
        "Event": [
            "SKU123 classified as 6214100090 by system",
            "Reviewed by compliance_officer_01",
            "Approved and published to product master",
        ],
    })
    st.dataframe(logs, use_container_width=True)
