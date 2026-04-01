
import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="HS Classification Demo", layout="wide")

def classify_product(description, material, origin, category, value):
    desc = (description or "").lower()
    material_l = (material or "").lower()

    if "scarf" in desc and "silk" in material_l:
        return {
            "hs6": "621410",
            "uk_code": "6214100090",
            "confidence": 0.94,
            "risk": "GREEN",
            "duty": "8%",
            "vat": "20%",
            "explanation": "Classified under silk scarves based on material composition and accessory type."
        }
    elif "bag" in desc and "leather" in material_l:
        return {
            "hs6": "420221",
            "uk_code": "4202210000",
            "confidence": 0.88,
            "risk": "AMBER",
            "duty": "16%",
            "vat": "20%",
            "explanation": "Classified under handbags with outer surface of leather."
        }
    elif "perfume" in desc or category == "beauty":
        return {
            "hs6": "330300",
            "uk_code": "3303001000",
            "confidence": 0.81,
            "risk": "RED",
            "duty": "6.5%",
            "vat": "20%",
            "explanation": "Classified under perfumes and toilet waters; flagged red due to regulated cosmetics handling."
        }
    else:
        return {
            "hs6": "000000",
            "uk_code": "0000000000",
            "confidence": 0.52,
            "risk": "AMBER",
            "duty": "TBD",
            "vat": "20%",
            "explanation": "Insufficient structured data; manual review recommended."
        }

st.sidebar.title("HS Classification Demo")
page = st.sidebar.radio("Navigate", ["Dashboard", "Classify", "Bulk Upload", "Review Queue", "Audit Trail"])

if page == "Dashboard":
    st.title("HS Classification Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total SKUs", "12,450")
    c2.metric("Auto-Classified", "78%")
    c3.metric("Pending Review", "1,120")
    c4.metric("Accuracy", "95.4%")

    st.subheader("Risk Distribution")
    risk_df = pd.DataFrame({
        "Risk": ["GREEN", "AMBER", "RED"],
        "Count": [9710, 2140, 600]
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
            result = classify_product(description, material, origin, category, value)
            st.session_state["last_result"] = {
                "description": description,
                "material": material,
                "origin": origin,
                "category": category,
                "value": value,
                **result
            }

    with right:
        st.info("This prototype uses rules-based mock logic for demo purposes.")

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
            "decision_timestamp": datetime.now().isoformat(timespec="seconds")
        })

elif page == "Bulk Upload":
    st.title("Bulk Upload")
    uploaded = st.file_uploader("Upload CSV with columns: description, material, origin, category, value", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        required = {"description", "material", "origin", "category", "value"}
        missing = required - set(df.columns)
        if missing:
            st.error(f"Missing required columns: {', '.join(sorted(missing))}")
        else:
            results = []
            for _, row in df.iterrows():
                out = classify_product(
                    str(row["description"]),
                    str(row["material"]),
                    str(row["origin"]),
                    str(row["category"]),
                    float(row["value"])
                )
                results.append(out)
            result_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
            st.success(f"Processed {len(result_df)} rows")
            st.dataframe(result_df, use_container_width=True)
            st.download_button(
                "Download Results CSV",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="hs_classification_results.csv",
                mime="text/csv"
            )
    else:
        st.caption("Use the sample CSV in the deployment bundle to test bulk processing.")

elif page == "Review Queue":
    st.title("Review Queue")
    review_df = pd.DataFrame({
        "Product": ["Silk Scarf", "Leather Bag", "Perfume"],
        "Suggested Code": ["6214100090", "4202210000", "3303001000"],
        "Confidence": ["94%", "88%", "81%"],
        "Risk": ["GREEN", "AMBER", "RED"],
        "Status": ["Auto-approved", "Needs analyst review", "Compliance sign-off"]
    })
    st.dataframe(review_df, use_container_width=True)
    st.write("**Manual review actions**")
    col1, col2 = st.columns(2)
    col1.button("Approve Selected")
    col2.button("Override Selected")

elif page == "Audit Trail":
    st.title("Audit Trail")
    logs = pd.DataFrame({
        "Timestamp": [
            datetime.now().strftime("%Y-%m-%d 09:12"),
            datetime.now().strftime("%Y-%m-%d 09:17"),
            datetime.now().strftime("%Y-%m-%d 09:18"),
        ],
        "Event": [
            "SKU123 classified as 6214100090 by system",
            "Reviewed by compliance_officer_01",
            "Approved and published to product master"
        ]
    })
    st.dataframe(logs, use_container_width=True)
