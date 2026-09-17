
# HS & Shipment Pre-Check

A Streamlit application for classifying products against UK commodity codes (HS codes), assessing customs risk, and managing a compliance review queue.

## Features

- **Single-item classification** — enter a description, material, origin, category, and declared value to get an HS6/UK commodity code, duty rate, VAT rate, confidence score, and risk level
- **Bulk upload** — process a CSV of up to 5,000 product rows in one go, with chunked progress reporting
- **Review queue** — editable table where analysts can approve, override, or clear classified items; supports bulk approve/override actions
- **Audit trail** — append-only log of all classification events, downloadable as CSV
- **Dashboard** — session-level metrics and risk distribution chart

## Files

| File | Description |
|---|---|
| `app.py` | Main Streamlit application (~1,700 lines) |
| `requirements.txt` | Python dependencies |
| `sample_products.csv` | 15-row test file covering all supported categories and edge cases |

## Deploy to Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload these files to the repo root.
3. Sign in to [Streamlit Community Cloud](https://streamlit.io/cloud).
4. Choose **New app**.
5. Select your repo, branch (`main`), and `app.py`.
6. Click **Deploy**.

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires Python 3.8+ (walrus operator and `from __future__ import annotations`).

## CSV format

Required columns (header names are **case-insensitive**):

| Column | Description | Example |
|---|---|---|
| `description` | Plain-text product name | `Luxury silk scarf` |
| `material` | Fibre / material composition | `100% silk` |
| `origin` | ISO 3166-1 alpha-2 country code | `IT` |
| `category` | See valid values below | `bags` |
| `value` | Declared customs value in GBP | `250.00` |

**Valid `category` values:** `fashion_accessories`, `bags`, `beauty`, `food`, `other`

The `value` column accepts most common formats: `250`, `1250.00`, `£1,250`, `GBP 1250`, `1.250,00` (European). Ambiguous formats (e.g., `1.250` — could be £1.25 or £1,250) are defaulted to £0 with a warning. Use `1250` or `1250.00` to avoid ambiguity.

## Architecture notes

- **Classification cache**: `_classify_product_cached` is decorated with `functools.lru_cache(maxsize=4096)`. Origin is excluded from the cache key (only affects the explanation note, not the code assignment), so products from different countries share entries.
- **Material-segment parsing**: material fields containing comma-, semicolon-, or forward-slash-separated segments (e.g., `"genuine leather outer; faux leather lining"`, `"80% leather / 20% suede"`) are checked per-segment so a faux qualifier in one segment does not suppress a genuine-material signal in another.
- **`types.MappingProxyType`**: cached results are returned as immutable proxy objects; `classify_product` makes a `dict` shallow-copy before appending the origin note, so the cache entry is never mutated.
- **Bulk processing**: rows are classified in chunks of `max(5, n // 100)` via `df.apply()` rather than per-row `iterrows()`, giving ~100 progress-bar updates for a 5,000-row file while avoiding per-row Series overhead.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| All items classified as UNCLASSIFIED | Category column is misspelled or missing | Use one of the five valid category values; unknown values fall back to keyword-only classification |
| Value column shows £0 with a warning | Ambiguous decimal/thousands format | Use unambiguous format: `1250` or `1250.00` |
| Encoding warning (replacement character `�`) | CSV saved with non-UTF-8 encoding | Re-save as UTF-8 in your spreadsheet app |
| `HIGH_VALUE_THRESHOLD` flag unexpected | Declared value ≥ £1,000 | Verify declared value; HIGH risk label is added automatically above this threshold |
