from __future__ import annotations

import functools
import hashlib
import io
import math
import re
from collections import Counter
import numpy as np
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="HS & Shipment Pre-Check", layout="wide")

def _make_word_re(*words: str) -> re.Pattern[str]:
    """Return a compiled whole-word alternation regex for the given keywords."""
    return re.compile(r'\b(?:' + '|'.join(re.escape(w) for w in words) + r')\b')

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

# Pre-compiled alternation patterns for keyword groups used in classification.
_CONFECTIONERY_RE = _make_word_re(
    "chocolate", "chocolates", "biscuit", "biscuits",
    "candy", "candies", "confection", "confections", "confectionery",
    "snack", "snacks", "cookie", "cookies",
    "sweets", "toffee", "toffees", "fudge",
    "lollipop", "lollipops",
    # Additional UK confectionery terms standard-rated at 20% VAT:
    "gummy", "gummies", "marshmallow", "marshmallows",
    "nougat", "nougats", "marzipan", "marzipans", "sherbet", "sherbets", "praline", "pralines",
    "truffle", "truffles", "bonbon", "bonbons",
    "licorice", "liquorice",
    # Caramel is included so that "truffle salt caramel" and similar compound
    # confection names are not incorrectly suppressed by the culinary-truffle
    # guard (which strips truffle words then re-checks for confectionery keywords;
    # without "caramel" the guard would fire and suppress is_confectionery for
    # "truffle salt caramel", routing it to UNCLASSIFIED or wrong 0% VAT).
    "caramel", "caramels",
)
_FASHION_RE = _make_word_re(
    "belt", "belts", "glove", "gloves",
    "hat", "hats", "brooch", "brooches", "headband", "headbands",
)
_BAG_RE = _make_word_re(
    "bag", "bags", "handbag", "handbags", "purse", "purses",
    "tote", "totes", "clutch", "clutches", "satchel", "satchels",
    "backpack", "backpacks", "rucksack", "rucksacks",
    "briefcase", "briefcases",
    # Wallets and coin purses are personal goods containers (HS 4202),
    # not clothing accessories (HS 6217).  They are intentionally excluded
    # from _FASHION_RE so leather wallets route to the 4202 branch.
    "wallet", "wallets",
    "pouch", "pouches",
    "crossbody", "crossbodies",
)
_FREE_MARKER_RE = re.compile(
    r'\b(?:fragrance|perfume)[-–— ]free\b'                          # fragrance-free, perfume free, etc.
    r'|\b(?:no|without)\s+(?:added\s+)?(?:fragrances?|perfumes?)\b' # no fragrance/fragrances, no added fragrance
    r'|\bunscented\b'                                                # unscented
)
# Restricted pattern for material-field perfume detection.  Deliberately excludes
# bare "fragrance" / "fragrances" because those are the standard INCI ingredient
# names that appear in virtually every fragranced cosmetic material list
# ("aqua, glycerin, fragrance") and would otherwise cause face creams, body
# lotions, and similar beauty products to be misclassified as perfumes.  Only
# compound forms ("fragrance compounds", "fragrance oil") and unambiguous product-
# type terms (perfume, cologne, eau de …) are matched in material fields.
_PERFUME_MATERIAL_RE = re.compile(
    r'\b(?:perfumes?|colognes?|aftershaves?'
    r'|eau[ -]de[ -](?:parfum|toilette|cologne)'
    r'|fragrance\s+(?:compounds?|oils?|bases?|concentrates?)'
    r')\b'
)
# Preceding qualifiers that indicate a synthetic imitation rather than the genuine
# material.  These complement the negative-lookahead approach used in _SILK_RE and
# _LEATHER_RE (which only catch following modifiers like "silk-effect", "leather-look")
# by also catching constructions like "faux leather", "vegan leather", "PU leather",
# "synthetic silk" — all of which would otherwise pass through the duty-code upgrade
# branches and attract the wrong (higher) duty rates.
_FAUX_SILK_RE = re.compile(
    r'\b(?:faux|vegan|synthetic|artificial|imitation|fake)[-\s]+silks?\b'
    r'|\bman[-\s]?made[-\s]+silks?\b'
)
_FAUX_LEATHER_RE = re.compile(
    r'\b(?:faux|vegan|synthetic|artificial|imitation|fake|pu|polyurethane|eco|bonded)[-\s]+leathers?\b'
)
# Explicit "genuine / real / authentic" qualifiers in the same material segment
# override a co-present faux marker.  This handles supplier material strings that
# list both materials without a separator, e.g. "genuine leather and faux leather
# trim" — the genuine qualifier must win so the item is not misclassified as
# non-leather (and thus under-declared at 3.7% duty instead of 16%).
_GENUINE_LEATHER_RE = re.compile(r'\b(?:genuine|real|authentic)[-\s]+leathers?\b')
_GENUINE_SILK_RE = re.compile(r'\b(?:genuine|real|authentic)[-\s]+silks?\b')
# Matches wallet / coin-purse product types.  Within the leather-bag branch these
# route to HS 4202.31 (leather outer surface, small articles such as wallets) rather
# than 4202.21 (handbags), correcting a ~12 pp duty-rate error.
_WALLET_RE = re.compile(r'\bwallets?\b')
_EURO_DECIMAL_RE = re.compile(r',\d{1,2}$')
_PERFUME_RE = re.compile(
    r'\b(?:perfumes?|fragrances?|colognes?|aftershaves?'
    r'|eau[ -]de[ -](?:parfum|toilette|cologne))\b'
)
# Non-perfume products whose description may contain "fragrance" as a modifier
# or ingredient name rather than as a product-type term.  Matched against desc
# to suppress false is_perfume signals for candles, diffusers, and fragrance oils,
# which are HS 3406/3307/3302 respectively, not HS 3303 toilet waters.
# The ingredient-context alternatives ("with fragrances", "added fragrances", etc.)
# prevent cosmetic descriptions such as "moisturizing lotion with fragrances" from
# being misclassified as perfume.  The bare "fragrances" term in _PERFUME_RE is
# intentionally broad (a product titled "Fragrances Gift Set" IS a perfume product),
# so suppression is applied here via context rather than by excluding the term from
# _PERFUME_RE itself.  The "with/added/enriched with/infused with" prefixes are
# unambiguous ingredient-list markers that never introduce perfume product names.
_FRAGRANCE_NON_PERFUME_RE = re.compile(
    r'\bfragrances?\s+(?:candle|candles|diffuser|diffusers|oil|oils|lamp|lamps|wax|warmer|warmers)\b'
    r'|\b(?:scented\s+candle|scented\s+candles|reed\s+diffuser|reed\s+diffusers'
    r'|wax\s+melt|wax\s+melts|oil\s+burner|oil\s+burners|aromatherapy\s+diffuser)\b'
    r'|\b(?:with|added|enriched\s+with|infused\s+with)\s+'
    r'(?:(?:natural|synthetic|artificial|floral|fruity|citrus|botanical|herbal)\s+)?fragrances?\b'
)
# Pre-compiled patterns to strip polysemous words for culinary vs confection disambiguation.
_TRUFFLE_WORD_RE = re.compile(r'\btruffles?\b')
# "caramel" / "caramels" appear both as chocolate confections and as culinary flavour
# descriptors (e.g. "caramel truffle oil", "caramel sauce").  Used alongside
# _TRUFFLE_WORD_RE in the culinary-truffle guard to suppress is_confectionery when
# the only confectionery signals in a description are "caramel" + "truffle" words and
# a culinary-context term is also present.
_CARAMEL_WORD_RE = re.compile(r'\bcaramels?\b')
# Culinary truffle (Tuber genus fungi) context: species qualifiers ("black truffle",
# "white truffle") or preparation/ingredient terms ("truffle oil", "truffle pasta").
# Used to suppress is_confectionery when "truffle"/"truffles" is the sole confectionery
# signal but the product is clearly a food ingredient, not a chocolate confection.
_TRUFFLE_CULINARY_RE = re.compile(
    r'\b(?:black|white|summer|winter|perigord|burgundy|fresh|dried)\s+truffles?\b'
    r'|\btruffles?\s+(?:oil|oils|sauce|sauces|paste|pastes|salt|shavings?|carpaccio|vinaigrette)\b'
    r'|\btruffles?\b.*\b(?:oil|sauce|paste|risotto|pasta|mushroom|mushrooms)\b'
    r'|\b(?:oil|sauce|paste|risotto|pasta|mushroom|mushrooms)\b.*\btruffles?\b'
)
_SCARF_RE = re.compile(r'\b(?:scarf|scarfs|scarves|shawl|shawls)\b')
# Engineering/woodworking "scarf": a scarf joint / scarf weld / scarf plane is a
# structural splice, not a textile.  These would otherwise route "scarf joint
# cutters" to HS 621490 (textile scarves, 12% duty) with 0.72 confidence.
# Garment-construction "shawl": shawl collar / shawl lapel / shawl neckline are
# structural garment design elements (a folded lapel style), not textile shawls.
# Without this guard a "silk shawl collar blazer" would be misclassified as a
# silk scarf (HS 621410, 8% duty) instead of remaining UNCLASSIFIED for review.
_SCARF_TECHNICAL_RE = re.compile(
    r'\bscarfs?\s+(?:joint|joints|weld|welds|cut|cuts|plane|planes|ring|rings)\b'
    r'|\b(?:joint|weld|cut|plane)\s+scarfs?\b'
    r'|\bshawl[-\s]+(?:collar|lapel|neckline|neck)\b'
)
# Negative-lookahead excludes compound modifiers such as "silk-effect", "silk-like",
# "leather-look", "leather-feel", etc. which describe synthetic imitations rather
# than the genuine material, preventing false duty-code upgrades for polyester/PU goods.
# s? covers the plural ("silks", "leathers") which appears in supplier-facing material
# fields (e.g. "woven silks", "fine leathers") and bulk CSV exports.
_SILK_RE = re.compile(r'\bsilks?\b(?![-\s]+(?:effect|like|look|feel|finish|touch|screen|road)\b)')
_LEATHER_RE = re.compile(r'\bleathers?\b(?![-\s]+(?:look|like|effect|feel|finish|touch)\b)')
# Compiled separator for splitting material fields on commas or semicolons.
_MAT_SEP_RE = re.compile(r'[,;]')

# Threshold at or above which items attract additional customs scrutiny
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

# Columns pulled from a classified result_df when populating the review queue
# during bulk upload.  Defined at module level so the list is created once.
_BULK_QUEUE_COLS = ["description", "value", "uk_code", "confidence", "explanation", "risk"]

# Maximum number of distinct (desc, material, category, high_value) tuples held in
# the classification cache.  Origin is excluded from the key because classification
# logic is identical regardless of origin — only the explanation note differs, and
# that is appended in the classify_product wrapper outside the cache.  Covers the
# vast majority of real-world SKU catalogues while keeping memory bounded to roughly
# 4 MB worst-case.
_CACHE_MAX_SIZE = 4096

# Hard ceiling on the number of data rows accepted in a single bulk CSV upload.
# Raising this also requires adjusting the nrows sentinel in _process_bulk_upload.
_MAX_BULK_ROWS = 5000


def _parse_value(raw) -> tuple[float, str]:
    """Convert raw value to (normalised_float, warning_message).

    The warning is non-empty only when the raw input was absent or invalid
    and has been defaulted to 0.0.
    Handles common CSV formats: "£1,250.00", "$500", "1,000.50",
    "GBP 250", "250 USD", "EUR1250,00", "1,250,000", "£12,345,678.90".
    """
    # bool / np.bool_ subclass int, so float(True)==1.0 would silently produce a
    # misleading £1 value.  Catch both before the numeric path so callers get a
    # warning instead of a wrong but plausible-looking result.  np.bool_ is NOT a
    # subclass of Python bool (isinstance(np.bool_(True), bool) is False), so it
    # must be checked explicitly.
    if isinstance(raw, (bool, np.bool_)):
        return 0.0, " Warning: declared value was not a number; defaulted to £0 for risk assessment."
    if isinstance(raw, str):
        # Strip currency symbols and ISO 4217 text codes in one pass; strip()
        # afterward removes any whitespace left between the code and the number
        # (e.g. "GBP 250" → "GBP 250" → sub → " 250" → strip → "250").
        s = _VALUE_STRIP_RE.sub('', raw.strip()).strip()
        if not s:
            return 0.0, " Warning: declared value was missing; defaulted to £0 for risk assessment."
        if s.startswith('-'):
            return 0.0, " Warning: declared value was negative; defaulted to £0 for risk assessment."
        # Detect European decimal format: comma followed by 1–2 digits at end,
        # with exactly one comma (e.g. "1.250,00" → "1250.00"). The single-comma
        # guard prevents "1,250,00" (two commas, a common typo) from matching the
        # Euro branch and producing the unparseable "1.250.00". Otherwise treat
        # commas as UK/US thousands separators (e.g. "1,250.00" → "1250.00").
        comma_count = s.count(',')
        dot_count = s.count('.')
        euro_tail = _EURO_DECIMAL_RE.search(s)
        if comma_count > 1:
            # Multiple commas are valid as UK/US thousands separators only when
            # every inter-comma group is exactly 3 digits and the leading group
            # is 1–3 digits (e.g. "1,250,000" → 1250000; "1,250,000.50" → 1250000.5).
            # Anything else (e.g. "1,2,345", "1,250,50") is genuinely ambiguous.
            # Without this check, large declared values like "£1,250,000" were
            # silently defaulted to £0, causing HIGH_VALUE_THRESHOLD to be missed
            # and risk ratings to be under-reported.
            _mparts = s.split(',')
            _last_dot = _mparts[-1].find('.')
            _last_base = _mparts[-1][:_last_dot] if _last_dot != -1 else _mparts[-1]
            _last_dec = _mparts[-1][_last_dot + 1:] if _last_dot != -1 else ''
            # Strip a leading '+' for the digit/length checks: some ERP systems
            # export positive values with an explicit '+' sign ("+1,250,000").
            # float() natively accepts a leading '+', so stripping it here only
            # affects the isdigit() and len() guards, not the final float parse.
            _mparts0 = _mparts[0].lstrip('+')
            if (
                _mparts0.isdigit()
                and 1 <= len(_mparts0) <= 3
                and int(_mparts0) != 0   # "0,000,000" is not a valid UK/US large-number format
                and all(len(p) == 3 and p.isdigit() for p in _mparts[1:-1])
                and len(_last_base) == 3
                and _last_base.isdigit()
                and (_last_dec == '' or _last_dec.isdigit())
            ):
                # Strip commas; the elif/else chain below is skipped because
                # no elif condition can match when comma_count > 1.
                s = s.replace(',', '')
            else:
                return 0.0, " Warning: declared value format is ambiguous (multiple commas); defaulted to £0 for risk assessment."
        elif dot_count >= 2 and comma_count == 0:
            # European notation: multiple periods as thousands separators with no
            # decimal part (e.g. "1.250.000" → 1250000). A single period is still
            # treated as a decimal point by the UK/US path below.
            # Only strip when every segment is all-digits and exactly 3 chars
            # (except the leading group which may be 1–3 digits); a trailing
            # 2-digit group (e.g. "1.250.00") or any non-digit segment is
            # ambiguous — warn rather than produce a wrong result.
            parts = s.split('.')
            if (
                parts[0].isdigit()
                and len(parts[0]) <= 3
                and int(parts[0]) != 0   # "0.000.000" is not a valid European thousands format
                and all(len(p) == 3 and p.isdigit() for p in parts[1:])
            ):
                s = s.replace('.', '')
            else:
                return 0.0, " Warning: declared value format is ambiguous (mixed dot groups); defaulted to £0 for risk assessment."
        elif euro_tail and comma_count == 1:
            # Genuine European decimal format: integer part may contain dots only
            # as thousands separators, where each dot-separated group is exactly
            # 3 digits (e.g. "1.250,00" → 1250.00, "1.250.000,99" → 1250000.99).
            # Reject patterns like "1.2,34" where a dot-before-comma integer part
            # has non-3-digit groups — these are ambiguous and bypass the dot-
            # before-comma guard in the else branch.
            ci = s.index(',')
            integer_part = s[:ci]
            if '.' in integer_part:
                int_segs = integer_part.split('.')
                if not (
                    int_segs[0].isdigit()
                    and 1 <= len(int_segs[0]) <= 3
                    and all(len(p) == 3 and p.isdigit() for p in int_segs[1:])
                ):
                    return 0.0, " Warning: declared value format is ambiguous (non-standard European notation); defaulted to £0 for risk assessment."
            s = s.replace('.', '').replace(',', '.')
        else:
            # UK/US path: commas are thousands separators.  Warn on two non-standard
            # patterns: (a) dot-before-comma without a recognised Euro decimal tail
            # (e.g. "1.250,000" — euro_tail only catches 1-2 digit tails, so a
            # 3-digit tail like ",000" falls through here); (b) comma-before-dot with
            # a non-3-digit inter-separator group (e.g. "1,50.00").
            if comma_count == 1 and dot_count == 1:
                ci = s.index(',')
                di = s.index('.')
                if di < ci:
                    # Dot precedes comma without a matching euro_tail — ambiguous.
                    return 0.0, " Warning: declared value format is ambiguous (dot before comma without standard decimal suffix); defaulted to £0 for risk assessment."
                if len(s[ci + 1:di]) != 3:
                    return 0.0, " Warning: declared value format is ambiguous (non-standard digit grouping); defaulted to £0 for risk assessment."
            elif comma_count == 0 and dot_count == 1:
                # Single dot with exactly 3 decimal digits is ambiguous ONLY when the
                # integer part is non-zero AND has at most 3 digits: "1.250" could be
                # £1.25 (decimal) or £1,250 (European thousands), differing by a factor
                # of 1,000 and potentially flipping a £1,250 item below
                # HIGH_VALUE_THRESHOLD.  When the integer part is "0" ("0.250", "0.999")
                # the European interpretation would require a leading-zero thousands group
                # ("0250"), which no standard ERP uses; treat it unambiguously as a plain
                # decimal (e.g. "0.250" = £0.25).
                # When the integer part has 4+ digits (e.g. "1250.000") the European
                # thousands-separator interpretation is impossible — a single separator
                # can only precede a 3-digit group with a 1-3 digit leading segment
                # (e.g. "1.250", "12.250", "123.250").  Parse these unambiguously.
                # Mirrors the guard in the multi-comma branch (int(_mparts[0]) != 0)
                # and the multi-dot branch (int(parts[0]) != 0).
                di = s.index('.')
                pre_dot = s[:di]
                post_dot = s[di + 1:]
                if (
                    pre_dot.isdigit()
                    and 1 <= len(pre_dot) <= 3
                    and len(post_dot) == 3
                    and post_dot.isdigit()
                    and int(pre_dot) != 0
                ):
                    return 0.0, (
                        " Warning: declared value format is ambiguous"
                        " (a single dot followed by exactly 3 digits could be a European"
                        " thousands separator, e.g. '1.250' = £1,250, or a decimal point,"
                        " e.g. '1.250' = £1.25); defaulted to £0 for risk assessment."
                    )
            s = s.replace(',', '')
        cleaned = s
    else:
        cleaned = raw
    try:
        v = float(cleaned)
    except (TypeError, ValueError):
        try:
            # bool() raises ValueError for array-like results (e.g. raw=[250, 300]);
            # that is caught here so the function never propagates an exception.
            is_missing = bool(pd.isna(raw))
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


def _is_normalised_float(value) -> bool:
    """Return True when value is already a finite, non-negative number.

    Used as a fast-path guard to skip a redundant _parse_value round-trip when
    the caller (e.g. classify_row) has already parsed the value via _parse_value.
    Accepts Python float, int, and numpy numeric types (e.g. np.float64) that are
    coercible to float — but not bool/np.bool_ (which subclass int) or str.
    """
    if isinstance(value, (bool, np.bool_, str)):
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v >= 0


def _normalise_value(value) -> float:
    """Convert value to a finite, non-negative float rounded to pence."""
    if _is_normalised_float(value):
        return round(float(value), 2)
    v, _ = _parse_value(value)
    return v


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


def classify_product(description, material, origin, category, value) -> dict:
    """Normalise inputs then delegate to the cached implementation."""
    v = _normalise_value(value)
    # _safe_str handles None, np.nan, pd.NA, and all other non-string types that
    # (x or "") does not: np.nan is truthy, so (np.nan or "") returns np.nan and
    # np.nan.strip() raises AttributeError; pd.NA raises TypeError on boolean eval.
    origin_upper = _safe_str(origin).strip().upper()
    origin_note = (
        f" Country of origin: {origin_upper}."
        if origin_upper
        else " Warning: country of origin not declared — required for customs clearance."
    )
    # Return a shallow copy so callers cannot mutate the lru_cache entry.
    # Origin is handled here (outside the cache) so products from different countries
    # with identical descriptions/materials/categories share the same cache entry.
    result = dict(_classify_product_cached(
        _safe_str(description).strip().lower(),
        _safe_str(material).strip().lower(),
        _safe_str(category).strip().lower(),
        v >= HIGH_VALUE_THRESHOLD,
    ))
    result["explanation"] = result["explanation"] + origin_note
    return result


@functools.lru_cache(maxsize=_CACHE_MAX_SIZE)
def _classify_product_cached(desc, material_lower, category_lower, high_value) -> dict:
    # high_value is a bool; using it instead of the raw value means products that
    # share the same description/material/category and the same high-value status
    # hit the same cache entry regardless of exact declared price.  Origin is NOT
    # part of the key — classification logic is identical across origins; only the
    # explanation note differs and that is appended by classify_product after the
    # cache lookup.
    hv_note = " High declared value flagged for additional customs scrutiny." if high_value else ""

    # Pre-compute all keyword flags once to avoid redundant regex evaluation.
    # _SCARF_TECHNICAL_RE excludes engineering/woodworking uses of "scarf" (e.g.
    # "scarf joint cutter") and garment-construction uses of "shawl" (e.g.
    # "shawl collar blazer", "shawl lapel jacket") — neither are textile articles
    # and both must not route to HS 6214 (scarves, 12% duty).
    is_scarf = bool(_SCARF_RE.search(desc)) and not bool(_SCARF_TECHNICAL_RE.search(desc))
    # Material is the authoritative source for composition.  Only fall back to
    # description when the material field was not supplied, so that terms like
    # "silk-effect polyester" or "leather-look PU" in a description do not
    # trigger the silk/leather duty codes when the actual material differs.
    # _FAUX_SILK_RE / _FAUX_LEATHER_RE guard against preceding qualifiers
    # ("faux leather", "vegan leather", "synthetic silk", "PU leather", etc.)
    # which would otherwise pass through _SILK_RE / _LEATHER_RE unchanged and
    # attract the wrong (higher) duty-code branches.
    # Material fields from ERP/supplier systems often list components separated by
    # commas (e.g. "faux silk lining, genuine silk outer shell").  Each segment is
    # checked independently so a faux qualifier in one segment does not suppress a
    # genuine-material signal in another.  The desc fallback keeps whole-string
    # matching since descriptions are free-text, not structured component lists.
    # _GENUINE_LEATHER_RE / _GENUINE_SILK_RE override the faux suppression within a
    # single unseparated segment that mentions both: "genuine leather and faux leather
    # trim" must still be flagged as genuine leather.
    _mat_segs = list(filter(None, (seg.strip() for seg in _MAT_SEP_RE.split(material_lower)))) if material_lower else []
    if _mat_segs:
        is_silk = False
        is_leather = False
        for _seg in _mat_segs:
            if not is_silk and _SILK_RE.search(_seg):
                if not _FAUX_SILK_RE.search(_seg) or _GENUINE_SILK_RE.search(_seg):
                    is_silk = True
            if not is_leather and _LEATHER_RE.search(_seg):
                if not _FAUX_LEATHER_RE.search(_seg) or _GENUINE_LEATHER_RE.search(_seg):
                    is_leather = True
            if is_silk and is_leather:
                break
    else:
        is_silk = bool(_SILK_RE.search(desc) and (
            not _FAUX_SILK_RE.search(desc) or _GENUINE_SILK_RE.search(desc)
        ))
        is_leather = bool(_LEATHER_RE.search(desc) and (
            not _FAUX_LEATHER_RE.search(desc) or _GENUINE_LEATHER_RE.search(desc)
        ))
    # Either "fragrance-free" or "perfume-free" in description or material negates
    # the product being a fragrance/perfume; both flags suppress ALL perfume signals
    # (including cologne, aftershave, eau-de) not just the keyword they name.
    # Description uses the full _PERFUME_RE (including bare "fragrance"/"fragrances"
    # as product-type words).  Material uses the narrower _PERFUME_MATERIAL_RE, which
    # excludes bare "fragrance"/"fragrances" because those are standard INCI ingredient
    # names in cosmetics — matching them would misclassify face creams and body lotions
    # as perfumes.  Compound forms ("fragrance compounds", "fragrance oil") and
    # product-type terms ("eau de parfum", "cologne") are still matched in material.
    # category_lower == "beauty" is intentionally NOT included: it is too broad and
    # would misclassify all cosmetics (face creams, lipstick, etc.) as perfumes.
    _free_marker = bool(
        _FREE_MARKER_RE.search(desc) or _FREE_MARKER_RE.search(material_lower)
    )
    # _FRAGRANCE_NON_PERFUME_RE guards "fragrance candle", "fragrance diffuser",
    # "fragrance oil" etc. from matching the bare "fragrance" alternative in
    # _PERFUME_RE.  These products are HS 3406/3307/3302, not HS 3303 toilet waters.
    # The suppressor is checked against both desc AND material_lower: if
    # _PERFUME_MATERIAL_RE fires solely on the material (e.g. material="fragrance oil
    # concentrate") without a perfume-type word in desc, omitting the material check
    # would leave the suppressor silent and misclassify the item as HS 3303.
    is_perfume = not _free_marker and bool(
        _PERFUME_RE.search(desc) or _PERFUME_MATERIAL_RE.search(material_lower)
    ) and not bool(
        _FRAGRANCE_NON_PERFUME_RE.search(desc) or _FRAGRANCE_NON_PERFUME_RE.search(material_lower)
    )
    # Non-fragrance beauty products (skincare, make-up, etc.) fall here.
    is_cosmetics = category_lower == "beauty" and not is_perfume
    # Only search desc, not material_lower: confectionery keywords such as "caramel",
    # "truffle", and "nougat" appear routinely as colour, texture, or ingredient
    # names in fashion and cosmetics material fields, and checking material_lower
    # would misroute those items to food classification.  desc is the authoritative
    # product-type signal; material_lower records physical composition.
    _conf_match = _CONFECTIONERY_RE.search(desc)
    is_confectionery = bool(_conf_match)
    # Culinary truffle guard: "truffle"/"truffles" is polysemous — it denotes both
    # a chocolate confection (standard-rated at 20% VAT) and Tuber genus fungi
    # (zero-rated food ingredient).  If "truffle"/"truffles" is the FIRST (and
    # potentially only) confectionery keyword and a culinary-context term is also
    # present, re-check whether any other confectionery keyword remains after
    # stripping the truffle words.  If not, suppress is_confectionery to prevent
    # "truffle oil" or "black truffle pasta" from attracting 20% VAT.
    # Secondary branch: "caramel" is also polysemous (confection AND culinary flavour
    # descriptor).  When "caramel" is the leftmost confectionery match (i.e. it appears
    # before "truffle" in the string) the original truffle check is skipped.  The
    # secondary branch handles this case: if the only confectionery signals are
    # "caramel" and "truffle" words, and a culinary-context term is present, suppress
    # is_confectionery.  Example: "caramel truffle oil" → culinary; contrast with
    # "truffle salt caramel" where "truffle" is leftmost and "caramel" survives
    # truffle-stripping, correctly preserving the confection classification.
    if is_confectionery and _TRUFFLE_CULINARY_RE.search(desc):
        _first_conf = _conf_match.group()
        if _first_conf in ('truffle', 'truffles') and not _CONFECTIONERY_RE.search(
            _TRUFFLE_WORD_RE.sub('', desc)
        ):
            is_confectionery = False
        elif _first_conf in ('caramel', 'caramels') and not _CONFECTIONERY_RE.search(
            _TRUFFLE_WORD_RE.sub('', _CARAMEL_WORD_RE.sub('', desc))
        ):
            is_confectionery = False
    is_fashion = category_lower == "fashion_accessories" or bool(_FASHION_RE.search(desc))
    # Confectionery keywords drive food classification only when the category is
    # blank (no signal) or explicitly "food".  Any non-empty category — whether a
    # known type like "bags"/"beauty" or an unknown bulk-CSV value like "electronics"
    # — is treated as a contradicting signal and suppresses the keyword override.
    # This prevents "chocolate-coloured sofa" (category: furniture) and
    # "chocolate gift bag" (category: bags) from being misclassified as food.
    # When category is blank, genuine-material signals (is_leather, is_silk) and
    # fashion-item keywords (is_fashion) also suppress confectionery food
    # classification: colour/texture names like "fudge brown", "toffee", and
    # "chocolate" appear routinely in fashion product descriptions ("chocolate
    # wallet", "toffee belt") and should not override a clear product-type signal.
    # category="food" always wins regardless of other flags.
    is_food = category_lower == "food" or (
        is_confectionery and not category_lower and not is_leather and not is_silk and not is_fashion
    )
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
    _bag_by_keyword = _bag_keyword and category_lower != "fashion_accessories" and not is_food
    _bag_by_category = category_lower == "bags" and not is_fashion and not is_scarf and not is_food
    is_bag = _bag_by_keyword or _bag_by_category

    # Scarf detection: food category overrides scarf keywords for consistency with the
    # is_bag guard above — category="food" is treated as an authoritative product-type
    # signal that suppresses textile classifications, preventing a data-entry error
    # (e.g. category="food" on a "silk scarf" description) from producing HS 6214.
    # is_bag overrides is_scarf: when description keywords name both a bag and a scarf
    # (e.g. "silk scarf print shopper bag", "silk shawl tote") the product is a bag
    # and the scarf word is a modifier.  Without this guard the scarf branch fires
    # first (it precedes is_bag in the elif chain) and misclassifies the item as
    # HS 621410 (silk scarf, 8% duty) instead of HS 4202 (travel goods/bags).
    if is_scarf and is_silk and not is_bag and not is_food:
        return {
            "hs6": "621410",
            "uk_code": "6214100090",
            "confidence": 0.94,
            "risk": RISK_RED if high_value else RISK_GREEN,
            "duty": "8%",
            "vat": "20%",
            "explanation": "Classified under silk scarves, shawls and similar articles based on material composition and accessory type." + hv_note,
        }
    elif is_bag and is_leather:
        # Wallets and similar small leather articles (HS 4202.31) attract the same
        # 16% duty as handbags (HS 4202.21) but have a distinct commodity code;
        # exporting the handbag code for a wallet is a declaration error.
        _is_wallet = bool(_WALLET_RE.search(desc))
        if _is_wallet:
            return {
                "hs6": "420231",
                "uk_code": "4202310000",
                "confidence": 0.82,
                "risk": RISK_RED if high_value else RISK_AMBER,
                "duty": "16%",
                "vat": "20%",
                "explanation": "Classified under leather wallets and similar small articles (HS 4202.31); verify precise subheading — coin purses: 4202.32." + hv_note,
            }
        return {
            "hs6": "420221",
            "uk_code": "4202210000",
            "confidence": 0.82,
            "risk": RISK_RED if high_value else RISK_AMBER,
            "duty": "16%",
            "vat": "20%",
            "explanation": "Classified under leather travel goods and handbags (HS 4202.21); verify specific subheading — wallets and small articles: 4202.31/4202.32." + hv_note,
        }
    elif is_bag:
        return {
            "hs6": "420229",
            "uk_code": "4202290000",
            "confidence": 0.65,
            "risk": RISK_RED if high_value else RISK_AMBER,
            "duty": "3.7%",
            "vat": "20%",
            "explanation": "Classified under travel goods, handbags and similar containers (HS 4202); verify material composition for precise subheading — leather surface attracts 4202.21/4202.31 (16% duty)." + hv_note,
        }
    elif is_scarf and not is_food:
        return {
            "hs6": "621490",
            "uk_code": "6214900000",
            "confidence": 0.72,
            "risk": RISK_RED if high_value else RISK_GREEN,
            "duty": "12%",
            "vat": "20%",
            "explanation": "Classified under scarves, shawls and similar articles (non-silk); verify fibre composition for precise subheading (wool: 621420, synthetic fibres: 621430, other fibres: 621490)." + hv_note,
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
        # When category="food" triggers without a confectionery keyword the item may
        # still be standard-rated — alert the analyst rather than asserting zero rate.
        vat_note = (
            " Note: confectionery and snack products (e.g. chocolate, biscuits, candy, sweets, toffee, fudge, caramel, snacks)"
            " are standard-rated at 20% VAT in the UK."
            if is_confectionery
            else " Note: verify VAT rate — most food is zero-rated in the UK, but confectionery"
            " (sweets, chocolates, gummies, marshmallows, etc.) is standard-rated at 20%."
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


def classify_row(row) -> pd.Series:
    """Apply classify_product to a DataFrame row; safe for use with df.apply()."""
    # Parse value before the try/except so val is always defined in the except
    # handler — preserving the correct risk rating even when classify_product
    # raises.  _parse_value is designed never to raise; this is purely defensive.
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
            result["explanation"] += val_warning
        return pd.Series(result)
    except Exception as e:
        row_idx = getattr(row, "name", None)
        # hasattr(__index__) covers Python int and numpy integer scalars.
        # bool is excluded explicitly: bool subclasses int, so True+1=2 and
        # False+1=1 would produce a misleading "Row 2:"/"Row 1:" prefix.
        display_idx = (
            (row_idx + 1)
            if hasattr(row_idx, "__index__") and not isinstance(row_idx, bool)
            else row_idx
        )
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


def _add_to_review_queue(result: dict) -> None:
    """Add a classified item to the review queue if not already present.

    Deduplicates on (description, high_value_flag, uk_code) so that re-clicking
    the button for the same product does not create duplicate queue entries, but
    a genuine reclassification that produces a different code or crosses the
    high-value threshold (and thus a different risk rating) is still added.
    Silently ignores ERROR and UNCLASSIFIED items — callers filter these, but
    this guard prevents accidental queue corruption if called directly.
    """
    if result.get("uk_code", UNCLASSIFIED_CODE) in {ERROR_CODE, UNCLASSIFIED_CODE}:
        return
    raw_val = result.get("value", 0.0)
    safe_val = _normalise_value(raw_val)
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
            "Product": _safe_str(result.get("description", "")),
            "Suggested Code": result.get("uk_code", UNCLASSIFIED_CODE),
            "Confidence": _format_confidence(result.get("confidence", 0.0)),
            "Explanation": _safe_str(result.get("explanation", "")),
            "Risk": result.get("risk", RISK_AMBER),
            "Status": STATUS_PENDING,
        })


def _apply_bulk_review(new_status: str, audit_event: str, toast_msg: str, toast_icon: str) -> None:
    """Set all pending review-queue items to new_status and log the action."""
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
        ts = datetime.now().isoformat(timespec="microseconds")
        skipped_note = (
            f"; {skipped_unclassified} item(s) skipped (unclassified or errored — require manual code assignment)"
            if skipped_unclassified
            else ""
        )
        st.session_state["audit_log"].append({"Timestamp": ts, "Event": audit_event.format(count=changed) + skipped_note})
        st.toast(toast_msg.format(count=changed), icon=toast_icon)
        st.session_state["_review_edit_version"] += 1
        st.rerun()
    elif skipped_unclassified:
        ts = datetime.now().isoformat(timespec="microseconds")
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
        st.toast("No pending items to action — all items have already been approved or overridden.", icon="ℹ️")


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
    # Mark the file as processed immediately. Validation errors are deterministic for
    # a given file (same content → same error), so leaving _bulk_file_id unset would
    # cause _process_bulk_upload to be re-invoked on every subsequent page rerun while
    # the same bad file remains selected.
    st.session_state["_bulk_file_id"] = file_id
    try:
        # Read one extra row so len(df) > _MAX_BULK_ROWS can detect oversized files.
        # keep_default_na=False prevents pandas from silently converting product
        # descriptions and other text fields that happen to spell "NA", "NULL",
        # "N/A", "NaN", etc. to NaN, which would cause those rows to be
        # misclassified as having empty descriptions.  Genuinely missing cells
        # (empty CSV fields) become "" which _safe_str and _parse_value already
        # handle identically to NaN.
        df = pd.read_csv(
            io.BytesIO(file_bytes),
            nrows=_MAX_BULK_ROWS + 1,
            encoding="utf-8-sig",
            encoding_errors="replace",
            keep_default_na=False,
            low_memory=False,
        )
        df.columns = df.columns.str.strip().str.lower()
        # Warn if any cell contains U+FFFD (the Unicode replacement character),
        # which indicates bytes that could not be decoded from the file's encoding.
        # Generator short-circuits on the first matching column instead of scanning
        # all columns then discarding the intermediate boolean Series.
        str_cols = df.select_dtypes(include=["object", "string"])
        if not str_cols.empty and any(
            col.str.contains("\ufffd", regex=False, na=False).any()
            for _, col in str_cols.items()
        ):
            st.session_state["_bulk_messages"].append(("warning", (
                "Some characters in the CSV could not be decoded and have been "
                "replaced with \ufffd. Re-save the file as UTF-8 to ensure accurate "
                "classification."
            )))
    except pd.errors.EmptyDataError:
        st.session_state["_bulk_messages"].append(("error", "The uploaded file is empty — it contains no columns or data."))
        return
    except pd.errors.ParserError:
        st.session_state["_bulk_messages"].append(("error", "CSV format is invalid — check that columns are comma-separated and the file is UTF-8 encoded."))
        return
    except Exception as e:
        st.session_state["_bulk_messages"].append(("error", f"Failed to read file: {e}"))
        return

    if len(df) > _MAX_BULK_ROWS:
        st.session_state["_bulk_messages"].append(("error", f"CSV exceeds the {_MAX_BULK_ROWS:,}-row limit (more than {_MAX_BULK_ROWS:,} rows detected). Split the file and re-upload."))
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
        st.session_state["_bulk_messages"].append(("warning", f"The following columns from your CSV will be replaced by classification results: {', '.join(overlapping)}"))
    # Drop any pre-existing result columns to avoid duplicate columns after concat.
    input_df = df.drop(columns=overlapping).reset_index(drop=True)
    n = len(input_df)
    _progress = st.progress(0.0, text=f"Classifying 0 of {n} rows…")
    # Chunk size chosen so progress updates occur roughly every 50 rows
    # regardless of file size — responsive for small files, not overwhelming
    # for large ones.  df.apply(classify_row, axis=1) is faster than an
    # explicit iterrows() loop because pandas manages the row-level iteration
    # internally, avoiding the overhead of constructing a fresh Python Series
    # object for every single row.
    _CLASSIFY_CHUNK = 50
    _chunk_results: list[pd.DataFrame] = []
    try:
        for _start in range(0, n, _CLASSIFY_CHUNK):
            _end = min(_start + _CLASSIFY_CHUNK, n)
            _chunk_results.append(
                input_df.iloc[_start:_end].apply(classify_row, axis=1)
            )
            _progress.progress(_end / n, text=f"Classifying row {_end} of {n}…")
    except Exception as e:
        st.session_state["_bulk_messages"].append(("error", f"Classification failed: {e}"))
        return
    finally:
        # Always dismiss the progress bar — runs even when except returns.
        _progress.empty()
    try:
        classified = pd.concat(_chunk_results, ignore_index=True)
        result_df = pd.concat([input_df, classified], axis=1)
    except Exception as e:
        st.session_state["_bulk_messages"].append(("error", f"Classification failed: {e}"))
        return

    # Compute error/unclassified masks once; reused for the summary, the queue
    # filter, and the Bulk Upload page display to avoid redundant column scans.
    _hs6 = result_df["hs6"]
    _is_error = _hs6 == ERROR_CODE
    _is_unclassified = _hs6 == UNCLASSIFIED_CODE

    try:
        error_count = int(_is_error.sum())
        unclassified_count = int(_is_unclassified.sum())
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
            "error_count": error_count,
            "unclassified_count": unclassified_count,
        }
    except Exception as e:
        st.session_state["_bulk_messages"].append(("error", f"Failed to summarise classification results: {e}"))
        return

    queueable_df = result_df[~(_is_error | _is_unclassified)]
    _queue_changed = False
    try:
        queue_before = len(st.session_state["review_items"])
        for row in queueable_df[_BULK_QUEUE_COLS].to_dict("records"):
            _add_to_review_queue({
                "description": _safe_str(row.get("description", "")),
                "value": row.get("value", 0.0),
                "uk_code": _safe_str(row.get("uk_code", "")),
                "confidence": row.get("confidence", 0.0),
                "explanation": _safe_str(row.get("explanation", "")),
                "risk": _safe_str(row.get("risk")) or RISK_AMBER,
            })
        _queue_changed = len(st.session_state["review_items"]) > queue_before
    except Exception as e:
        st.session_state["_bulk_messages"].append(("warning", f"Review queue could not be fully populated: {e}"))
    # Invalidate the Review Queue data_editor only when new items were actually
    # added — an unchanged queue does not need the widget reset, which would
    # discard in-progress edits without cause.
    if _queue_changed:
        st.session_state["_review_edit_version"] += 1
    # _bulk_file_id was already set at the top of this function.


# Initialise session state keys once so all pages can rely on them existing.
st.session_state.setdefault("review_items", [])
st.session_state.setdefault("review_keys", set())
st.session_state.setdefault("audit_log", [])
st.session_state.setdefault("bulk_result", None)
st.session_state.setdefault("_bulk_file_id", None)
st.session_state.setdefault("_bulk_messages", [])
st.session_state.setdefault("last_result", None)
# Version counter for the Review Queue data_editor key.  Incrementing it forces
# Streamlit to discard the widget's stored edit delta, preventing a stale delta
# from replaying against freshly-updated item statuses after a bulk action rerun.
st.session_state.setdefault("_review_edit_version", 0)
if "seed_logs" not in st.session_state:
    _seed_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
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
    # Single pass over session_items to build both status and risk counters.
    status_counts: Counter = Counter()
    risk_counts: Counter = Counter()
    for _item in session_items:
        status_counts[_item["Status"]] += 1
        risk_counts[_item["Risk"]] += 1
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
        risk_df = pd.DataFrame(
            {"Risk": [RISK_GREEN, RISK_AMBER, RISK_RED],
             "Count": [risk_counts[RISK_GREEN], risk_counts[RISK_AMBER], risk_counts[RISK_RED]]},
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
        category = st.selectbox(
            "Category",
            ["fashion_accessories", "bags", "beauty", "food", "other"],
            help=(
                "Choose the closest category. 'other' relies solely on description keywords and "
                "may return UNCLASSIFIED for food or confectionery items — select 'food' for edible products."
            ),
        )
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
                    st.session_state["last_result"] = None
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
        col_hs6, col_code, col_conf = st.columns(3)
        col_hs6.metric("HS6", r["hs6"])
        col_code.metric("UK Commodity Code", r["uk_code"])
        col_conf.metric("Confidence", _format_confidence(r["confidence"]))

        col_risk, col_duty, col_vat = st.columns(3)
        col_risk.metric("Risk", r["risk"])
        col_duty.metric("Duty", r["duty"])
        col_vat.metric("VAT", r["vat"])

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
        # usedforsecurity=False is required on FIPS-enabled Python 3.9+ systems.
        # Python 3.8 doesn't accept that kwarg (TypeError); on FIPS Python 3.8 the
        # plain md5() fallback also fails with ValueError, so fall back to SHA-256
        # (always available, including in FIPS mode) purely for file-identity dedup.
        try:
            _hex = hashlib.md5(raw_bytes, usedforsecurity=False).hexdigest()
        except TypeError:
            try:
                _hex = hashlib.md5(raw_bytes).hexdigest()
            except ValueError:
                _hex = hashlib.sha256(raw_bytes).hexdigest()
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
        problem_rows = bulk["error_count"] + bulk["unclassified_count"]
        if problem_rows == len(result_df):
            st.error(bulk["summary"])
        elif problem_rows > 0:
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
        # num_rows="fixed" prevents row deletion/insertion so the zip-based status-sync
        # loop below always compares items[i] against the correct edited row at index i.
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
            num_rows="fixed",
            hide_index=True,
            use_container_width=True,
            key=f"review_queue_editor_{st.session_state['_review_edit_version']}",
        )

        # Detect per-row status changes: iterate once, track whether anything changed,
        # then rerun only if needed.  A single O(n) pass avoids the previous approach
        # of a separate O(n) list comparison followed by a second O(n) zip iteration.
        # Guard: num_rows="fixed" prevents insertion/deletion so lengths should always
        # match; if they diverge (Streamlit edge case), skip the sync this rerun rather
        # than silently dropping the last rows via zip truncation.
        changed = False
        ts = None
        _status_pairs = (
            zip(review_df["Status"], edited_df["Status"])
            if len(edited_df) == len(review_df)
            else []
        )
        for i, (orig_status, new_status) in enumerate(_status_pairs):
            if orig_status != new_status:
                if ts is None:
                    ts = datetime.now().isoformat(timespec="microseconds")
                items[i]["Status"] = new_status
                st.session_state["audit_log"].append({
                    "Timestamp": ts,
                    "Event": (
                        f"Review Queue: '{items[i]['Product']}' "
                        f"status changed from {orig_status} to {new_status}"
                    ),
                })
                changed = True
        if changed:
            st.session_state["_review_edit_version"] += 1
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
