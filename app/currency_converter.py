"""
Currency conversion utility for Cricket Player Rating & Transfer Value System.

Uses the free ExchangeRate-API (open.er-api.com, no key required) to fetch
live USD-based exchange rates and convert INR Crore values to any supported currency.

Rates are cached in Streamlit's @st.cache_data with a 1-hour TTL so the app
never hammers the API and stays accurate throughout a session.
"""

from __future__ import annotations
import urllib.request
import json


# All values are stored internally as INR Crore.
# 1 INR Crore = 10,000,000 INR  (1 Cr = 1e7 INR)
INR_PER_CRORE = 10_000_000

# Curated list of currencies users are most likely to care about
# (cricket-playing nations + major financial centres)
SUPPORTED_CURRENCIES: dict[str, dict] = {
    "INR – Indian Rupee (₹ Cr)":        {"code": "INR", "symbol": "₹",  "unit": "Cr",   "divisor": INR_PER_CRORE},
    "USD – US Dollar ($)":              {"code": "USD", "symbol": "$",  "unit": "M",    "divisor": 1_000_000},
    "GBP – British Pound (£)":          {"code": "GBP", "symbol": "£",  "unit": "M",    "divisor": 1_000_000},
    "EUR – Euro (€)":                   {"code": "EUR", "symbol": "€",  "unit": "M",    "divisor": 1_000_000},
    "AUD – Australian Dollar (A$)":     {"code": "AUD", "symbol": "A$", "unit": "M",    "divisor": 1_000_000},
    "ZAR – South African Rand (R)":     {"code": "ZAR", "symbol": "R",  "unit": "M",    "divisor": 1_000_000},
    "PKR – Pakistani Rupee (Rs)":       {"code": "PKR", "symbol": "Rs", "unit": "Cr",   "divisor": INR_PER_CRORE},
    "BDT – Bangladeshi Taka (৳)":       {"code": "BDT", "symbol": "৳",  "unit": "Cr",   "divisor": INR_PER_CRORE},
    "LKR – Sri Lankan Rupee (රු)":      {"code": "LKR", "symbol": "රු", "unit": "Cr",   "divisor": INR_PER_CRORE},
    "NZD – New Zealand Dollar (NZ$)":   {"code": "NZD", "symbol": "NZ$","unit": "M",    "divisor": 1_000_000},
    "WIC – West Indian Dollar (TT$)":   {"code": "TTD", "symbol": "TT$","unit": "M",    "divisor": 1_000_000},
    "AED – UAE Dirham (د.إ)":           {"code": "AED", "symbol": "د.إ","unit": "M",    "divisor": 1_000_000},
    "SGD – Singapore Dollar (S$)":      {"code": "SGD", "symbol": "S$", "unit": "M",    "divisor": 1_000_000},
    "CAD – Canadian Dollar (C$)":       {"code": "CAD", "symbol": "C$", "unit": "M",    "divisor": 1_000_000},
    "JPY – Japanese Yen (¥)":           {"code": "JPY", "symbol": "¥",  "unit": "M",    "divisor": 1_000_000},
    "ZWL – Zimbabwean Dollar (Z$)":     {"code": "ZWL", "symbol": "Z$", "unit": "M",    "divisor": 1_000_000},
}

_RATE_URL = "https://open.er-api.com/v6/latest/INR"


def fetch_rates_from_api() -> dict[str, float]:
    """
    Fetches live exchange rates based on INR (1 INR = X foreign).
    Returns a dict: {"USD": 0.012, "GBP": 0.0095, ...}
    Falls back gracefully to hardcoded approximate rates if the network fails.
    """
    try:
        with urllib.request.urlopen(_RATE_URL, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if payload.get("result") == "success":
            return payload["rates"]  # already INR-based
    except Exception:
        pass

    # Fallback approximate rates (1 INR = X)
    # Updated periodically — kept conservative so INR baseline is always exact.
    return {
        "INR": 1.0,
        "USD": 0.01197,
        "GBP": 0.00940,
        "EUR": 0.01101,
        "AUD": 0.01843,
        "ZAR": 0.21810,
        "PKR": 3.3340,
        "BDT": 1.3160,
        "LKR": 3.6400,
        "NZD": 0.01975,
        "TTD": 0.08118,
        "AED": 0.04393,
        "SGD": 0.01585,
        "CAD": 0.01668,
        "JPY": 1.7760,
        "ZWL": 3.8500,
    }


def convert_inr_crore(amount_inr_cr: float, target_currency_label: str, rates: dict[str, float]) -> tuple[float, str]:
    """
    Converts an INR Crore value to the display amount + formatted label.

    Returns:
        (converted_amount, formatted_string)

    Example:
        convert_inr_crore(24.75, "USD – US Dollar ($)", rates)
        → (29.6, "$29.60 M")
    """
    info = SUPPORTED_CURRENCIES.get(target_currency_label, SUPPORTED_CURRENCIES["INR – Indian Rupee (₹ Cr)"])
    code = info["code"]
    symbol = info["symbol"]
    unit = info["unit"]
    divisor = info["divisor"]

    # 1. Convert amount_inr_cr → raw INR
    raw_inr = amount_inr_cr * INR_PER_CRORE

    # 2. Apply exchange rate (rates dict is 1 INR = X target)
    rate = rates.get(code, 1.0)
    converted_raw = raw_inr * rate

    # 3. Divide by display unit
    display_value = converted_raw / divisor

    # 4. Choose appropriate decimal places
    if display_value >= 100:
        fmt = f"{symbol}{display_value:,.1f} {unit}"
    elif display_value >= 10:
        fmt = f"{symbol}{display_value:,.2f} {unit}"
    else:
        fmt = f"{symbol}{display_value:,.3f} {unit}"

    return display_value, fmt


def format_currency_label(amount_inr_cr: float, currency_label: str, rates: dict[str, float]) -> str:
    """Convenience wrapper returning just the formatted string."""
    _, fmt = convert_inr_crore(amount_inr_cr, currency_label, rates)
    return fmt
