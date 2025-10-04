"""
This is an implementation of an MCP server that provides currency conversion
"""

from mcp.server.fastmcp import FastMCP
from src.util.utils import get_root_dir
import json

# Initialize MCP server
mcp = FastMCP("Currency_Converter")

# Load the official currency rates (from 26 September 2025)
with open(get_root_dir() / 'data' / 'currencies' / 'currency_rates.json', 'r') as f:
    currency_rates = json.load(f)

# Load the currency names
with open(get_root_dir() / 'data' / 'currencies' / 'currency_names.json', 'r') as f:
    currency_names = json.load(f)
currency_names.update({"EUR": "Euro"}) # add Euro manually as it's not in the list

# EUR is the base currency with rate 1.0
currency_rates["EUR"] = 1.0

@mcp.tool()
def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """
    Convert currency from one type to another using exchange rates relative to EUR.
    
    Args:
        amount: The amount to convert
        from_currency: Source currency code (e.g., 'USD', 'EUR', 'GBP')
        to_currency: Target currency code (e.g., 'USD', 'EUR', 'GBP')
    
    Returns:
        Dictionary with conversion result and metadata
    """
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()
    
    if from_currency not in currency_rates:
        return {"error": f"Currency {from_currency} not supported"}
    
    if to_currency not in currency_rates:
        return {"error": f"Currency {to_currency} not supported"}
    
    # Convert to EUR first, then to target currency
    eur_amount = amount / currency_rates[from_currency]
    converted_amount = eur_amount * currency_rates[to_currency]
    
    return {
        "original_amount": amount,
        "original_currency": from_currency,
        "converted_amount": converted_amount,
        "target_currency": to_currency,
        "exchange_rate": currency_rates[to_currency] / currency_rates[from_currency],
        "source": "currency_rates.json"
    }

@mcp.resource("currency://available_currencies")
def get_available_currencies() -> dict:
    """
    Get list of all available currencies with their full names.
    
    Returns:
        Dictionary mapping currency codes to full names
    """
    return {
        "currencies": currency_names,
        "base_currency": "EUR",
        "rates_date": "2025-09-26",
        "total_currencies": len(currency_names)
    }

if __name__ == "__main__":
    mcp.run(transport="stdio")
