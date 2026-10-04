"""Model-facing tools for procurement option selection."""

from copy import deepcopy

from transformers.utils import get_json_schema


def search_suppliers(material_id: str, countries: list[str] | None = None):
    """Search suppliers that advertise a material in optional countries.

    Args:
        material_id: Material requested by the user.
        countries: Optional allowed supplier countries.
    """


def get_supplier_profile(supplier_id: str):
    """Get certifications, reliability, and risk for a known supplier.

    Args:
        supplier_id: Supplier mentioned by the user or returned by search.
    """


def request_quote(
    supplier_id: str,
    material_id: str,
    quantity: float,
    unit: str,
    required_date: str,
):
    """Request price, availability, and readiness from one supplier.

    Args:
        supplier_id: Known supplier to quote.
        material_id: Requested material identifier.
        quantity: Requested quantity.
        unit: Requested unit.
        required_date: Required delivery date in YYYY-MM-DD format.
    """


def get_delivery_options(quote_id: str, destination: str):
    """Get transport options for an observed quote and destination.

    Args:
        quote_id: Quote returned by request_quote.
        destination: User's delivery destination.
    """


def submit_procurement_plan(quote_id: str, delivery_option_id: str):
    """Submit one observed quote and its delivery option as the final plan.

    Args:
        quote_id: Previously observed supplier quote.
        delivery_option_id: Previously observed delivery option for that quote.
    """


def report_no_feasible_option():
    """Finish the task by reporting that no feasible option exists."""


TOOL_FUNCTIONS = [
    search_suppliers,
    get_supplier_profile,
    request_quote,
    get_delivery_options,
    submit_procurement_plan,
    report_no_feasible_option,
]

CHAT_TOOLS = [get_json_schema(function) for function in TOOL_FUNCTIONS]
for tool in CHAT_TOOLS:
    parameters = tool["function"]["parameters"]
    parameters.setdefault("required", [])
    parameters["additionalProperties"] = False


def _openai_parameters(parameters):
    """Convert Transformers schemas to OpenAI strict-function schemas."""
    parameters = deepcopy(parameters)
    for property_schema in parameters.get("properties", {}).values():
        if property_schema.pop("nullable", False):
            property_schema["type"] = [property_schema["type"], "null"]
    parameters["required"] = list(parameters.get("properties", {}))
    return parameters


TOOLS = [
    {
        "type": "function",
        **tool["function"],
        "parameters": _openai_parameters(tool["function"]["parameters"]),
        "strict": True,
    }
    for tool in CHAT_TOOLS
]
