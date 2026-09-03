from __future__ import annotations

from datetime import date, datetime
import re
from typing import Any

from .models import ClientProfile


def _get(profile: dict[str, Any], *path: str, default: Any = None) -> Any:
    value: Any = profile
    for key in path:
        if not isinstance(value, dict):
            return default
        value = value.get(key, default)
    return value


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    matches = re.findall(r"[\d,.]+", str(value).replace(",", ""))
    if not matches:
        return None
    try:
        return float(matches[0])
    except ValueError:
        return None


def parse_income_range(value: Any) -> tuple[float | None, str | None]:
    original = str(value).strip() if value not in (None, "") else None
    if not original:
        return None, None
    numbers = [float(item.replace(",", "")) for item in re.findall(r"[\d,.]+", original)]
    if len(numbers) >= 2:
        return (numbers[0] + numbers[1]) / 2, original
    return (numbers[0], original) if numbers else (None, original)


def calculate_age(date_of_birth: str, today: date | None = None) -> int:
    cleaned = re.sub(r"\[cite:\s*\d+\]", "", str(date_of_birth)).strip()
    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", cleaned)
    if year_match and not re.search(r"[-/]\d{1,2}[-/]\d{1,2}", cleaned):
        current = today or date.today()
        return current.year - int(year_match.group(1))
    parsed = datetime.strptime(cleaned, "%Y-%m-%d").date()
    current = today or date.today()
    return current.year - parsed.year - ((current.month, current.day) < (parsed.month, parsed.day))


def count_dependants(profile: dict[str, Any]) -> int:
    relations = _get(profile, "Section_A_Know_Your_Client", "1_Personal_Information",
                     "c_Details_of_Spouse_and_Dependants", default=[])
    if not isinstance(relations, list):
        return 0
    return sum(1 for person in relations if str(person.get("Relation", "")).lower() not in {"spouse", "partner"})


def normalize_profile(profile: dict[str, Any]) -> tuple[ClientProfile, list[str], dict[str, Any]]:
    base = "Section_A_Know_Your_Client"
    personal = _get(profile, base, "1_Personal_Information", default={})
    details = _get(personal, "a_Personal_Details", default={})
    employment = _get(personal, "b_Employment_Details", default={})
    cash_flow = _get(profile, base, "3_Cash_Flow_and_Budget", "a_Cash_Flow", default={})
    budget = _get(profile, base, "3_Cash_Flow_and_Budget", "b_Budget", default={})
    assets = _get(profile, base, "4_Assets_and_Liabilities", "a_Assets", default={})
    priorities = _get(profile, base, "5_Personal_Priorities", "a_Health_Insurance_Concerns", default={})
    portfolio = _get(profile, base, "2_Existing_Insurance_Portfolio", "Portfolio_Summary", default=[])

    income, income_original = parse_income_range(employment.get("Monthly_Income_Range"))
    dob = details.get("Date_of_Birth")
    warnings: list[str] = []
    if not dob:
        raise ValueError("Date_of_Birth is required to calculate age")
    age = calculate_age(dob)

    annual_expenses = _number(cash_flow.get("Estimated_total_annual_expenses"))
    monthly_expenses = annual_expenses / 12 if annual_expenses is not None else None
    if monthly_expenses is None:
        warnings.append("Monthly expenses are unavailable; emergency-fund gap is not calculated.")

    death_tpd = 0.0
    critical_illness = 0.0
    premiums = 0.0
    for item in portfolio if isinstance(portfolio, list) else []:
        benefit = _number(item.get("Total_Benefit_Amount")) or 0.0
        benefit_type = str(item.get("Types_of_Benefit", "")).lower()
        if "critical" in benefit_type:
            critical_illness += benefit
        if "death" in benefit_type or "tpd" in benefit_type or "permanent" in benefit_type:
            death_tpd += benefit
        premium_text = str(item.get("Annual_Premium", "")).lower()
        premium = _number(premium_text) or 0.0
        premiums += premium if "monthly" in premium_text or "/month" in premium_text else premium / 12

    take_home = (
        _number(cash_flow.get("Remarks"))
        if "take-home" in str(cash_flow.get("Remarks", "")).lower()
        else None
    )
    if take_home is None:
        warnings.append("Monthly take-home pay is unavailable; percentage budget checks are limited. "
                        "Single_Amount is not treated as take-home pay.")

    investments = None
    if budget.get("Annual_Amount") not in (None, ""):
        investments = (_number(budget.get("Annual_Amount")) or 0.0) / 12
    if investments is None:
        warnings.append("Monthly investments are unavailable; investment gap is not calculated.")

    client_context = {
        "occupation": employment.get("Current_Occupation"),
        "employment_status": employment.get("Employment_Status"),
        "personal_priorities": {
            key: value for key, value in priorities.items() if value not in (None, "")
        },
        "dependant_relations": [
            person.get("Relation") for person in _get(
                personal,
                "c_Details_of_Spouse_and_Dependants",
                default=[],
            ) if isinstance(person, dict) and person.get("Relation")
        ],
    }

    profile_model = ClientProfile(
        client_id=details.get("NRIC_Passport_No") or details.get("Full_Name") or "anonymous",
        age=age,
        marital_status=details.get("Marital_Status") or "unspecified",
        dependents=count_dependants(profile),
        monthly_income_range_original=income_original,
        monthly_gross_income=income,
        monthly_take_home_pay=take_home,
        monthly_expenses=monthly_expenses,
        current_emergency_cash=_number(assets.get("Emergency_Fund")),
        existing_death_tpd_cover=death_tpd,
        existing_ci_cover=critical_illness,
        current_monthly_premiums=premiums,
        monthly_investments=investments,
        client_context=client_context,
    )
    user_advice = profile.get("Section_B_Our_Advice_and_Reasons_Why")
    return profile_model, warnings, user_advice if isinstance(user_advice, dict) else None