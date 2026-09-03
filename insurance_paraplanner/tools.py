from __future__ import annotations

import sqlite3
from typing import Any


def _fetch_one(connection: sqlite3.Connection, query: str, values: tuple[Any, ...]) -> dict[str, Any] | None:
    connection.row_factory = sqlite3.Row
    row = connection.execute(query, values).fetchone()
    return dict(row) if row else None


def match_archetypes(
    connection: sqlite3.Connection,
    age: int,
    marital_status: str,
    dependents: int,
) -> list[dict[str, Any]]:
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT a.*, b.emergency_fund_months_min, b.emergency_fund_months_max,
               b.emergency_fund_irregular_months, b.death_tpd_income_multiplier,
               b.ci_income_multiplier, b.max_insurance_spend_percent,
               b.min_investment_percent
        FROM archetypes a
        LEFT JOIN archetype_benchmarks b ON b.archetype_id = a.archetype_id
        WHERE ? BETWEEN a.target_age_min AND a.target_age_max
        ORDER BY a.target_age_min DESC
        """,
        (age,),
    ).fetchall()
    status = marital_status.lower()
    matches = []
    for row in rows:
        item = dict(row)
        searchable = f"{item['archetype_name']} {item['family_and_dependent_context']}".lower()
        score = 0
        if dependents and any(word in searchable for word in ("family", "child", "parent", "dependent")):
            score += 3
        if "married" in status and any(word in searchable for word in ("family", "married", "spouse")):
            score += 3
        if any(word in status for word in ("single", "unmarried")) and any(
            word in searchable for word in ("fresh", "single", "workforce")
        ):
            score += 3
        if "parent" in status or "care" in status:
            if any(word in searchable for word in ("parent", "care")):
                score += 3
        item["match_score"] = score
        item["benchmarks"] = {
            key: item[key]
            for key in (
                "emergency_fund_months_min",
                "emergency_fund_months_max",
                "emergency_fund_irregular_months",
                "death_tpd_income_multiplier",
                "ci_income_multiplier",
                "max_insurance_spend_percent",
                "min_investment_percent",
            )
        }
        matches.append(item)
    return sorted(matches, key=lambda item: (item["match_score"], item["target_age_min"]), reverse=True)


def calculate_financial_gaps(profile: Any, benchmarks: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    summary: dict[str, Any] = {}
    warnings: list[str] = []
    expenses = profile.monthly_expenses
    if expenses is not None and profile.current_emergency_cash is not None:
        months = benchmarks.get("emergency_fund_months_max") or benchmarks.get("emergency_fund_months_min")
        if months is not None:
            summary["emergency_target"] = months * expenses
            summary["emergency_gap"] = max(0, summary["emergency_target"] - profile.current_emergency_cash)
    else:
        warnings.append("Emergency-fund gap unavailable because expenses or emergency cash is missing.")

    annual_income = (profile.monthly_gross_income or profile.monthly_take_home_pay)
    if annual_income is not None:
        annual_income *= 12
    multiplier = benchmarks.get("death_tpd_income_multiplier")
    if annual_income is not None and multiplier is not None and profile.existing_death_tpd_cover is not None:
        summary["death_tpd_target"] = annual_income * multiplier
        summary["death_tpd_gap"] = max(0, summary["death_tpd_target"] - profile.existing_death_tpd_cover)
    else:
        warnings.append("Death/TPD gap unavailable because income, benchmark, or existing cover is missing.")

    multiplier = benchmarks.get("ci_income_multiplier")
    if annual_income is not None and multiplier is not None and profile.existing_ci_cover is not None:
        summary["ci_target"] = annual_income * multiplier
        summary["ci_gap"] = max(0, summary["ci_target"] - profile.existing_ci_cover)
    else:
        warnings.append("Critical-illness gap unavailable because income, benchmark, or existing cover is missing.")

    take_home = profile.monthly_take_home_pay
    spend_percent = benchmarks.get("max_insurance_spend_percent")
    if take_home is not None and spend_percent is not None:
        summary["max_protection_budget"] = take_home * spend_percent / 100
        if profile.current_monthly_premiums is not None:
            summary["protection_budget_remaining"] = max(
                0, summary["max_protection_budget"] - profile.current_monthly_premiums
            )
    else:
        warnings.append("Protection budget unavailable because take-home pay or benchmark is missing.")

    investment_percent = benchmarks.get("min_investment_percent")
    if take_home is not None and investment_percent is not None and profile.monthly_investments is not None:
        summary["investment_target"] = take_home * investment_percent / 100
        summary["investment_gap"] = max(0, summary["investment_target"] - profile.monthly_investments)
    else:
        warnings.append("Investment gap unavailable because investments, pay, or benchmark is missing.")
    return summary, warnings


def filter_policy_metadata(
    connection: sqlite3.Connection,
    category: str,
    client_age: int,
) -> list[dict[str, Any]]:
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT * FROM policies
        WHERE category LIKE ?
          AND (min_entry_age IS NULL OR min_entry_age <= ?)
          AND (max_entry_age IS NULL OR max_entry_age >= ?)
        ORDER BY product_name
        """,
        (f"%{category}%", client_age, client_age),
    ).fetchall()
    return [dict(row) for row in rows]


def search_policy_clauses(collection: Any, policy_ids: list[str], query: str, limit: int = 3) -> list[dict[str, Any]]:
    if not policy_ids:
        return []
    result = collection.query(query_texts=[query], n_results=limit, where={"policy_id": {"$in": policy_ids}})
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    return [{"text": text, "metadata": metadata} for text, metadata in zip(documents, metadatas)]


def search_archetype_guidance(collection: Any, archetype_id: str, query: str, limit: int = 3) -> list[dict[str, Any]]:
    result = collection.query(
        query_texts=[query],
        n_results=limit,
        where={"archetype_id": archetype_id},
    )
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    return [{"text": text, "metadata": metadata} for text, metadata in zip(documents, metadatas)]


def validate_budget_cap(take_home_pay: float, proposed_premium: float, percent: float) -> bool:
    return proposed_premium <= take_home_pay * percent / 100


def validate_age_eligibility(client_age: int, min_age: int | None, max_age: int | None) -> bool:
    return (min_age is None or client_age >= min_age) and (max_age is None or client_age <= max_age)
