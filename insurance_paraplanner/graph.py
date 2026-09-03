from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .models import AdvisoryResult, ArchetypeMatch, ClientProfile, ProductRequirement, Recommendation
from .normalizer import normalize_profile
from .observability import configure_logging
from .tools import calculate_financial_gaps, filter_policy_metadata, match_archetypes, search_policy_clauses


DEFAULT_DATABASE = Path(__file__).parents[1] / "data ingestion" / "policies.db"
DEFAULT_CHROMA = Path(__file__).parents[1] / "data ingestion" / "chroma_db"


def _requirements(profile: ClientProfile, gaps: dict[str, Any], benchmarks: dict[str, Any]) -> list[ProductRequirement]:
    requirements = []
    budget = gaps.get("protection_budget_remaining")
    if gaps.get("death_tpd_gap", 0) > 0:
        requirements.append(ProductRequirement(
            requirement_id="REQ-DEATH-TPD",
            target_need="Mortality",
            product_category="Term Life",
            min_sum_assured=gaps["death_tpd_gap"],
            max_monthly_premium=budget,
            required_features=["Death", "TPD"],
        ))
    if gaps.get("ci_gap", 0) > 0:
        requirements.append(ProductRequirement(
            requirement_id="REQ-CI",
            target_need="Critical Illness",
            product_category="Term Life",
            min_sum_assured=gaps["ci_gap"],
            max_monthly_premium=budget,
            required_features=["Critical Illness"],
        ))
    return requirements


def _recommend(requirement: ProductRequirement, profile: ClientProfile, collection: Any, connection: sqlite3.Connection, benchmark: dict[str, Any]) -> tuple[list[Recommendation], list[str]]:
    warnings: list[str] = []
    candidates = filter_policy_metadata(connection, requirement.product_category, profile.age)
    if requirement.max_monthly_premium is None:
        warnings.append(
            f"{requirement.requirement_id}: recommendation withheld because premium "
            "affordability cannot be verified from the available policy data."
        )
        return [], warnings
    recommendations: list[Recommendation] = []
    for candidate in candidates[:5]:
        evidence = search_policy_clauses(collection, [candidate["policy_id"]], "benefits, exclusions, waiting periods, and eligibility", 2)
        recommendations.append(Recommendation(
            requirement_id=requirement.requirement_id,
            policy_id=candidate["policy_id"],
            product_name=candidate["product_name"],
            category=candidate["category"],
            evidence=evidence,
            validation_warnings=list(warnings),
        ))
    if not recommendations:
        warnings.append(f"{requirement.requirement_id}: no eligible policy metadata matched the age and category constraints.")
    return recommendations, warnings


def run_advisory(
    profile_json: dict[str, Any],
    database_path: str | Path = DEFAULT_DATABASE,
    chroma_path: str | Path = DEFAULT_CHROMA,
) -> dict[str, Any]:
    from dotenv import load_dotenv
    from .workflow import run_workflow

    load_dotenv(Path(__file__).parents[1] / ".env")
    logger, log_path = configure_logging()
    logger.info("advisory_started database=%s chroma=%s", database_path, chroma_path)
    result = run_workflow(profile_json, database_path, chroma_path)
    logger.info(
        "advisory_finished archetype=%s recommendations=%d warnings=%d",
        result.get("classified_archetype"),
        len(result.get("final_recommendations", [])),
        len(result.get("validation_warnings", [])),
    )
    result["log_file"] = str(log_path)
    return result
