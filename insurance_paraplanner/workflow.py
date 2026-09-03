from __future__ import annotations

from pathlib import Path
import sqlite3
import logging
from typing import Any, TypedDict

from .models import AdvisoryResult, ArchetypeMatch, ClientProfile, ProductRequirement, Recommendation
from .normalizer import normalize_profile
from .tools import calculate_financial_gaps, filter_policy_metadata, match_archetypes


logger = logging.getLogger("insurance_paraplanner")


class WorkflowState(TypedDict, total=False):
    profile_json: dict[str, Any]
    database_path: str
    chroma_path: str
    profile: ClientProfile
    normalization_warnings: list[str]
    classification_warnings: list[str]
    user_provided_advice: dict[str, Any] | None
    matches: list[dict[str, Any]]
    benchmark: dict[str, Any]
    benchmark_summary: dict[str, Any]
    gap_warnings: list[str]
    requirements: list[ProductRequirement]
    recommendations: list[Recommendation]
    recommendation_warnings: list[str]
    agent_observations: list[str]
    result: dict[str, Any]


def _requirements(gaps: dict[str, Any]) -> list[ProductRequirement]:
    budget = gaps.get("protection_budget_remaining")
    requirements: list[ProductRequirement] = []
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


def normalize_node(state: WorkflowState) -> WorkflowState:
    logger.info("stage=normalize_started")
    profile, warnings, advice = normalize_profile(state["profile_json"])
    logger.info("stage=normalize_finished age=%s dependents=%s warnings=%d", profile.age, profile.dependents, len(warnings))
    return {
        "profile": profile,
        "normalization_warnings": warnings,
        "user_provided_advice": advice,
    }


def classify_node(state: WorkflowState) -> WorkflowState:
    logger.info("stage=classify_started")
    observations: list[str] = []
    classification_warnings: list[str] = []
    ai_selection: str | None = None
    try:
        from .agents import run_classifier_agent

        classifier_output = run_classifier_agent(
            state["profile"],
            state["database_path"],
            state["chroma_path"],
        )
        ai_selection = classifier_output.classified_archetype
        observations.append(classifier_output.model_dump_json())
    except Exception as error:
        classification_warnings.append(f"Classifier AI selection unavailable: {error}")
        observations.append(f"Classifier ReAct agent unavailable: {error}")
    with sqlite3.connect(state["database_path"]) as connection:
        matches = match_archetypes(
            connection,
            state["profile"].age,
            state["profile"].marital_status,
            state["profile"].dependents,
        )
    if not matches:
        logger.warning("stage=classify_finished matches=0")
        return {
            "matches": [],
            "benchmark": {},
            "agent_observations": observations,
            "classification_warnings": classification_warnings,
        }
    if ai_selection:
        selected = next((item for item in matches if item["archetype_id"] == ai_selection), None)
        if selected is not None:
            matches = [selected] + [item for item in matches if item["archetype_id"] != ai_selection]
            logger.info("classifier_ai_selection_validated selected=%s", ai_selection)
        else:
            classification_warnings.append(
                f"Classifier selected invalid archetype {ai_selection}; deterministic candidate retained."
            )
            logger.warning("classifier_ai_selection_rejected selected=%s", ai_selection)
    logger.info("stage=classify_finished matches=%d selected=%s", len(matches), matches[0]["archetype_id"])
    return {
        "matches": matches,
        "benchmark": matches[0]["benchmarks"],
        "agent_observations": observations,
        "classification_warnings": classification_warnings,
    }


def calculate_node(state: WorkflowState) -> WorkflowState:
    logger.info("stage=calculate_started")
    if not state["matches"]:
        return {"benchmark_summary": {}, "gap_warnings": [], "requirements": []}
    summary, warnings = calculate_financial_gaps(state["profile"], state["benchmark"])
    logger.info("stage=calculate_finished gaps=%d warnings=%d", sum(1 for key, value in summary.items() if key.endswith("_gap") and value > 0), len(warnings))
    return {
        "benchmark_summary": summary,
        "gap_warnings": warnings,
        "requirements": _requirements(summary),
    }


def recommend_node(state: WorkflowState) -> WorkflowState:
    logger.info("stage=recommend_started requirements=%d", len(state.get("requirements", [])))
    if not state["requirements"]:
        return {"recommendations": [], "recommendation_warnings": []}
    recommendations: list[Recommendation] = []
    warnings: list[str] = []
    observations = list(state.get("agent_observations", []))
    try:
        from .agents import run_recommender_agent
        import chromadb

        agent_output = run_recommender_agent(
            state["profile"],
            state["requirements"],
            state["database_path"],
            state["chroma_path"],
        )
        observations.append(agent_output.model_dump_json())
        with sqlite3.connect(state["database_path"]) as connection:
            candidates_by_requirement = {
                requirement.requirement_id: {
                    item["policy_id"]: item
                    for item in filter_policy_metadata(
                        connection,
                        requirement.product_category,
                        state["profile"].age,
                    )
                }
                for requirement in state["requirements"]
            }
        for recommendation in agent_output.recommendations:
            candidates = candidates_by_requirement.get(recommendation.requirement_id, {})
            if recommendation.policy_id not in candidates:
                logger.warning("recommendation_rejected policy_id=%s reason=not_sql_candidate", recommendation.policy_id)
                warnings.append(
                    f"Rejected AI recommendation {recommendation.policy_id}: "
                    "policy was not returned by deterministic SQL filtering."
                )
                continue
            if not recommendation.evidence:
                logger.warning("recommendation_rejected policy_id=%s reason=no_clause_evidence", recommendation.policy_id)
                warnings.append(
                    f"Rejected AI recommendation {recommendation.policy_id}: "
                    "no policy clause evidence was supplied."
                )
                continue
            recommendations.append(recommendation)
        warnings.extend(agent_output.warnings)
    except ImportError:
        warnings.append("ReAct dependencies are not installed; AI recommendations were not generated.")
    except Exception as error:
        warnings.append(f"Policy clause retrieval unavailable: {error}")
    return {
        "recommendations": recommendations,
        "recommendation_warnings": warnings,
        "agent_observations": observations,
    }


def finalize_node(state: WorkflowState) -> WorkflowState:
    logger.info("stage=finalize_started")
    if not state["matches"]:
        result = AdvisoryResult(
            validation_warnings=(
                state.get("normalization_warnings", [])
                + state.get("classification_warnings", [])
                + ["No age-compatible archetype was found."]
            ),
            user_provided_advice=state.get("user_provided_advice"),
            agent_observations=state.get("agent_observations", []),
        )
        return {"result": result.model_dump()}

    matches = state["matches"]
    summary = state.get("benchmark_summary", {})
    result = AdvisoryResult(
        classified_archetype=matches[0]["archetype_id"],
        archetype_candidates=[ArchetypeMatch(
            archetype_id=item["archetype_id"],
            archetype_name=item["archetype_name"],
            match_score=item["match_score"],
            benchmarks=item["benchmarks"],
        ) for item in matches],
        benchmark_summary=summary,
        identified_gaps=[key for key, value in summary.items() if key.endswith("_gap") and value > 0],
        product_specifications=state.get("requirements", []),
        final_recommendations=state.get("recommendations", []),
        assumptions=["Income ranges are represented by their midpoint.", "Recommendations are advisory only."],
        validation_warnings=(
            state.get("normalization_warnings", [])
            + state.get("classification_warnings", [])
            + state.get("gap_warnings", [])
            + state.get("recommendation_warnings", [])
        ),
        user_provided_advice=state.get("user_provided_advice"),
        agent_observations=state.get("agent_observations", []),
    )
    logger.info("stage=finalize_finished recommendations=%d", len(result.final_recommendations))
    return {"result": result.model_dump()}


def build_workflow():
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(WorkflowState)
    graph.add_node("normalize", normalize_node)
    graph.add_node("classify", classify_node)
    graph.add_node("calculate", calculate_node)
    graph.add_node("recommend", recommend_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "normalize")
    graph.add_edge("normalize", "classify")
    graph.add_edge("classify", "calculate")
    graph.add_edge("calculate", "recommend")
    graph.add_edge("recommend", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=MemorySaver())


def run_workflow(
    profile_json: dict[str, Any],
    database_path: str | Path,
    chroma_path: str | Path,
) -> dict[str, Any]:
    workflow = build_workflow()
    state = workflow.invoke(
        {
            "profile_json": profile_json,
            "database_path": str(database_path),
            "chroma_path": str(chroma_path),
        },
        config={"configurable": {"thread_id": "advisory-run"}, "recursion_limit": 10},
    )
    return state["result"]
