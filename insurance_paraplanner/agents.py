from __future__ import annotations

import json
import os
import sqlite3
from typing import Any

from .models import ClassifierOutput, ClientProfile, ProductRequirement, RecommenderOutput
from .tools import (
    calculate_financial_gaps,
    filter_policy_metadata,
    match_archetypes,
    search_archetype_guidance,
    search_policy_clauses,
)
import logging
from .observability import AgentTerminalLogger


logger = logging.getLogger("insurance_paraplanner")


CLASSIFIER_PROMPT = """
You are the Classifier Agent for an insurance planning workflow.
Use tools to inspect the supplied client profile and determine the best archetype.
You may select only from the returned SQLite archetype records. Use the benchmark
values returned by the tools; never invent thresholds or financial values.
Use the archetype ChromaDB search when life-stage context is useful. Return a concise JSON object with keys: classified_archetype, candidate_ids,
benchmark_summary, identified_gaps, and product_specifications. Product specifications
must represent gaps only. Do not recommend products. This is advisory analysis only.
Use the client's JSON-derived context and dependent context when searching archetype guidance.
Compare the returned guidance with the client's priorities before choosing among valid
SQLite candidates. The ChromaDB guidance informs the selection, but SQLite remains the
authority for valid archetype IDs and benchmark values.
""".strip()

RECOMMENDER_PROMPT = """
You are the Recommender Agent for an insurance planning workflow.
For each product requirement, use SQLite metadata filtering first, then use ChromaDB
policy clause search to inspect benefits, exclusions, waiting periods, and eligibility.
Ignore the client's budget as a candidate-search filter because the current policy data
does not contain personalized premium quotations. Find policies that address the required
coverage and are age/category eligible. Do not evaluate, mention, or use affordability,
budget, or premium information. Never invent a premium. Never relax a coverage requirement.
Return a concise JSON object with a recommendations array and
warnings array. Each recommendation must include requirement_id, policy_id, product_name,
    category, rationale, fit_notes, risks, and evidence. The evidence field must be a list of
    objects with text and source keys. Respond only in English. Do not include
products that were not returned by the filter tool. This is advisory analysis only and
requires human review.
""".strip()


def _client(api_key: str):
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=api_key,
        temperature=0,
        max_retries=2,
        callbacks=[AgentTerminalLogger(logger)],
    )


def _api_key() -> str:
    api_key = os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API_KEY or GEMINI_API_KEY environment variable")
    return api_key


def _message_text(content: Any) -> str:
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("text"):
                parts.append(str(part["text"]))
        return "\n".join(parts)
    return str(content)


def _classifier_tools(database_path: str, chroma_path: str):
    from langchain_core.tools import tool
    import chromadb

    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(name="archetype_guides")

    @tool
    def retrieve_archetype_rules(age: int, marital_status: str, dependents: int) -> str:
        """Find age-compatible archetypes and rank them using family context."""
        logger.info("classifier_tool=retrieve_archetype_rules age=%s dependents=%s", age, dependents)
        with sqlite3.connect(database_path) as connection:
            result = json.dumps(match_archetypes(connection, age, marital_status, dependents), default=str)
            logger.info("classifier_tool=retrieve_archetype_rules result_chars=%d", len(result))
            return result

    @tool
    def calculate_gaps(profile_json: str, benchmarks_json: str) -> str:
        """Calculate deterministic financial gaps from a normalized profile and benchmarks."""
        logger.info("classifier_tool=calculate_gaps")
        profile = ClientProfile.model_validate_json(profile_json)
        summary, warnings = calculate_financial_gaps(profile, json.loads(benchmarks_json))
        return json.dumps({"summary": summary, "warnings": warnings})

    @tool
    def search_archetype_context(archetype_id: str, query: str) -> str:
        """Retrieve contextual guidance for a selected archetype."""
        logger.info("classifier_tool=search_archetype_context archetype=%s query=%s", archetype_id, query)
        result = json.dumps(search_archetype_guidance(collection, archetype_id, query), default=str)
        logger.info("classifier_tool=search_archetype_context result_chars=%d", len(result))
        return result

    return [retrieve_archetype_rules, calculate_gaps, search_archetype_context]


def _recommender_tools(database_path: str, chroma_path: str):
    from langchain_core.tools import tool
    import chromadb

    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(name="policy_clauses")

    @tool
    def filter_policies(category: str, client_age: int) -> str:
        """Return policy metadata eligible by category and age."""
        logger.info("recommender_tool=filter_policies category=%s age=%s", category, client_age)
        with sqlite3.connect(database_path) as connection:
            return json.dumps(filter_policy_metadata(connection, category, client_age), default=str)

    @tool
    def search_clauses(policy_ids_json: str, query: str) -> str:
        """Search policy clauses only within the supplied candidate policy IDs."""
        logger.info("recommender_tool=search_clauses query=%s", query)
        policy_ids = json.loads(policy_ids_json)
        return json.dumps(search_policy_clauses(collection, policy_ids, query), default=str)

    return [filter_policies, search_clauses]


def run_classifier_agent(
    profile: ClientProfile,
    database_path: str,
    chroma_path: str,
    max_steps: int = 28,
) -> ClassifierOutput:
    from langgraph.prebuilt import create_react_agent

    logger.info("classifier_agent_started max_steps=%s", max_steps)
    agent = create_react_agent(
        _client(_api_key()),
        tools=_classifier_tools(database_path, chroma_path),
        prompt=CLASSIFIER_PROMPT,
    )
    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(profile.model_dump(), default=str),
                }
            ]
        },
        config={"recursion_limit": max_steps},
    )
    content = _message_text(response["messages"][-1].content).strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    output = ClassifierOutput.model_validate_json(content)
    logger.info("classifier_agent_finished final_response=%s", output.model_dump_json())
    return output


def run_recommender_agent(
    profile: ClientProfile,
    requirements: list[ProductRequirement],
    database_path: str,
    chroma_path: str,
    max_steps: int = 12,
) -> RecommenderOutput:
    from langgraph.prebuilt import create_react_agent

    logger.info("recommender_agent_started requirements=%d max_steps=%s", len(requirements), max_steps)
    agent = create_react_agent(
        _client(_api_key()),
        tools=_recommender_tools(database_path, chroma_path),
        prompt=RECOMMENDER_PROMPT,
    )
    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "profile": profile.model_dump(exclude={
                                "monthly_take_home_pay",
                                "current_monthly_premiums",
                            }),
                            "requirements": [
                                item.model_dump(exclude={"max_monthly_premium"})
                                for item in requirements
                            ],
                        },
                        default=str,
                    ),
                }
            ]
        },
        config={"recursion_limit": max_steps},
    )
    content = _message_text(response["messages"][-1].content).strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    output = RecommenderOutput.model_validate_json(content)
    logger.info("recommender_agent_finished final_response=%s", output.model_dump_json())
    return output
