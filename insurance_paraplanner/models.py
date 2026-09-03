from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ClientProfile(BaseModel):
    client_id: str
    age: int
    marital_status: str
    dependents: int = Field(ge=0)
    monthly_income_range_original: Optional[str] = None
    monthly_gross_income: Optional[float] = None
    monthly_take_home_pay: Optional[float] = None
    monthly_expenses: Optional[float] = None
    current_emergency_cash: Optional[float] = None
    existing_death_tpd_cover: Optional[float] = None
    existing_ci_cover: Optional[float] = None
    current_monthly_premiums: Optional[float] = None
    monthly_investments: Optional[float] = None
    client_context: dict[str, Any] = Field(default_factory=dict)


class ProductRequirement(BaseModel):
    requirement_id: str
    target_need: str
    product_category: str
    min_sum_assured: float = Field(ge=0)
    max_monthly_premium: Optional[float] = Field(default=None, ge=0)
    required_features: list[str] = Field(default_factory=list)


class ArchetypeMatch(BaseModel):
    archetype_id: str
    archetype_name: str
    match_score: int
    benchmarks: dict[str, Any] = Field(default_factory=dict)


class ClassifierOutput(BaseModel):
    classified_archetype: str
    candidate_ids: list[str] = Field(default_factory=list)
    benchmark_summary: dict[str, Any] = Field(default_factory=dict)
    identified_gaps: dict[str, Any] = Field(default_factory=dict)
    product_specifications: dict[str, Any] = Field(default_factory=dict)


class Recommendation(BaseModel):
    requirement_id: str
    policy_id: str
    product_name: str
    category: str
    rationale: str = ""
    fit_notes: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)

    @field_validator("fit_notes", "risks", "validation_warnings", mode="before")
    @classmethod
    def normalize_text_lists(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value

    @model_validator(mode="before")
    @classmethod
    def normalize_evidence(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        if not value.get("evidence") and value.get("retrieved_clause_evidence"):
            evidence = value["retrieved_clause_evidence"]
            value = dict(value)
            value["evidence"] = (
                evidence
                if isinstance(evidence, list)
                else [{"text": str(evidence), "source": "policy_clauses"}]
            )
        return value


class RecommenderOutput(BaseModel):
    recommendations: list[Recommendation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AdvisoryResult(BaseModel):
    classified_archetype: Optional[str] = None
    archetype_candidates: list[ArchetypeMatch] = Field(default_factory=list)
    benchmark_summary: dict[str, Any] = Field(default_factory=dict)
    identified_gaps: list[str] = Field(default_factory=list)
    product_specifications: list[ProductRequirement] = Field(default_factory=list)
    final_recommendations: list[Recommendation] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)
    user_provided_advice: Optional[dict[str, Any]] = None
    agent_observations: list[str] = Field(default_factory=list)
    advisory_disclaimer: str = (
        "This is an educational, advisory output only. It is not regulated financial "
        "advice, a guarantee of suitability, or a replacement for review by a licensed "
        "financial planner."
    )