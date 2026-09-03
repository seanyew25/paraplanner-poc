Here is a comprehensive Product Requirements Document (PRD) and technical specification designed specifically for a coding assistant or software engineering team. It integrates robust system reliability principles, strict typing, and clear state management to ensure a production-grade deployment.

---

# Multi-Agent Financial Advisory System: Technical Specification

## 1. System Architecture & Framework

This system utilizes a hierarchical multi-agent architecture built on **LangGraph**. It transitions away from standard linear pipelines into a cyclical, state-driven control loop where a central Supervisor orchestrates specialized worker agents.

- **Framework:** LangGraph (Python) via the `StateGraph` or `langgraph-supervisor` implementation.
- **State Management:** Strongly typed Pydantic models passed through the graph state to prevent LLM formatting hallucinations.
- **Databases:**
- _Relational (SQL):_ SQLite for deterministic archetype matching, benchmark calculations,
  and policy metadata filtering.
- _Vector (Semantic):_ ChromaDB in `chunker/chroma_db`. The `archetype_guides` collection
  stores contextual archetype guidance and `policy_clauses` stores policy wording.

- **LLM Provider:** Google Gemini API (via `google-genai` SDK) utilizing Structured Outputs for precise schema adherence.

---

## 2. Input Contract, Graph State & Data Models (Pydantic)

To ensure reliable handoffs between agents, the global graph state must maintain a strict
schema. The public entry point accepts the existing profile JSON and normalizes it before
running the graph. Date of birth is authoritative for age. Income ranges use their
midpoint for calculations, while the original range is retained for auditability.
Missing financial values must be reported as unavailable rather than silently inferred.
The profile form will provide `Emergency_Savings` explicitly; the input adapter must map
it to `current_emergency_cash` and must not estimate it from assets, CPF, or investments.

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class ClientProfile(BaseModel):
    client_id: str
    age: int  # Derived from Date_of_Birth; not the authoritative input.
    marital_status: str
    dependents: int = Field(ge=0)
    monthly_income_range_original: Optional[str] = None
    monthly_gross_income: Optional[float] = None
    monthly_take_home_pay: float
    monthly_expenses: float
    current_emergency_cash: float  # Mapped from the form's Emergency_Savings field.
    existing_death_tpd_cover: float
    existing_ci_cover: float
    current_monthly_premiums: float
    monthly_investments: float

class ProductRequirement(BaseModel):
    requirement_id: str
    target_need: str = Field(description="'Mortality', 'Critical Illness', or 'Wealth'")
    product_category: str = Field(description="'Term Life', 'Whole Life', or 'Investment'")
    min_sum_assured: float
    max_monthly_premium: float
    required_features: List[str]

class GraphState(BaseModel):
    client_profile: ClientProfile
    classified_archetype: Optional[str] = None
    archetype_candidates: List[str] = Field(default_factory=list)
    identified_gaps: List[str] = Field(default_factory=list)
    product_specifications: List[ProductRequirement] = Field(default_factory=list)
    final_recommendations: List[dict] = Field(default_factory=list)
    user_provided_advice: Optional[dict] = None
    messages: list = Field(default_factory=list)

def run_advisory(profile_json: dict) -> dict:
    """Public JSON-in/JSON-out entry point for the agentic workflow."""

```

The input adapter must extract the date of birth, marital status, spouse/dependant
count, `Emergency_Savings`, cash-flow values, existing insurance portfolio, and the
user-provided Section B advice. `Emergency_Savings` is the client's current emergency
fund balance and must be normalized to a numeric amount. Section B may be analyzed for
consistency, but agents must not overwrite it or present generated text as the user's
original advice.

---

## 3. Agent Node Specifications

### Agent 1: The Supervisor (Manager)

- **Role:** The central router and state manager.
- **Execution Logic:**

1. Receives the raw profile JSON.
2. Validates and normalizes the profile, deriving age from date of birth and using
   midpoint income for calculations.
3. Routes the state to the `Classifier Agent`.
4. Receives the `product_specifications` from the Classifier.
5. Routes the state to the `Recommender Agent`.
6. Validates recommendations and compiles the final advisory payload.
7. Terminates the graph loop with an advisory-only disclaimer.

- **Routing Mechanism:** Utilizes LangGraph's conditional edges (`add_conditional_edges`) to route based on the presence of data in the `GraphState` (e.g., if `product_specifications` is empty, go to Classifier; if populated, go to Recommender).

### Agent 2: The Classifier (Gap Analysis)

- **Role:** Diagnostic engine responsible for mathematical gap analysis and categorization.
- **System Prompt Directive:** "You are a diagnostic financial analyst. Classify the user based on the MAS Basic Financial Planning Guide. Calculate exact numerical shortfalls."
- **Tools Provided:**
- `retrieve_archetype_rules(age: int, marital_status: str, dependents: int)`: Queries
  SQLite for age-compatible archetypes, then ranks overlapping matches using marital
  status and dependant count. It retrieves benchmarks from `archetype_benchmarks`.
- `retrieve_archetype_guidance(archetype_id: str, query: str)`: Searches ChromaDB
  collection `archetype_guides` after deterministic classification.

- **Expected Output:** Mutates the `GraphState` by populating the `classified_archetype`, `identified_gaps`, and generating strict `ProductRequirement` contracts.

### Agent 3: The Recommender (Search & Match)

- **Role:** Execution engine that matches the Classifier's contract against real-world policy documents.
- **System Prompt Directive:** "You are a product matching specialist. You must strictly adhere to the budget and category constraints provided in the ProductRequirement contract."
- **Tools Provided (The Dual-Layer Search):**
- `sql_metadata_filter(category: str, max_premium: float, max_age: int)`: Executes a deterministic SQL query to return a list of valid `policy_id`s (e.g., filtering out participating plans if pure term is requested).

- `vector_clause_search(policy_id_list: List[str], query: str)`: Executes a semantic
  search in ChromaDB collection `policy_clauses`, strictly filtered by the `policy_id`s
  returned from the SQL step to verify specific exclusions or waiting periods.

- **Expected Output:** Mutates the `GraphState` by appending verified product matches to `final_recommendations`.

---

## 4. Deterministic Guardrails & Validation

To ensure high system reliability and compliance, AI reasoning must be sandboxed by deterministic Python functions. The coding assistant should implement these as standard functions that run _between_ agent nodes.

- **Premium Cap Validator:**

```python
def validate_budget_cap(take_home_pay: float, proposed_premium: float) -> bool:
    # Ensures total protection spend is at most 15% of take-home pay
    return proposed_premium <= (take_home_pay * 0.15)

```

- **Investment Threshold Check:** Verifies that at least 10% of the take-home pay is allocated to investments.
- **Archetype Benchmark Check:** Calculates emergency fund, Death/TPD, critical illness,
  insurance-spend, and investment gaps from the matched SQLite benchmark.

- **Graph Interruption:** If a recommended product violates the `validate_budget_cap`, the Supervisor must catch the exception and route the state back to the Recommender Agent with an error message, forcing it to search for a cheaper alternative before surfacing the output to the user.
- **Advisory Boundary:** Recommendations are advisory only. The system must not claim to
  provide regulated financial advice, guarantee suitability, or replace a licensed
  financial planner. Final output must identify assumptions, missing data, source clauses,
  and the need for human review.

---

## 5. Telemetry and Error Handling

For proper lifecycle management and debugging of the autonomous loops, the implementation must include:

- **LangSmith / LangGraph Checkpointers:** Implement `MemorySaver` to persist the graph state at each step. This allows for "time-travel" debugging if the Recommender Agent hallucinates a product match.
- **Iteration Caps:** The Supervisor must have a `recursion_limit` (e.g., `max_steps=5`) to prevent infinite loops if the Recommender cannot find a product that fits the budget.

## 6. Public Execution Contract

The first implementation is a Python function, independent of a web UI:

```python
result = run_advisory(profile_json)
```

The returned JSON must contain `classified_archetype`, `archetype_candidates`,
`benchmark_summary`, `identified_gaps`, `product_specifications`,
`final_recommendations`, `assumptions`, `validation_warnings`, and
`advisory_disclaimer`.

Recommendations must be generated by the AI Recommender Agent after it uses verified
SQLite metadata and matching policy clauses. Each recommendation must include a rationale,
fit notes, risks, policy ID, and retrieved evidence. The Supervisor must validate every
AI-proposed policy ID against SQLite candidates and reject recommendations without clause
evidence or with unmet deterministic constraints. If no product satisfies the constraints,
return no product recommendation and explain the unmet requirement instead of relaxing it.

No REST API or frontend is required in the first implementation. A future adapter may
call `run_advisory` from FastAPI without changing the agent graph or database contracts.

## 7. Agent Tools

The agents must use deterministic Python tools for calculations and database access.
No separate calculator agent or external calculator service is required.

### Classifier Tools

```python
calculate_age(date_of_birth)
parse_income_range(income_range)
count_dependants(profile_json)
match_archetype(age, marital_status, dependents)
get_archetype_benchmarks(archetype_id)
calculate_financial_gaps(profile, benchmarks)
```

`parse_income_range` must use the midpoint for calculations and preserve the original
income range. `calculate_age` must derive age from date of birth. Archetype matching must
use age, marital status, and dependant count, especially when age ranges overlap.
The normalizer must map the form's `Emergency_Savings` value to
`ClientProfile.current_emergency_cash`.

### Recommender Tools

```python
filter_policy_metadata(category, max_monthly_premium, client_age)
search_policy_clauses(policy_ids, query)
search_archetype_guidance(archetype_id, query)
```

The Recommender Agent is a bounded ReAct agent using Gemini Flash Lite. Its final
structured output is the source of policy recommendations, but it is not trusted on its
own: the Supervisor performs deterministic candidate, evidence, age, feature, and budget
validation before including any recommendation in the final result.

`filter_policy_metadata` queries SQLite and returns eligible policy IDs. The policy IDs
must then constrain `search_policy_clauses` in ChromaDB collection `policy_clauses`.
`search_archetype_guidance` searches ChromaDB collection `archetype_guides` after the
Classifier has selected an archetype.

### Validation Tools

```python
validate_budget_cap(take_home_pay, proposed_premium)
validate_sum_assured(required_amount, proposed_amount)
validate_age_eligibility(client_age, min_age, max_age)
validate_required_features(policy_text, required_features)
validate_recommendation(recommendation, client_profile)
```

The budget validator must enforce the matched archetype's insurance-spend benchmark.
Validation failures must prevent the recommendation from reaching the final response.

### Financial Calculations

```python
emergency_target = target_months * monthly_expenses
emergency_gap = max(0, emergency_target - current_emergency_cash)

death_tpd_target = annual_income * death_tpd_multiplier
death_tpd_gap = max(0, death_tpd_target - existing_death_tpd_cover)

ci_target = annual_income * ci_multiplier
ci_gap = max(0, ci_target - existing_ci_cover)

max_protection_budget = monthly_take_home_pay * insurance_spend_percent / 100
investment_target = monthly_take_home_pay * investment_percent / 100
investment_gap = max(0, investment_target - monthly_investments)
```

All calculations must be deterministic, auditable, and based on normalized profile data
and SQLite benchmarks. Missing inputs must produce an explicit warning rather than an
invented value.
