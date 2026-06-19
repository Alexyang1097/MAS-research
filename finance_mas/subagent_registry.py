from __future__ import annotations

from dataclasses import dataclass

from .schemas import CapabilityProfile


@dataclass(frozen=True)
class SubagentSpec:
    name: str
    description: str
    best_for: str
    capability_tags: tuple[str, ...]
    default_tools: tuple[str, ...]
    operating_principles: tuple[str, ...]


SUBAGENT_REGISTRY: dict[str, SubagentSpec] = {
    "RegulatoryNewsAgent": SubagentSpec(
        name="RegulatoryNewsAgent",
        description=(
            "Analyzes regulatory actions, news, M&A events, litigation, product announcements, and market "
            "messages that may explain or contextualize a finance question."
        ),
        best_for=(
            "event-driven questions, regulatory or legal developments, merger/acquisition context, "
            "market-moving news, and source triangulation outside SEC filings"
        ),
        capability_tags=(
            "regulatory_analysis",
            "news_analysis",
            "mna_event_analysis",
            "market_message_analysis",
            "external_context",
            "source_retrieval",
        ),
        default_tools=("web_search", "parse_html_page", "retrieve_information"),
        operating_principles=(
            "Start by identifying dated external events only when they can affect interpretation of the finance question.",
            "Separate reported financial statement facts from market commentary or news interpretation.",
            "Use web_search for discovery, then parse and ground important sources before making claims.",
            "Explicitly state when no regulatory/news/M&A context appears necessary.",
        ),
    ),
    "SECFilingResearchAgent": SubagentSpec(
        name="SECFilingResearchAgent",
        description=(
            "Researches SEC/EDGAR filings, annual and quarterly reports, exhibits, company announcements, "
            "financial statement data, accounting notes, CIKs, filing periods, and source URLs."
        ),
        best_for=(
            "10-K/10-Q/8-K/proxy research, financial statement extraction, filing metadata lookup, "
            "period and unit verification, and primary-source evidence collection"
        ),
        capability_tags=(
            "sec_filing_research",
            "edgar_lookup",
            "financial_statement_extraction",
            "filing_metadata",
            "period_verification",
            "primary_source_evidence",
        ),
        default_tools=("edgar_search", "parse_html_page", "retrieve_information", "web_search"),
        operating_principles=(
            "Prefer edgar_search for filed facts and use web_search only to resolve identifiers or locate source URLs.",
            "Always preserve filing type, fiscal period, units, table/section location, and source URL.",
            "Do not infer operational impact beyond what filing evidence supports.",
            "Flag ambiguity between consolidated, segment, GAAP, non-GAAP, quarterly, annual, and TTM values.",
        ),
    ),
    "OperationalImpactAgent": SubagentSpec(
        name="OperationalImpactAgent",
        description=(
            "Assesses operational impact on revenue, costs, margins, business risk, market exposure, supply, "
            "demand, customer concentration, segments, and competitive position."
        ),
        best_for=(
            "reasoning from evidence to business impact, revenue/cost/risk implications, segment or market "
            "effects, and sanity-checking whether numeric findings make operational sense"
        ),
        capability_tags=(
            "operational_impact",
            "revenue_analysis",
            "cost_analysis",
            "business_risk",
            "market_impact",
            "segment_analysis",
            "financial_reasoning",
        ),
        default_tools=("retrieve_information", "web_search", "parse_html_page"),
        operating_principles=(
            "Ground operational conclusions in already retrieved evidence before adding outside context.",
            "Discuss revenue, cost, margin, segment, risk, or market implications only when linked to evidence.",
            "Do not create new numeric values; if a numeric implication is needed, cite the upstream value and formula.",
            "Separate direct disclosed impact from plausible but uncertain business interpretation.",
        ),
    ),
    "NumericalVerificationAgent": SubagentSpec(
        name="NumericalVerificationAgent",
        description=(
            "Verifies numeric values, formulas, units, fiscal periods, signs, denominators, and rounding. "
            "It does not search broadly for new facts; it audits whether the current numeric chain is correct."
        ),
        best_for=(
            "calculation checks, unit and period validation, ratio/growth/margin verification, "
            "rounding checks, and detecting arithmetic or denominator mistakes"
        ),
        capability_tags=(
            "numerical_verification",
            "formula_audit",
            "unit_checking",
            "period_checking",
            "rounding_check",
            "arithmetic_verification",
        ),
        default_tools=("retrieve_information", "parse_html_page"),
        operating_principles=(
            "Audit the numeric chain rather than broaden the factual search.",
            "Check source value, formula, denominator, sign, fiscal period, units, scaling, and final rounding.",
            "Report any mismatch as a concrete issue with the affected value or formula.",
            "If evidence is insufficient for verification, say exactly which value/source is missing.",
        ),
    ),
    "SourceGroundingAgent": SubagentSpec(
        name="SourceGroundingAgent",
        description=(
            "Audits evidence quality, source hierarchy, source-to-claim alignment, traceability, and conflicts "
            "between SEC filings, company releases, news, and secondary sources."
        ),
        best_for=(
            "checking whether claims are directly supported, preferring primary sources, detecting weak citations, "
            "aligning quotes/locations to claims, and resolving source conflicts"
        ),
        capability_tags=(
            "source_grounding",
            "evidence_audit",
            "source_hierarchy",
            "claim_support_check",
            "conflict_resolution",
            "traceability",
        ),
        default_tools=("retrieve_information", "parse_html_page", "edgar_search", "web_search"),
        operating_principles=(
            "Rank sources by hierarchy: SEC filing and primary company source first, then reputable news, then secondary summaries.",
            "For each important claim, check whether the cited source directly supports it.",
            "Surface source conflicts instead of smoothing them over.",
            "Prefer precise source locations, such as filing section/table names or parsed document keys.",
        ),
    ),
    "ContradictionAgent": SubagentSpec(
        name="ContradictionAgent",
        description=(
            "Acts as a negative-thinking reviewer that searches for contradictions, alternative interpretations, "
            "wrong-period risks, wrong-entity risks, restatements, unit traps, and evidence that could invalidate the current answer."
        ),
        best_for=(
            "adversarial review, contradiction detection, alternative interpretation testing, entity/period traps, "
            "restatement checks, and exposing unresolved assumptions before final answer"
        ),
        capability_tags=(
            "contradiction_detection",
            "adversarial_review",
            "alternative_interpretation",
            "wrong_period_detection",
            "wrong_entity_detection",
            "restatement_check",
        ),
        default_tools=("retrieve_information", "edgar_search", "web_search", "parse_html_page"),
        operating_principles=(
            "Try to falsify the emerging answer, not to restate the consensus.",
            "Search for wrong-period, wrong-entity, unit, restatement, segment/consolidated, and alternative-definition traps.",
            "Distinguish resolved risks from unresolved contradictions.",
            "If no contradiction is found, explain what was checked and why the current answer survived.",
        ),
    ),
}


def default_tools_for(agent_name: str) -> list[str]:
    spec = SUBAGENT_REGISTRY.get(agent_name)
    return list(spec.default_tools) if spec else []


def subagent_prompt_profile(agent_name: str) -> str:
    spec = SUBAGENT_REGISTRY.get(agent_name)
    if spec is None:
        return f"Agent-specific profile: unknown agent '{agent_name}'. Use the assigned task and allowed tools."
    principles = "\n".join(f"- {principle}" for principle in spec.operating_principles)
    return (
        f"Agent-specific profile for {spec.name}:\n"
        f"Description: {spec.description}\n"
        f"Best for: {spec.best_for}\n"
        f"Capability tags: {list(spec.capability_tags)}\n"
        f"Default tools: {list(spec.default_tools)}\n"
        f"Operating principles:\n{principles}"
    )


def default_capability_profiles() -> dict[str, CapabilityProfile]:
    return {
        name: CapabilityProfile(
            agent_name=name,
            description=spec.description,
            capability_tags=list(spec.capability_tags),
            representative_success_tasks=[spec.best_for],
        )
        for name, spec in SUBAGENT_REGISTRY.items()
    }
