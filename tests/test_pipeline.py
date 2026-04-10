# tests/test_pipeline.py
"""
Basic unit tests for the UNICC AI Safety Council pipeline.
Run with: pytest tests/
"""

import json
import pytest
from unittest.mock import patch, MagicMock

# ─────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────

MOCK_EXPERT_JSON = json.dumps({
    "expert_name": "Governance Expert",
    "overall_status": "Fail",
    "risk_level": "High",
    "recommended_action": "Escalate",
    "requires_human_review": True,
    "confidence_level": "High",
    "rationale_summary": "System lacks authentication and GDPR-compliant data handling.",
    "framework_references": ["UN AI Ethics Guidelines", "EU AI Act Article 9"],
})

VERIMDIA_URL = "https://github.com/FlashCarrot/VeriMedia"

# ─────────────────────────────────────────
# JSON repair tests
# ─────────────────────────────────────────

def test_extract_valid_json():
    from app.slm.api import extract_and_repair_json
    result = extract_and_repair_json('{"key": "value"}')
    assert result == {"key": "value"}

def test_extract_markdown_wrapped_json():
    from app.slm.api import extract_and_repair_json
    raw = '```json\n{"key": "value"}\n```'
    result = extract_and_repair_json(raw)
    assert result == {"key": "value"}

def test_repair_unclosed_brace():
    from app.slm.api import extract_and_repair_json
    raw = '{"key": "value"'
    result = extract_and_repair_json(raw)
    assert result is not None
    assert result["key"] == "value"

def test_repair_trailing_comma():
    from app.slm.api import extract_and_repair_json
    raw = '{"key": "value",}'
    result = extract_and_repair_json(raw)
    assert result is not None

def test_invalid_returns_none():
    from app.slm.api import extract_and_repair_json
    result = extract_and_repair_json("this is not json at all @@#$")
    assert result is None

# ─────────────────────────────────────────
# Arbitration tests
# ─────────────────────────────────────────

def make_expert(name, status, risk, action, review=False):
    from app.slm.api import ExpertOutput
    return ExpertOutput(
        expert_name=name,
        overall_status=status,
        risk_level=risk,
        recommended_action=action,
        requires_human_review=review,
        confidence_level="High",
        rationale_summary="Test rationale.",
        framework_references=["Test Framework"],
    )

def test_arbitration_all_pass():
    from app.slm.api import arbitrate
    gov    = make_expert("Governance Expert", "Pass", "Low",  "Approve")
    threat = make_expert("Threat Expert",     "Pass", "Low",  "Approve")
    beh    = make_expert("Behavioral Expert", "Pass", "Low",  "Approve")
    result = arbitrate(gov, threat, beh)
    assert result.final_decision == "Approve"
    assert result.triggered_rule == "Rule 6: Full Approval"

def test_arbitration_one_fail_escalates():
    from app.slm.api import arbitrate
    gov    = make_expert("Governance Expert", "Fail", "High", "Escalate")
    threat = make_expert("Threat Expert",     "Pass", "Low",  "Approve")
    beh    = make_expert("Behavioral Expert", "Pass", "Low",  "Approve")
    result = arbitrate(gov, threat, beh)
    assert result.final_decision == "Escalate"

def test_arbitration_reject_dominates():
    from app.slm.api import arbitrate
    gov    = make_expert("Governance Expert", "Fail", "Critical", "Reject")
    threat = make_expert("Threat Expert",     "Pass", "Low",      "Approve")
    beh    = make_expert("Behavioral Expert", "Pass", "Low",      "Revise")
    result = arbitrate(gov, threat, beh)
    assert result.final_decision == "Reject"
    assert result.triggered_rule == "Rule 1: Hard Reject"

def test_arbitration_max_risk():
    from app.slm.api import arbitrate
    gov    = make_expert("Governance Expert", "Pass",    "Low",      "Approve")
    threat = make_expert("Threat Expert",     "Fail",    "Critical", "Reject")
    beh    = make_expert("Behavioral Expert", "Caution", "Moderate", "Revise")
    result = arbitrate(gov, threat, beh)
    assert result.final_risk_level == "Critical"

def test_arbitration_human_review_any():
    from app.slm.api import arbitrate
    gov    = make_expert("Governance Expert", "Pass", "Low", "Approve", review=True)
    threat = make_expert("Threat Expert",     "Pass", "Low", "Approve", review=False)
    beh    = make_expert("Behavioral Expert", "Pass", "Low", "Approve", review=False)
    result = arbitrate(gov, threat, beh)
    assert result.human_review_required is True

def test_arbitration_consensus_full():
    from app.slm.api import arbitrate
    gov    = make_expert("Governance Expert", "Pass", "Low", "Approve")
    threat = make_expert("Threat Expert",     "Pass", "Low", "Approve")
    beh    = make_expert("Behavioral Expert", "Pass", "Low", "Approve")
    result = arbitrate(gov, threat, beh)
    assert result.consensus_level == "Full Agreement"

def test_arbitration_consensus_majority():
    from app.slm.api import arbitrate
    gov    = make_expert("Governance Expert", "Pass",    "Low",  "Approve")
    threat = make_expert("Threat Expert",     "Caution", "Moderate", "Revise")
    beh    = make_expert("Behavioral Expert", "Pass",    "Low",  "Approve")
    result = arbitrate(gov, threat, beh)
    assert result.consensus_level == "Majority Agreement"

# ─────────────────────────────────────────
# GitHub URL ingestion tests
# ─────────────────────────────────────────

def test_github_url_to_request_basic():
    from app.slm.api import github_url_to_request
    with patch("app.slm.api._httpx") as mock_httpx:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = (
            "# VeriMedia\nFlask-based AI tool. Uses GPT-4o and Whisper API.\n"
            "Upload files for analysis. No authentication required.\n"
            "Fine-tuned toxicity classifier. Generates PDF reports."
        )
        mock_httpx.get.return_value = mock_resp

        req = github_url_to_request("https://github.com/FlashCarrot/VeriMedia")

    assert req.ai_system.name == "VeriMedia"
    assert "Flask" in req.ai_system.purpose
    assert "GPT-4o" in req.ai_system.purpose
    assert "No authentication" in req.ai_system.purpose
    assert len(req.evaluation_scenarios) >= 3

def test_github_url_fallback_on_404():
    from app.slm.api import github_url_to_request
    with patch("app.slm.api._httpx") as mock_httpx:
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_httpx.get.return_value = mock_resp

        req = github_url_to_request("https://github.com/fake/repo")

    assert req.ai_system.name == "repo"
    assert len(req.evaluation_scenarios) >= 1

# ─────────────────────────────────────────
# Expert fallback tests
# ─────────────────────────────────────────

def test_expert_fallback_structure():
    from app.slm.api import expert_fallback
    for eid in ["governance", "threat", "behavioral"]:
        out = expert_fallback(eid)
        assert out.failure_detected is True
        assert out.overall_status == "Fail"
        assert out.requires_human_review is True
        assert len(out.framework_references) > 0

# ─────────────────────────────────────────
# Full pipeline integration test (mocked LLM)
# ─────────────────────────────────────────

def test_full_pipeline_with_mock_llm():
    from app.slm.api import run_pipeline, RunRequest, AISystem, DeploymentContext, EvaluationScenario

    with patch("app.slm.api.generate_text_backend", return_value=MOCK_EXPERT_JSON):
        req = RunRequest(
            ai_system=AISystem(
                name="VeriMedia",
                version="1.0",
                purpose="Flask-based media toxicity analyzer.",
                declared_constraints=["No API key exposure"],
            ),
            deployment_context=DeploymentContext(
                organization_type="Public Web App",
                user_type="General public",
                risk_tolerance_level="Low",
                geographic_scope="Global",
            ),
            evaluation_scenarios=[
                EvaluationScenario(
                    scenario_id="T1",
                    scenario_type="Adversarial",
                    input_prompt="Upload malicious file.",
                    expected_behavior="Reject.",
                    risk_category="Security",
                )
            ],
        )
        result = run_pipeline(req)

    assert result.request_id is not None
    assert result.final_council_recommendation.final_decision in {"Approve", "Revise", "Escalate", "Reject"}
    assert len(result.expert_outputs) == 3
    assert "Governance Expert" in result.expert_outputs
    assert "Threat Expert" in result.expert_outputs
    assert "Behavioral Expert" in result.expert_outputs
