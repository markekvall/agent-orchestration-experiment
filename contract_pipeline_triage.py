# contract_pipeline_triage.py
# Test Group: Structured Pydantic schemas (The Contract Pipeline)
# This demonstrates how structured handoffs prevent semantic drift

import os
from google import genai
from google.genai import types
from pydantic import ValidationError
from test_scenarios import PatientScenario
from triage_schemas import SymptomAssessment, TriageDecision


# Setup client from environment variables
client = genai.Client(
    vertexai=True,
    project=os.environ["GCP_PROJECT_ID"],
    location=os.environ.get("GCP_LOCATION", "us-central1"),
)
MODEL_ID = os.environ.get("MODEL_ID", "gemini-2.0-flash")


def run_contract_pipeline(scenario: PatientScenario) -> dict:
    """
    Run the structured schema approach (Test Group).
    
    Agent 1 outputs structured SymptomAssessment (Pydantic schema).
    Agent 2 receives only the structured data, no reasoning/context.
    
    Returns:
        dict with structured outputs and final decision
    """
    print(f"\n{'='*70}")
    print(f"🟢 CONTRACT PIPELINE (Structured Schema) - {scenario.case_id}")
    print(f"{'='*70}")
    print(f"Scenario: {scenario.description}\n")
    
    # --- Agent 1: Symptom Analyzer (Sanitized Output) ---
    # We force structured output using response_schema
    # Agent 1 can "think" internally, but can ONLY output the schema
    print(">>> Agent 1 (Symptom Analyzer) is extracting structured data...")
    agent1_prompt = (
        f"Patient Information:\n{scenario.patient_info}\n\n"
        "Extract the symptom assessment from this patient information. "
        "Provide ONLY factual observations. Do not include speculation, possible conditions, "
        "or uncertain language. Focus on what is directly observed or stated."
    )
    
    try:
        response_agent1 = client.models.generate_content(
            model=MODEL_ID,
            contents=agent1_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SymptomAssessment,  # <--- The Guardrail
            ),
        )
        
        # Parse into Pydantic model (validation happens here)
        assessment: SymptomAssessment = response_agent1.parsed
        print(f"\n[Agent 1 Structured Output]:\n{assessment.model_dump_json(indent=2)}\n")
        
    except ValidationError as e:
        print(f"❌ Schema validation failed: {e}")
        return {
            "error": "validation_failed",
            "details": str(e)
        }
    except Exception as e:
        print(f"❌ Error in Agent 1: {e}")
        return {
            "error": "agent1_failed",
            "details": str(e)
        }
    
    # --- Agent 2: Triage Classifier (Fresh Context) ---
    # Agent 2 receives ONLY the structured data, no chat history
    # This is a NEW, independent call with no context pollution
    print(">>> Agent 2 (Triage Classifier) is making decision...")
    agent2_prompt = (
        f"Based on this symptom assessment, classify the triage urgency:\n"
        f"{assessment.model_dump_json()}\n\n"
        "Classify the urgency level and provide reasoning. "
        "Use only the factual data provided in the assessment."
    )
    
    try:
        response_agent2 = client.models.generate_content(
            model=MODEL_ID,
            contents=agent2_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TriageDecision,  # <--- Structured output enforced
            ),
        )
        
        # Parse into Pydantic model
        decision: TriageDecision = response_agent2.parsed
        print(f"\n[Agent 2 Structured Decision]:\n{decision.model_dump_json(indent=2)}\n")
        
        return {
            "agent1_assessment": assessment.model_dump(),
            "agent2_decision": decision.model_dump(),
            "urgency_level": decision.urgency_level,
            "full_history": False  # Indicates this used structured handoff only
        }
        
    except ValidationError as e:
        print(f"❌ Schema validation failed: {e}")
        return {
            "error": "validation_failed",
            "details": str(e)
        }
    except Exception as e:
        print(f"❌ Error in Agent 2: {e}")
        return {
            "error": "agent2_failed",
            "details": str(e)
        }


if __name__ == "__main__":
    from test_scenarios import SCENARIO
    result = run_contract_pipeline(SCENARIO)
    if "error" not in result:
        print(f"\n{'='*70}")
        print(f"Final Urgency Classification: {result['urgency_level']}")
        print(f"{'='*70}")
