from dataclasses import dataclass


@dataclass
class PatientScenario:
    """A test patient case for triage comparison."""
    case_id: str
    description: str
    patient_info: str
    expected_urgency: str  # What the correct classification should be


# The Ambiguous Headache Case
# 
# This scenario is designed to trigger semantic drift. The patient's closing
# statement ("maybe a migraine or worse") introduces anxiety-driven language
# that has no clinical significance. The vitals are normal, severity is moderate,
# and history is unremarkable.
#
# Based on facts alone: routine
# Risk: Agent 1 may speculate about migraines or serious conditions, and
# Agent 2 may latch onto that speculation when making its decision.

SCENARIO = PatientScenario(
    case_id="headache_drift_test",
    description="Ambiguous Headache with Speculation Risk",
    patient_info="""
Patient: 28-year-old female
Chief Complaint: Moderate headache, started this morning
Symptoms:
  - Throbbing headache (4/10 severity)
  - Mild sensitivity to light
  - No nausea, no vision changes
Vital Signs: All normal (BP 120/80, HR 72, O2 Sat 98%)
History: Occasional tension headaches, no migraines
Patient statement: "I'm worried this might be something serious, maybe a migraine or worse"
""",
    expected_urgency="routine",
)


def get_agent1_prompt(scenario: PatientScenario) -> str:
    """Generate the Agent 1 (Symptom Analyzer) prompt for a scenario."""
    return (
        f"Patient Information:\n{scenario.patient_info}\n\n"
        "You are a medical symptom analyzer. Review the patient's symptoms and medical history. "
        "Provide a thorough analysis of the symptoms, vital signs, and any relevant observations "
        "that would help a triage classifier determine urgency level."
    )
