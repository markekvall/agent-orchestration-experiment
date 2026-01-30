from pydantic import BaseModel, Field
from typing import Literal


class SymptomAssessment(BaseModel):
    """Structured assessment from Agent 1 (Symptom Analyzer).
    
    This schema enforces that only factual data is passed to Agent 2,
    preventing speculative language from contaminating the triage decision.
    """
    primary_symptoms: list[str] = Field(
        description="List of observed symptoms, no speculation or possible conditions"
    )
    severity_score: int = Field(
        ge=1, 
        le=10,
        description="Severity rating from 1 (mild) to 10 (critical)"
    )
    vital_signs_abnormal: bool = Field(
        description="Whether any vital signs are outside normal range"
    )
    pain_level: Literal["none", "mild", "moderate", "severe"] = Field(
        description="Current pain level"
    )
    time_since_onset: str = Field(
        description="Time since symptoms began (e.g., '2 hours', '3 days')"
    )
    relevant_history: list[str] = Field(
        description="Only factual medical history relevant to current symptoms. No speculation, no 'possible' conditions."
    )


class TriageDecision(BaseModel):
    """Structured decision from Agent 2 (Triage Classifier).
    
    This schema ensures the final decision is clear and actionable.
    """
    urgency_level: Literal["immediate", "urgent", "routine", "non_urgent"] = Field(
        description="Triage urgency classification"
    )
    reasoning: str = Field(
        description="Brief explanation of the classification decision"
    )
    recommended_action: str = Field(
        description="Specific recommended next step (e.g., 'Send to ER immediately', 'Schedule within 24 hours')"
    )
