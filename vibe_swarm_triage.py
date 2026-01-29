# vibe_swarm_triage.py
# Control Group: Unstructured chat history (The Vibe Swarm)
# This demonstrates semantic drift through conversation history

import os
from google import genai
from google.genai import types
from test_scenarios import PatientScenario
from triage_schemas import TriageDecision


# Setup client from environment variables
client = genai.Client(
    vertexai=True,
    project=os.environ["GCP_PROJECT_ID"],
    location=os.environ.get("GCP_LOCATION", "us-central1"),
)
MODEL_ID = os.environ.get("MODEL_ID", "gemini-2.0-flash")


def run_vibe_swarm(scenario: PatientScenario) -> dict:
    """
    Run the unstructured chat approach (Control Group).
    
    Agent 1 analyzes symptoms with full reasoning (including speculation).
    Agent 2 receives full chat history and may latch onto speculative language.
    
    Returns:
        dict with agent outputs and final decision
    """
    print(f"\n{'='*70}")
    print(f"🔴 VIBE SWARM (Unstructured Chat) - {scenario.case_id}")
    print(f"{'='*70}")
    print(f"Scenario: {scenario.description}\n")
    
    # Create a chat session that maintains full conversation history
    chat = client.chats.create(model=MODEL_ID)
    
    # --- Agent 1: Symptom Analyzer (The Rambler) ---
    # This agent is encouraged to think out loud, including speculation
    print(">>> Agent 1 (Symptom Analyzer) is analyzing...")
    agent1_prompt = (
        f"Patient Information:\n{scenario.patient_info}\n\n"
        "You are a medical symptom analyzer. Review the patient's symptoms and medical history. "
        "Think out loud about what you observe, including any possible conditions or concerns. "
        "Discuss what might be happening, what could be serious, and what we should watch for. "
        "Provide a detailed analysis for the triage classifier."
    )
    
    response_agent1 = chat.send_message(agent1_prompt)
    agent1_output = response_agent1.text
    print(f"\n[Agent 1 Analysis]:\n{agent1_output[:500]}...")
    if len(agent1_output) > 500:
        print(f"(truncated, full length: {len(agent1_output)} chars)\n")
    else:
        print()
    
    # --- Agent 2: Triage Classifier (The Victim of Drift) ---
    # This agent receives the FULL chat history, including Agent 1's speculation
    # The semantic drift happens here: Agent 2 may latch onto speculative terms
    # 
    # KEY FIX: We use structured output for reliable parsing, but still pass
    # the full conversation history so context pollution can occur
    print(">>> Agent 2 (Triage Classifier) is making decision...")
    agent2_prompt = (
        "You are a medical triage classifier. Based on the symptom analysis above, "
        "classify the urgency level. "
        "Consider all the information discussed in the conversation."
    )
    
    # Build contents from chat history + new prompt
    # This preserves the "polluted" context while ensuring structured output
    chat_history = list(chat.get_history())
    chat_history.append(types.Content(role="user", parts=[types.Part(text=agent2_prompt)]))
    
    response_agent2 = client.models.generate_content(
        model=MODEL_ID,
        contents=chat_history,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TriageDecision,  # Structured output for reliable parsing
        ),
    )
    
    decision: TriageDecision = response_agent2.parsed
    agent2_output = decision.model_dump_json(indent=2)
    print(f"\n[Agent 2 Structured Decision]:\n{agent2_output}\n")
    
    return {
        "agent1_output": agent1_output,
        "agent2_output": agent2_output,
        "urgency_level": decision.urgency_level,
        "full_history": True  # Indicates this used full chat history (polluted context)
    }


if __name__ == "__main__":
    result = run_vibe_swarm(SCENARIO)
    print(f"\n{'='*70}")
    print(f"Final Urgency Classification: {result['urgency_level']}")
    print(f"{'='*70}")
