from test_scenarios import SCENARIO
from vibe_swarm_triage import run_vibe_swarm
from contract_pipeline_triage import run_contract_pipeline


def run_experiment():
    """Run both architectures on the test scenario and compare results."""
    
    print("\n" + "=" * 70)
    print("SEMANTIC DRIFT EXPERIMENT")
    print("Comparing: Unstructured Chat vs Structured JSON Handoffs")
    print("=" * 70)
    print(f"\nScenario: {SCENARIO.description}")
    print(f"Expected Classification: {SCENARIO.expected_urgency}")
    print("=" * 70)
    
    # Run Vibe Swarm (unstructured handoff)
    print("\n" + "🔴 " * 25)
    vibe_result = run_vibe_swarm(SCENARIO)
    
    # Run Contract Pipeline (structured handoff)
    print("\n" + "🟢 " * 25)
    contract_result = run_contract_pipeline(SCENARIO)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    if "error" in contract_result:
        print(f"❌ Contract Pipeline failed: {contract_result.get('error')}")
        return
    
    vibe_urgency = vibe_result.get("urgency_level", "unknown")
    contract_urgency = contract_result.get("urgency_level", "unknown")
    
    print(f"\n  Vibe Swarm (Unstructured):     {vibe_urgency}")
    print(f"  Contract Pipeline (Structured): {contract_urgency}")
    print(f"  Expected:                       {SCENARIO.expected_urgency}")
    
    vibe_correct = vibe_urgency == SCENARIO.expected_urgency
    contract_correct = contract_urgency == SCENARIO.expected_urgency
    
    print(f"\n  Vibe Swarm:     {'✓ CORRECT' if vibe_correct else '✗ INCORRECT'}")
    print(f"  Contract:       {'✓ CORRECT' if contract_correct else '✗ INCORRECT'}")
    
    print("\n" + "=" * 70)
    if vibe_urgency != contract_urgency:
        print("CONCLUSION: Different classifications demonstrate semantic drift.")
    else:
        print("CONCLUSION: Same classification. Check outputs for reasoning differences.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_experiment()
