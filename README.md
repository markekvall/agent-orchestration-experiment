# Semantic Drift Experiment

An experiment demonstrating how unstructured handoffs between AI agents can cause semantic drift, leading to incorrect outcomes.

## The Problem

When agents pass free-form text to each other, speculation and reasoning get treated as facts by downstream agents. This experiment shows how a simple medical triage scenario produces different results depending on whether agents communicate via:

1. **Unstructured chat history** (the "Vibe Swarm")
2. **Structured JSON schemas** (the "Contract Pipeline")

## The Scenario

A patient presents with a moderate headache (4/10 severity), normal vitals, and a history of tension headaches. The patient mentions: *"I'm worried this might be something serious, maybe a migraine or worse."*

Based on clinical facts alone, this is a **routine** case.

## What Happens

**Vibe Swarm:** Agent 1 speculates about possible conditions. Agent 2 receives this full context and escalates to **urgent**, recommending a CT scan.

**Contract Pipeline:** Agent 1 outputs only structured facts. Agent 2 receives clean data and correctly classifies as **routine**.

## Setup

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Google Cloud

You need a GCP project with Vertex AI enabled.

```bash
# Authenticate with Google Cloud
gcloud auth application-default login
```

### 3. Set environment variables

Copy the example file and fill in your project ID:

```bash
cp .env.example .env
```

Edit `.env`:
```
GCP_PROJECT_ID=your-project-id
GCP_LOCATION=us-central1
MODEL_ID=gemini-2.0-flash
```

### 4. Run the experiment

```bash
source .env && python run_comparison.py
```

Or export the variables manually:
```bash
export GCP_PROJECT_ID=your-project-id
python run_comparison.py
```

## Files

- `run_comparison.py` - Main entry point, runs both architectures
- `vibe_swarm_triage.py` - Unstructured chat handoff
- `contract_pipeline_triage.py` - Structured JSON handoff
- `triage_schemas.py` - Pydantic schemas for structured handoffs
- `test_scenarios.py` - The test patient scenario

## Key Insight

The Contract Pipeline doesn't use a smarter model. It uses cleaner inputs. Both architectures use the same underlying model (Gemini 2.0 Flash). The difference is information hygiene: what crosses the boundary between agents.
