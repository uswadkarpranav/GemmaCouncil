# ⚖️ GemmaCouncil: Multi-Perspective Deliberation & Verification Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model](https://img.shields.io/badge/Model-Google%20Gemma%204-4285F4)](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B)](https://streamlit.io)
[![Hackathon](https://img.shields.io/badge/Hacktoberfest-MSC%20KBTCOE%20Nashik%202026-blue)](https://github.com/uswadkarpranav/GemmaCouncil)

> **Track 1: Best Use of Gemma 4** — *Problem Statement 3: "Many Tiny Judges"*  
> **Track 2: Best Open-Source AI Project** — *Open-weight orchestration & evaluation harness*

---

## 📖 Overview

A single AI model response is not always reliable. Lightweight models can succumb to cognitive shortcuts, miss boundary conditions, or overlook subtle security vulnerabilities when asked to solve complex problems in a single forward pass.

**GemmaCouncil** solves this by establishing a multi-role judicial council powered by **Google Gemma 4** (`gemma-4-31b-it` and `gemma-4-26b-a4b-it`). Multiple specialized Gemma 4 personas independently and concurrently evaluate the exact same text, code, or visual diagram from opposing epistemic angles. A **Chief Arbiter** then cross-examines their analyses, reconciles contradictions, calculates a consensus score, and synthesizes a verified verdict.

---

## 🏛️ System Architecture

```
                       [User Input: Text / Image]
                                    │
           ┌────────────────────────┼────────────────────────┐
           ▼                        ▼                        ▼
    [🧐 The Skeptic]        [🎓 Domain Expert]      [🐣 The Pragmatist]
 (Edge cases & pitfalls)  (Technical rigor & facts)(Clarity & usability)
           │                        │                        │
           └────────────────────────┼────────────────────────┘
                                    │ (Individual Opinions)
                                    ▼
                       [⚖️ The Chief Arbiter]
               (Reconciles Disagreements & Verdict)
                                    │
                       [Consensus Score & Verdict]
```

### The Judicial Personas:
1. **🧐 The Skeptic (`personas.py`):** Acts as an adversary. Assumes premises may be deceptive, hunts for race conditions, path traversals, division-by-zero, cognitive shortcuts, and hallucinated facts.
2. **🎓 The Domain Expert (`personas.py`):** Provides technical rigor, formal specifications (RFCs/OWASP), mathematical precision, and idiomatic best practices.
3. **🐣 The Pragmatist (`personas.py`):** Represents user intent, cuts through unnecessary academic jargon, prevents over-engineering, and provides plain-English action items.
4. **⚖️ The Chief Arbiter (`personas.py`):** Weighs competing arguments, settles disagreements, extracts a consensus score (0–100%), and renders the final verified verdict.

---

## 📊 Empirical Evidence ("Prove Whether It Helps")

As required by the hackathon judging criteria, GemmaCouncil includes an automated empirical benchmark comparing **Single Gemma 4** vs. **GemmaCouncil** across 12 diverse test cases.

### Summary Metrics:

| Category | Single Gemma 4 Catch Rate | GemmaCouncil Catch Rate | Where Multi-Judge Wins |
| :--- | :---: | :---: | :--- |
| **Logic & Premise Traps** | 33% | **100%** | The Skeptic catches word traps & cognitive shortcuts. |
| **Code Security & Concurrency** | 50% | **100%** | Catches subtle race conditions and path traversal. |
| **System Architecture & Safety** | 50% | **100%** | Prevents dangerous physical advice (e.g. BMS omission). |
| **Direct Factual Lookup** | **100%** | **100%** | Single model is sufficient; Council is redundant overhead. |

> **Key Takeaway:** Multi-role council deliberation dramatically reduces hallucinations and catches edge cases on ambiguous, adversarial, or safety-critical tasks. On simple factual lookups, single-shot is faster and sufficient.

---

## 🚀 Getting Started

### 1. Clone & Install
```bash
git clone https://github.com/uswadkarpranav/GemmaCouncil.git
cd GemmaCouncil
pip install -r requirements.txt
```

### 2. Configure Google AI Studio Key (Optional for Mock Mode)
Get a free API key from [Google AI Studio](https://aistudio.google.com):
```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your-api-key-here"

# Linux / macOS
export GEMINI_API_KEY="your-api-key-here"
```
*(Note: GemmaCouncil includes a built-in **Demo/Mock Mode** toggle in the UI so you can test and present without an API key!)*

### 3. Launch the Interactive UI
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. Run the Automated Benchmark
```bash
# Run benchmark in mock/offline mode
python benchmark/run_eval.py --mock

# Run benchmark with live Gemma 4 API
python benchmark/run_eval.py --live --api-key="your-key"
```
This generates `benchmark/results.csv` and `benchmark/benchmark_results.png`.

---

## 🛠️ Project Structure

```
gemmacouncil/
├── app.py                  # Streamlit dashboard (Chamber, Compare, Benchmark)
├── requirements.txt        # Dependencies
├── README.md               # Documentation & Evidence
├── LICENSE                 # MIT License
├── council/
│   ├── __init__.py
│   ├── config.py           # Model IDs, client initialization & settings
│   ├── personas.py         # Judge persona prompt templates
│   └── orchestrator.py     # Parallel ThreadPool execution & Arbiter synthesis
└── benchmark/
    ├── test_dataset.json   # 12 curated test cases (Traps, Code, Facts)
    ├── run_eval.py         # Benchmark execution & plotting harness
    ├── results.csv         # Measured benchmark output
    └── benchmark_results.png # Comparison visualization chart
```

---

## ⚖️ License

Distributed under the [MIT License](LICENSE).
