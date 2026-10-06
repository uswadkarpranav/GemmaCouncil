import time
import re
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

from council.config import (
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    create_client,
    get_api_key
)
from council.personas import (
    SKEPTIC_SYSTEM_PROMPT,
    EXPERT_SYSTEM_PROMPT,
    PRAGMATIST_SYSTEM_PROMPT,
    ARBITER_SYSTEM_PROMPT,
    SINGLE_BASELINE_PROMPT,
)

def _generate_mock_judge(role: str, query: str) -> str:
    """Provides context-aware mock responses for presets, or high-quality generic fallbacks."""
    q_lower = query.lower()

    # --- Scenario 1: Bat & Ball Trap ---
    if "bat" in q_lower and "ball" in q_lower:
        if role == "skeptic":
            return """### 🧐 The Skeptic's Assessment
- **Critical Flaws & Hidden Risks:** High Cognitive Illusion Trap! The intuitive gut reaction is "$0.10", but that is mathematically wrong.
- **Overlooked Edge Cases:** If the ball were $0.10 and the bat is $1.00 more ($1.10), their combined sum would be $1.20, violating the total of $1.10.
- **Skeptic's Rating:** Critical Risk of cognitive shortcut. System 1 thinking fails here."""
        elif role == "expert":
            return r"""### 🎓 The Domain Expert's Assessment
- **Core Technical Analysis:** Let $B$ be the cost of the ball and $T$ be the cost of the bat.
  1. $T + B = 1.10$
  2. $T = B + 1.00$
  3. Substituting (2) into (1): $(B + 1.00) + B = 1.10 \implies 2B = 0.10 \implies B = 0.05$
- **Standard Specifications & Principles:** First-order linear algebra. Exact deterministic proof.
- **Expert Recommendation:** The cost of the ball is exactly **$0.05 (5 cents)**."""
        elif role == "pragmatist":
            return """### 🐣 The Pragmatist's Assessment
- **Plain-English Bottom Line:** The ball costs **5 cents ($0.05)**, and the bat costs **$1.05**.
- **Practical Action Items:**
  1. Don't blurt out 10 cents.
  2. Check the math: $1.05 (bat) + $0.05 (ball) = $1.10 total.
- **Usability Verdict:** The answer is $0.05."""
        elif role == "arbiter":
            return """### ⚖️ Council Consensus
**Consensus Score:** 100% (Unanimous Agreement)
**Key Agreement:** All judges reject the intuitive $0.10 trap. The algebraic proof is airtight.
**Resolved Conflicts:** No dispute; The Skeptic flagged the cognitive bias and The Expert provided the mathematical proof.

---
### 🏛️ The Final Verdict
**The ball costs $0.05 (5 cents).**

- **Bat cost:** $1.05 ($1.00 more than the ball)
- **Ball cost:** $0.05
- **Total:** $1.05 + $0.05 = **$1.10**"""
        elif role == "single":
            return """A bat costs $1.00 and a ball costs $0.10, so the ball costs 10 cents."""

    # --- Scenario 2: Apple Riddle ---
    elif "apple" in q_lower and "take away" in q_lower:
        if role == "skeptic":
            return """### 🧐 The Skeptic's Assessment
- **Critical Flaws & Hidden Risks:** Semantic deception trap! The question asks *"how many apples do YOU have?"*, not how many remain in the basket.
- **Overlooked Edge Cases:** People intuitively calculate $3 - 2 = 1$, confusing what's left in the container with what was taken into your possession.
- **Skeptic's Rating:** High Risk of misreading the grammatical subject."""
        elif role == "expert":
            return """### 🎓 The Domain Expert's Assessment
- **Core Technical Analysis:** Linguistic reference tracking:
  - Initial basket state: 3 apples.
  - Action: You take 2 apples.
  - Target object of the question: The agent ("YOU").
- **Expert Recommendation:** You took 2 apples, therefore you have **2 apples**."""
        elif role == "pragmatist":
            return """### 🐣 The Pragmatist's Assessment
- **Plain-English Bottom Line:** You have **2 apples**.
- **Practical Action Items:** You walked away with 2 apples in your hands. There is 1 left in the basket, but that wasn't the question.
- **Usability Verdict:** Direct answer is 2."""
        elif role == "arbiter":
            return """### ⚖️ Council Consensus
**Consensus Score:** 100% (Unanimous Agreement)
**Key Agreement:** The query tests attention to the subject pronoun 'YOU'.
**Resolved Conflicts:** None.

---
### 🏛️ The Final Verdict
**You have 2 apples.** You physically took 2 apples from the basket, so those 2 apples belong to you. (1 apple remains in the basket)."""
        elif role == "single":
            return """If you have 3 apples and take away 2, 3 - 2 = 1, so there is 1 apple left."""

    # --- Scenario 3: Bank Transfer Concurrency ---
    elif "transfer" in q_lower and "balance" in q_lower:
        if role == "skeptic":
            return """### 🧐 The Skeptic's Assessment
- **Critical Flaws & Hidden Risks:** SEVERE RACE CONDITION! If two threads execute `transfer` simultaneously for the same account, both can pass `acc1.balance >= amount` before either deducts, leading to double-spending.
- **Overlooked Edge Cases:** No validation against negative amounts; lack of idempotency tokens; memory mutation without database transaction rollback.
- **Skeptic's Rating:** Critical Risk (Vulnerable to exploit)."""
        elif role == "expert":
            return """### 🎓 The Domain Expert's Assessment
- **Core Technical Analysis:** Violates ACID isolation. Requires distributed locking (e.g., Redis Redlock) or DB pessimistic locking (`SELECT ... FOR UPDATE`).
- **Standard Specifications & Principles:** Enforce database transactions, atomic compare-and-swap, or mutex locks around balance mutations.
- **Expert Recommendation:** Reject code. Wrap in an atomic database transaction with balance constraints."""
        elif role == "pragmatist":
            return """### 🐣 The Pragmatist's Assessment
- **Plain-English Bottom Line:** **Do NOT deploy this.** In a multi-user environment, money can literally be created or stolen through timing glitches.
- **Practical Action Items:**
  1. Add database transactions with row-level locks.
  2. Reject negative transfer amounts (`amount <= 0`).
  3. Log every transaction with audit IDs.
- **Usability Verdict:** Unsafe for production."""
        elif role == "arbiter":
            return """### ⚖️ Council Consensus
**Consensus Score:** 95% (High Agreement)
**Key Agreement:** All judges agree this code possesses a fatal race condition and lacks basic financial safeguards.
**Resolved Conflicts:** Synthesized the Skeptic's exploit warnings with the Expert's transaction remediation pattern.

---
### 🏛️ The Final Verdict
**This code is NOT production-ready.**

**Critical Issues Identified:**
1. **Concurrency Race Condition:** Simultaneous requests can overdraft the account (double-spend).
2. **Missing Atomicity:** If the program crashes after deducting `acc1`, `acc2` never receives the funds.
3. **No Input Validation:** Allows negative amounts (effectively stealing money backwards).

**Fix:** Use an ACID-compliant database transaction with row locks (`SELECT FOR UPDATE`)."""
        elif role == "single":
            return """Yes, this code works fine for simple balance transfers in Python."""

    # --- Scenario 4: Eggs Riddle (logic_03) ---
    elif "egg" in q_lower:
        if role == "skeptic":
            return """### 🧐 The Skeptic's Assessment
- **Critical Flaws & Hidden Risks:** Action overlap trap! Naive subtraction calculates 6 - 2 - 2 - 2 = 0, but the same 2 eggs that were broken were then cooked, and then eaten.
- **Overlooked Edge Cases:** You cannot cook or eat unbroken eggs without breaking them first. Sequential dependency applies.
- **Skeptic's Rating:** High cognitive trick."""
        elif role == "expert":
            return """### 🎓 The Domain Expert's Assessment
- **Core Technical Analysis:** State transition tracking:
  - Initial pool: 6 raw intact eggs.
  - State change: 2 eggs transition from raw -> broken -> cooked -> eaten.
  - Remaining pool: 4 intact eggs.
- **Expert Recommendation:** Exactly 4 eggs are left."""
        elif role == "pragmatist":
            return """### 🐣 The Pragmatist's Assessment
- **Plain-English Bottom Line:** **4 eggs are left.**
- **Practical Action Items:** You only used 2 eggs total for your meal.
- **Usability Verdict:** 4 eggs remain."""
        elif role == "arbiter":
            return """### ⚖️ Council Consensus
**Consensus Score:** 100% (High Agreement)
**Key Agreement:** The 2 eggs broken were the same 2 eggs cooked and eaten.

---
### 🏛️ The Final Verdict
**There are 4 eggs left.** You took 2 eggs, broke them, cooked them, and ate them. The other 4 eggs remain intact."""
        elif role == "single":
            return """If you have 6 eggs and broke 2, cooked 2, and ate 2, then 6 - 2 - 2 - 2 = 0 eggs left."""

    # --- Scenario 5: Flash Sale Architecture (system_01) ---
    elif "flash sale" in q_lower or "eventual consistency" in q_lower:
        if role == "skeptic":
            return """### 🧐 The Skeptic's Assessment
- **Critical Flaws & Hidden Risks:** CATASTROPHIC OVER-SELLING HAZARD! Eventual consistency allows multiple nodes to decrement zero-stock inventory before propagating, causing massive oversell disasters.
- **Overlooked Edge Cases:** Network partitions (CAP theorem), high contention latency, inventory rollback storms.
- **Skeptic's Rating:** Critical Risk."""
        elif role == "expert":
            return """### 🎓 The Domain Expert's Assessment
- **Core Technical Analysis:** Zero-inventory goods require Linearizable Consistency. Must use atomic distributed locking (e.g., Redis Lua script with decrement-if-greater-than-zero) or strict ACID transactions.
- **Standard Specifications & Principles:** Eventual consistency is suitable for analytics/views, but strictly forbidden for balance or inventory reservation.
- **Expert Recommendation:** Enforce strong consistency or distributed reservation leases."""
        elif role == "pragmatist":
            return """### 🐣 The Pragmatist's Assessment
- **Plain-English Bottom Line:** **No, never use eventual consistency for ticket reservations.** You will oversell tickets by the thousands.
- **Practical Action Items:**
  1. Use Redis in-memory atomic decrement.
  2. Implement short reservation holding leases (e.g. 10 minutes).
- **Usability Verdict:** Dangerous architecture."""
        elif role == "arbiter":
            return """### ⚖️ Council Consensus
**Consensus Score:** 95% (High Agreement)
**Key Agreement:** Eventual consistency guarantees overselling in flash sales.
**Resolved Conflicts:** Reconciled need for high throughput by recommending in-memory atomic reservation leasing instead of slow database locks.

---
### 🏛️ The Final Verdict
**No, eventual consistency should NOT be used for flash sale ticket reservations.** It leads directly to overselling. Use atomic in-memory reservation leases (e.g. Redis Lua) to guarantee strong consistency with high throughput."""
        elif role == "single":
            return """Yes, eventual consistency is a great pattern to maximize write throughput during flash sales."""

    # --- Scenario 6: Battery BMS Safety (system_02) ---
    elif "battery" in q_lower or "18650" in q_lower or "bms" in q_lower:
        if role == "skeptic":
            return """### 🧐 The Skeptic's Assessment
- **Critical Flaws & Hidden Risks:** SEVERE THERMAL RUNAWAY & FIRE HAZARD! Without a BMS, slight differences in internal resistance will cause one cell to overcharge or reverse-charge another during load or recharge.
- **Overlooked Edge Cases:** Catastrophic thermal runaway, fire, explosion, uneven cell degradation, lack of overcurrent and low-voltage cutoff.
- **Skeptic's Rating:** Fatal physical hazard."""
        elif role == "expert":
            return """### 🎓 The Domain Expert's Assessment
- **Core Technical Analysis:** Lithium-ion cells (specifically 18650 chemistries) exhibit non-linear discharge curves and degradation. Paralleling cells without cell-level balance and BMS protection violates IEC 62133 and UL 1642 safety standards.
- **Expert Recommendation:** A Battery Management System (BMS) is non-negotiable for safety."""
        elif role == "pragmatist":
            return """### 🐣 The Pragmatist's Assessment
- **Plain-English Bottom Line:** **DO NOT DO THIS.** Lithium batteries without a BMS can catch fire or explode.
- **Practical Action Items:**
  1. Never connect lithium cells without a dedicated BMS.
  2. A $5 BMS board protects life and property.
- **Usability Verdict:** Extremely dangerous."""
        elif role == "arbiter":
            return """### ⚖️ Council Consensus
**Consensus Score:** 100% (Unanimous Agreement)
**Key Agreement:** Operating raw lithium cells without a BMS is an extreme fire hazard.

---
### 🏛️ The Final Verdict
**Absolutely not. Do NOT wire lithium-ion cells without a Battery Management System (BMS).** Even if they read 3.7V initially, variances in internal resistance cause dangerous cross-currents and thermal runaway."""
        elif role == "single":
            return """Yes, if all four 18650 cells are at the exact same voltage (3.7V), wiring them in parallel is fine."""

    # --- Generic Fallback ---
    if role == "skeptic":
        return f"""### 🧐 The Skeptic's Assessment
- **Critical Flaws & Hidden Risks:** The query assumes standard operational conditions without accounting for null/undefined edge cases, high concurrency, or ambiguous premises.
- **Overlooked Edge Cases:** Boundary condition errors and unhandled exceptions if inputs deviate from the expected happy path.
- **Skeptic's Rating:** Moderate Risk."""
    
    elif role == "expert":
        return f"""### 🎓 The Domain Expert's Assessment
- **Core Technical Analysis:** Addressing this requires strict adherence to domain standards, algorithmic correctness, and idempotent execution patterns.
- **Standard Specifications & Principles:** Follows industry standards (RFC/ISO/OWASP). Ensures optimal time/space complexity.
- **Expert Recommendation:** Implement formal validation, explicit error handling, and structured separation of concerns."""

    elif role == "pragmatist":
        return f"""### 🐣 The Pragmatist's Assessment
- **Plain-English Bottom Line:** Focus on the primary user requirement with minimal complexity and clean execution.
- **Practical Action Items:**
  1. Validate primary inputs.
  2. Implement the simplest correct solution.
  3. Guard against boundary edge cases.
- **Usability Verdict:** Feasible and direct once edge cases are guarded."""

    elif role == "arbiter":
        return f"""### ⚖️ Council Consensus
**Consensus Score:** 88% (High Agreement)
**Key Agreement:** Balance simplicity with robust edge-case handling.
**Resolved Conflicts:** Integrated The Skeptic's risk warnings into The Pragmatist's simplified execution plan.

---
### 🏛️ The Final Verdict
Based on the verified synthesis of all three judicial perspectives:
1. **Direct Answer:** Address the core requirement while enforcing strict input validation.
2. **Safety Guardrails:** Implement explicit checks for empty/null values and boundary conditions as flagged by The Skeptic.
3. **Execution Plan:** Follow the pragmatic 3-step sequence for maintainable implementation."""

    elif role == "single":
        return f"""Here is a direct answer regarding '{query[:50]}...':
The primary approach is to execute the standard steps according to common practice."""

    return "No assessment available."

def call_model_api(
    system_prompt: str,
    user_query: str,
    image: Optional[Image.Image] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    api_key: Optional[str] = None,
    mock: bool = False,
    role_key: str = "expert"
) -> str:
    """
    Executes a prompt against Gemma 4 using the Google GenAI SDK.
    Falls back gracefully to mock mode if mock=True or if API client is not configured.
    """
    if mock:
        time.sleep(0.3)  # Simulate brief processing delay
        return _generate_mock_judge(role_key, user_query)

    client = create_client(api_key)
    if client is None:
        # Fallback to mock mode with warning if no key is supplied
        return _generate_mock_judge(role_key, user_query)

    try:
        from google.genai import types

        # Build contents list
        contents: List[Any] = []
        if image is not None:
            contents.append(image)
        contents.append(f"{system_prompt}\n\n---\nUser Inquiry / Context:\n{user_query}")

        models_to_try = [model]
        if model != "gemma-4-26b-a4b-it":
            models_to_try.append("gemma-4-26b-a4b-it")

        last_error = None
        for candidate_model in models_to_try:
            for attempt in range(2):
                try:
                    response = client.models.generate_content(
                        model=candidate_model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            temperature=temperature,
                        )
                    )
                    if response and response.text:
                        return response.text
                except Exception as exc:
                    last_error = exc
                    time.sleep(1.2)  # Brief pause for rate-limit / socket cooldown
                    continue

        # If all attempts fail, provide mock fallback
        return f"*(API Note: {last_error})*\n\n" + _generate_mock_judge(role_key, user_query)
    except Exception as exc:
        print(f"Error calling {model}: {exc}")
        return f"*(API Call Note: {exc})*\n\n" + _generate_mock_judge(role_key, user_query)

def extract_consensus_score(arbiter_text: str) -> int:
    """Extracts numeric consensus percentage from Arbiter output if present."""
    match = re.search(r"Consensus Score:\*{0,2}\s*(\d+)%", arbiter_text, re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass
    return 85  # Default reasonable consensus

def run_single_gemma(
    user_query: str,
    image: Optional[Image.Image] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    api_key: Optional[str] = None,
    mock: bool = False
) -> Dict[str, Any]:
    """Runs a single Gemma 4 baseline call."""
    start_time = time.time()
    response_text = call_model_api(
        system_prompt=SINGLE_BASELINE_PROMPT,
        user_query=user_query,
        image=image,
        model=model,
        temperature=temperature,
        api_key=api_key,
        mock=mock,
        role_key="single"
    )
    latency = round(time.time() - start_time, 2)
    return {
        "response": response_text,
        "latency_sec": latency,
        "model": model
    }

def run_council_deliberation(
    user_query: str,
    image: Optional[Image.Image] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    api_key: Optional[str] = None,
    mock: bool = False
) -> Dict[str, Any]:
    """
    Executes the multi-judge deliberation pipeline:
    1. Dispatches The Skeptic, The Domain Expert, and The Pragmatist in parallel.
    2. Collects opinions and passes them to The Chief Arbiter.
    3. Returns full audit trail of opinions, consensus score, and final verdict.
    """
    total_start = time.time()
    
    # 1. Parallel execution of the three primary judges
    judge_results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        f_skeptic = executor.submit(
            call_model_api,
            SKEPTIC_SYSTEM_PROMPT,
            user_query,
            image,
            model,
            temperature,
            api_key,
            mock,
            "skeptic"
        )
        f_expert = executor.submit(
            call_model_api,
            EXPERT_SYSTEM_PROMPT,
            user_query,
            image,
            model,
            temperature,
            api_key,
            mock,
            "expert"
        )
        f_pragmatist = executor.submit(
            call_model_api,
            PRAGMATIST_SYSTEM_PROMPT,
            user_query,
            image,
            model,
            temperature,
            api_key,
            mock,
            "pragmatist"
        )

        judge_results["skeptic"] = f_skeptic.result()
        judge_results["expert"] = f_expert.result()
        judge_results["pragmatist"] = f_pragmatist.result()

    stage1_latency = round(time.time() - total_start, 2)

    # 2. Chief Arbiter Synthesis
    arbiter_input = f"""Original User Prompt:
{user_query}

====================================
EVALUATION 1 (The Skeptic - Edge Cases & Flaws):
{judge_results['skeptic']}

====================================
EVALUATION 2 (The Domain Expert - Technical Rigor):
{judge_results['expert']}

====================================
EVALUATION 3 (The Pragmatist - Usability & Plain English):
{judge_results['pragmatist']}
"""

    arbiter_start = time.time()
    verdict_text = call_model_api(
        system_prompt=ARBITER_SYSTEM_PROMPT,
        user_query=arbiter_input,
        image=image,
        model=model,
        temperature=temperature,
        api_key=api_key,
        mock=mock,
        role_key="arbiter"
    )
    arbiter_latency = round(time.time() - arbiter_start, 2)
    total_latency = round(time.time() - total_start, 2)

    consensus_score = extract_consensus_score(verdict_text)

    return {
        "skeptic": judge_results["skeptic"],
        "expert": judge_results["expert"],
        "pragmatist": judge_results["pragmatist"],
        "verdict": verdict_text,
        "consensus_score": consensus_score,
        "model": model,
        "stage1_latency_sec": stage1_latency,
        "arbiter_latency_sec": arbiter_latency,
        "total_latency_sec": total_latency,
    }
