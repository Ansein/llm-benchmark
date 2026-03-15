# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **LLM Privacy Externality Benchmark Framework** — an academic research system evaluating large language models' ability to understand *economic mechanisms* (not just outcomes) in data privacy externality games. Built for KDD/TKDE paper submission.

**Core hypothesis:** LLMs should understand not just "what is the final sharing rate?" but "why does the sharing rate change when correlation increases?"

## Setup

```bash
pip install openai httpx numpy pandas scipy matplotlib seaborn
```

**Windows encoding fix (required before running any script):**
```bash
export PYTHONIOENCODING=utf-8
```

Copy `configs/model_configs.example.json` to `configs/model_configs.json` and fill in API keys. The live config file is gitignored. Do not commit real API keys.

## Common Commands

```bash
# Run all scenarios, all configured models
python run_evaluation.py --scenarios A B C

# Run a single model on a single scenario
python run_evaluation.py --single --scenarios A --models deepseek-v3.2

# Print summary of existing results without re-running
python run_evaluation.py --summary-only

# Scenario B: prompt version experiments (6 versions)
python run_prompt_experiments.py --versions b.v1 b.v2 b.v3 b.v4 b.v5 b.v6 --model deepseek-v3.2

# Scenario B: 3x3 parameter sensitivity analysis
python run_sensitivity_b.py --models deepseek-v3.2 gpt-5.1 --num-trials 3

# Scenario C: iterative mode
python -m src.evaluators.evaluate_scenario_c --mode iterative --model deepseek-v3.2

# Scenario C: fictitious play mode
python -m src.evaluators.evaluate_scenario_c --mode fp --fp_config all --model deepseek-v3.2

# Generate ground truth data
python -m src.scenarios.generate_scenario_c_gt
python scripts/generate_sensitivity_b_gt.py

# Visualization
python visualize_scenario_c_results.py
python visualize_scenario_c_advanced.py
python compare_gpt_models_sensitivity.py
python compare_models_sensitivity.py
python plot_prompt_comparison_academic.py
python plot_quadrant_analysis.py
```

## Architecture

### 4-Step Pipeline

```
Step 1: Theoretical Solvers  →  Ground Truth (data/ground_truth/)
Step 2: LLM-Embedded Game Play  →  Evaluation Results (evaluation_results/)
Step 3: Diagnostic Metrics (MAE, Jaccard, convergence, labels)
Step 4: Robustness (sensitivity_results/ + fictitious play)
```

### Three Evaluation Scenarios

| Scenario | Economic Paper | Mechanism | LLM Role |
|---|---|---|---|
| A | Rhodes & Zhou (2019) | Price transmission externality | Consumer deciding data disclosure (up to 10 iterative rounds) |
| B | Acemoglu et al. (2022) "Too Much Data" | Inference externality (correlated types) | User deciding data sharing; 6 prompt versions × 12 models |
| C | Bergemann & Bonatti (2022) "Economics of Social Data" | Social data free-riding + anonymization | Intermediary and/or consumer in Stackelberg game; 4 configurations |

### Key Source Files

- **`src/evaluators/llm_client.py`** — Central LLM interface. Wraps `openai.OpenAI` with retry/timeout logic, call logging, and JSON response parsing. All models accessed via OpenAI-compatible APIs.
- **`src/scenarios/scenario_b_too_much_data.py`** — Scenario B solver: Bayesian posterior covariance (Kalman-like), set enumeration for disclosure equilibria, computes Jaccard/Hamming/F1 metrics.
- **`src/scenarios/scenario_c_social_data.py`** — Scenario C solver: fixed-point iteration for equilibrium participation rate under two data structures (Common Preferences / Common Experience).
- **`src/scenarios/scenario_c_social_data_optimization.py`** — Mixed optimizer: grid search over `m∈[0,3]` + `scipy.optimize.minimize` L-BFGS-B for N-dimensional compensation vector.
- **`run_prompt_experiments.py`** — Contains all 6 prompt versions (b.v0–b.v6) hardcoded in `PromptVersionParser._get_hardcoded_prompts()`. Versions progressively add: parameter explanations → externality concept → submodularity → structured framing → full mathematical formulas.

### Model Configuration

`configs/model_configs.json` (gitignored) — one entry per LLM:
```json
{
  "config_name": "my-model",
  "model_type": "openai_chat",
  "model_name": "model-id-from-provider",
  "api_key": "sk-...",
  "client_args": { "base_url": "https://api.provider.com/v1" },
  "generate_args": { "temperature": 0.5 }
}
```

Adding a new model requires only adding an entry here — no code changes needed.

### Output Structure

- `data/ground_truth/` — Pre-generated theoretical benchmark JSONs
- `evaluation_results/` — Per-model JSONs + summary CSVs; subfolders: `prompt_experiments_b/`, `eas_analysis/`, `fp_deepseek-r1/`, `fp_gpt-5/`
- `sensitivity_results/scenario_b/` — 3×3 grid runs, `model_comparison_plots/`, `gpt_family_comparison_plots/`

### 12 Evaluated LLMs

GPT-5, GPT-5.1, GPT-5.1-2025-11-13, GPT-5.2 (OpenAI); DeepSeek-v3-0324, DeepSeek-v3.1, DeepSeek-v3.2, DeepSeek-r1; Qwen-Plus, Qwen-Plus-2025-12-01, Qwen3-Max, Qwen3-Max-2026-01-23.

---

## Phase 2: Intelligent Research Framework

### Research Stages

- **Phase 1 (complete):** Academic research for KDD/TKDE submission. All three scenarios (A/B/C) implemented. Results and visualizations frozen.
- **Phase 2 (current):** Intelligentize and structurize the research framework. The core goal is that researchers provide a paper; agents handle scenario extraction, solver construction, market simulation design, experiment execution, and analysis. Human researchers only define methodology and hold three review gates.

### Phase 2 Implementation Status

- **Scope**: Scenario B only (Acemoglu et al. 2022 "Too Much Data"). Scenarios A and C are out of scope for Phase 2.
- **Tech stack**: OpenAI-compatible SDK (provider-agnostic). `agents_complete.py` (old AgentScope code) is archived and irrelevant.
- **Implementation plan**: Plan A first (shared files + Python coordination), then migrate to Plan B (Claude Code multi-agent SDK, true concurrency). Tool signatures are identical in both plans — only the message routing changes.
- **Full design document**: `phase2/DESIGN.md`
- **Code location**: `phase2/` directory
- **User config**: `phase2/config.json` (gitignored) — copy from `config.example.json`, set provider/model/api_key/base_url.

#### Layer completion

| Layer | Status | Notes |
|-------|--------|-------|
| Layer 1 — Understanding | **Complete & tested** | Full end-to-end run passed, Gate 1 approved |
| Layer 2 — Design | Not started | Market Designer + Orchestrator |
| Layer 3 — Execution | Not started | Runner + Analysis |
| Layer 4 — Optimization | Not started | Prompt Engineer |

#### Known issues & fixes applied

- **Rate limit**: Agent API (vectorengine.ai / Claude proxy) has strict RPM. Layer 1 runs agents sequentially (Paper → Extractor → Solver Builder). Solver Builder uses `max_rate_limit_retries=1` to fail fast to fallback solver.
- **Solver Builder fallback**: When API is rate-limited, `_write_fallback_solver()` writes a validated Phase-1-equivalent solver and signs off. Validation: all 3 tests pass.
- **Extractor always uses fallback paradigm**: Claude calls `write_paradigm` but the model's tool call may not fire; `_write_default_paradigm()` is the reliable path. Output is correct.
- **XML wrapping**: Paper text passed to Claude must be wrapped in `<paper_content>` tags to prevent false prompt-injection detection.
- **`_wait_for_reply` race**: Fixed — polls by `reply_to` id across all statuses, not by `status=pending`.

### Phase 2 Directory Structure

```
phase2/
├── DESIGN.md                  # Full Layer 1–4 design document
├── config.example.json        # Copy to config.json and fill in API credentials
├── requirements.txt           # anthropic / openai / pdfplumber / pyyaml
├── layer1_coordinator.py      # Layer 1 entry point (Plan A)
├── workspace/                 # Shared memory for all agents
│   ├── input/paper.pdf        # Researcher provides this
│   ├── paper_parse.json       # Paper Agent output ✓
│   ├── paradigm.yaml          # Scenario Extractor output ✓
│   ├── solver/solver_b.py     # Solver Builder output ✓
│   ├── market_config.yaml     # Market Designer output (Layer 2, pending)
│   ├── raw_results/           # Runner output (Layer 3, pending)
│   ├── metrics/               # Analysis output (Layer 3, pending)
│   ├── hypothesis_status.json # (Layer 3, pending)
│   ├── messages/              # Inter-agent messages (Plan A)
│   ├── gate1_request.json     # Layer 1 joint sign-off ✓
│   └── gate1_approved.json    # Human approval ✓
├── agents/
│   ├── paper_agent.py         # ✓ tested
│   ├── scenario_extractor.py  # ✓ tested
│   ├── solver_builder.py      # ✓ tested (fallback path)
│   ├── market_designer.py     # pending
│   ├── orchestrator.py        # pending
│   ├── runner.py              # pending
│   ├── analysis.py            # pending
│   └── prompt_engineer.py     # pending
├── tools/
│   ├── file_io.py             # ✓ tested
│   ├── agent_api_client.py    # ✓ tested (supports anthropic + openai-compatible)
│   └── pdf_reader.py          # ✓ tested
└── tests/                     # BaseTest + 3 registered types (pending)
```

### Core Design Principles

**1. Paradigm-as-Contract**
Researchers define *what to test and why*; agents implement *how to run the tests*. The paradigm YAML is the stable contract between researcher intent and agent behavior.

**2. Collaborative Network, Not Pipeline**
Agents communicate peer-to-peer. Multiple agents work in parallel and iterate with each other. The Orchestrator is a meta-coordinator that intervenes only for cross-layer resource decisions or unresolvable conflicts — not a central router every message passes through.

**3. Input = Paper PDF**
The system starts from an academic data market / privacy externality paper. Agents extract the scenario, formalize hypotheses, and build the solver. Ground Truth is produced by a deterministic solver; its mathematical correctness is the human researcher's responsibility.

**4. Test Registry for Extensibility**
Test types are registered abstractions, not hardcoded logic. New scenarios that require new test forms trigger an agent-driven process to define and register a new `BaseTest` subclass, reviewed by a human before use.

---

### System Architecture: Eight Agents Across Four Layers

![Phase 2 Agent Architecture](docs/phase2_architecture.png)

All agents share a **Shared Memory** workspace:
`paper_parse.json` / `paradigm.yaml` / `market_config.yaml` / `solver code` / `raw_results/` / `metrics/` / `hypothesis_status`

---

#### Layer 1 — Understanding Layer

Three agents form a working group around the paper. They iterate peer-to-peer until paradigm and solver jointly converge, then submit together for Gate 1.

| Agent | Core Question | Peer Interactions |
|---|---|---|
| **Paper Agent** | What does this paper say? | ↔ Scenario Extractor: re-read targeted sections on demand; ↔ Solver Builder: clarify math notation and boundary conditions |
| **Scenario Extractor Agent** | What game is this? | ↔ Paper Agent: resolve ambiguous formulations; ↔ Solver Builder: ensure hypotheses H1/H2/H3 are numerically testable |
| **Solver Builder Agent** | How to compute the equilibrium? | ↔ Scenario Extractor: negotiate testable hypothesis forms; ↔ Paper Agent: validate solver against paper's numerical examples |

Solver Builder self-validates: analytical solution vs. numerical solution at special parameter values; boundary condition sanity checks (e.g. ρ=0, n=1); comparison against any numerical examples in the paper.

> **Gate 1 — Solver Review**: All three agents jointly sign off → human validates mathematical correctness → solver enters `solvers/` and GT generation proceeds.

---

#### Layer 2 — Design Layer

Agents negotiate experiment design. No single agent decides unilaterally; cost, feasibility, and scientific value are jointly weighed.

| Agent | Core Question | Peer Interactions |
|---|---|---|
| **Market Designer Agent** | How to set up the simulated market? | ↔ Analysis Agent: what parameter ranges best distinguished models historically; ↔ Runner Agent: implementation feasibility and cost estimate; ↔ Orchestrator: scope and resource tradeoffs |
| **Orchestrator Agent** | What experiments to run now? Is coverage sufficient? | ↔ Market Designer: parameter space declaration; ↔ Analysis Agent: hypothesis coverage assessment; ↔ Runner Agent: cost and concurrency estimate |

Market Designer is responsible for:
- **Role assignment**: which participants are LLMs vs. rule-based solvers; how many LLM agents; what roles (consumer / intermediary / data seller)
- **Information structure**: what each role observes; complete vs. incomplete information; public knowledge vs. private signals; asymmetry design
- **User heterogeneity**: type distributions (θ_i prior), correlation ρ parameterization, per-user differences in an n-user market
- **Interaction protocol**: simultaneous / sequential / iterative (Fictitious Play); max rounds; convergence condition; per-round belief update rule
- **Initial role prompt templates**: system prompt per role, decision output format, background knowledge injection level

Orchestrator carries a **Test Strategist** capability: reads each hypothesis's `nature` label in the paradigm YAML and selects the appropriate test type from the Test Registry. If no registered type matches, triggers the new test type design process (see Test Registry below).

> **Gate 2 — Market Design Review**: Design layer agents reach consensus → human validates that role setup and heterogeneity settings faithfully represent the paper's model.

---

#### Layer 3 — Execution Layer

Agents coordinate dynamically during runs, not only at handoff boundaries.

| Agent | Core Question | Peer Interactions |
|---|---|---|
| **Runner Agent** | How to get LLMs running reliably? | ↔ Analysis Agent: receive mid-run feedback; adjust trial count per condition without waiting for full completion |
| **Analysis Agent** | What do results mean? Are hypotheses validated? | ↔ Runner Agent: authorize dynamic trial adjustment; ↔ Orchestrator: report hypothesis PASS/FAIL/INCONCLUSIVE status |

Runner responsibilities: concurrent LLM trial execution, structured logging per call (run_id, role, prompt, raw response, latency, token count), retry with error classification (API timeout / JSON parse failure / rate limit / content refusal), checkpoint/resume.

Analysis responsibilities:
- **Outcome layer**: MAE on share_rate, profit, social welfare
- **Mechanism layer**: Jaccard (structural consistency), EAS (directional consistency), Counterfactual Reasoning Score (see below)
- **Process layer**: FP convergence rounds, strategy stability CV
- Hypothesis verification: PASS / FAIL / INCONCLUSIVE per H_i
- Mechanism diagnosis labels: *Strong Understanding* (high Jaccard + positive EAS) / *Surface Fitting* (high Jaccard + negative EAS, i.e. mechanism hallucination) / *Weak Understanding* (low Jaccard)
- Figure generation (quadrant plots, heatmaps, convergence curves) and LaTeX fragment output

**Counterfactual Reasoning Score** (inspired by COMA's counterfactual idea, implemented as an evaluation metric — no RL training involved): present LLM with a counterfactual prompt ("if ρ increases from 0.3 to 0.7, how does your decision change and why?"), compare directional reasoning against theoretical comparative statics. This is the fourth Mechanism-layer metric and the deepest test of mechanism understanding.

---

#### Layer 4 — Optimization Layer

Triggered only when Analysis identifies prompt quality as the bottleneck, not information quantity or market design.

| Agent | Core Question | Peer Interactions |
|---|---|---|
| **Prompt Engineer Agent** | Why did the prompt fail? How to fix it? | ↔ Analysis Agent: obtain failure cases and error pattern classification; ↔ Market Designer: disambiguate prompt failure vs. information design failure before proposing a fix |

> **Gate 3 — Prompt Change Review**: Optimization layer three-way discussion (Analysis + Market Designer + Prompt Engineer) converges on root cause → human confirms that the proposed new prompt version does not break cross-version comparability → new version enters `role_prompts/` and Orchestrator schedules a comparison run.

---

### Test Registry & Extensibility

The three current test types are instances of abstract test classes. Each implements `BaseTest`:

```python
class BaseTest:
    def setup(self, params) -> TestEnv: ...
    def run(self, env, model) -> RawResult: ...
    def evaluate(self, raw, ground_truth) -> Metrics: ...
    def verify_hypothesis(self, metrics, hypothesis) -> HypothesisResult: ...
```

| Abstract Class | Current Instance | Hypothesis Nature Tag |
|---|---|---|
| Knowledge Injection Test | Prompt Ladder (v0→v5) | `knowledge_dependent` |
| Comparative Static Test | Sensitivity Analysis (3×3 grid) | `comparative_static` |
| Dynamic Convergence Test | Fictitious Play | `dynamic_convergence` |

Paradigm YAML declares hypothesis nature so Orchestrator can select the right test type:

```yaml
hypotheses:
  - id: H1
    statement: "ρ increases → equilibrium share_rate decreases"
    nature: comparative_static
    preferred_test: SensitivityTest
    parameters_to_vary: [rho]
    success_criterion: "EAS < 0 for all rho grid points"
  - id: H2
    statement: "More externality context → better mechanism understanding"
    nature: knowledge_dependent
    preferred_test: PromptLadderTest
    success_criterion: "Jaccard monotone increasing v0→v5"
  - id: H3
    statement: "LLM converges to BNE through belief updating"
    nature: dynamic_convergence
    preferred_test: FictitiousPlayTest
    success_criterion: "strategy_delta < 0.01 within 20 rounds"
```

**When no registered test type matches a new scenario's hypothesis**: Scenario Extractor flags the gap → Solver Builder + Market Designer collaboratively implement a new `BaseTest` subclass → human approves → registered permanently for reuse by future scenarios.

---

### Human Researcher's Role

| Before Phase 2 | Phase 2 |
|---|---|
| Read paper, write paradigm YAML | Provide paper PDF |
| Write solver code | Review agent-built solver (Gate 1) |
| Design market simulation | Review agent-designed market setup (Gate 2) |
| Design prompt experiments | Review agent-proposed prompt changes (Gate 3) |
| Analyze results manually | Review agent-generated hypothesis reports |

Researchers go from *doing experiments* to *gatekeeping experiment validity*.

---

### Key Collaborative Patterns

**Understanding iteration**: Scenario Extractor cannot formalize H1 due to ambiguous comparative statics in paper → queries Paper Agent to re-read appendix → Solver Builder proposes finite-difference numerical verification as substitute for analytical monotonicity proof → three agents converge on testable form.

**Cost negotiation**: Market Designer proposes 5-user × 12-model × 9-trial grid → Runner estimates 1,620 LLM calls (~$180, 6 hours) → Orchestrator proposes 3-model pilot first → Analysis confirms pilot sufficient to validate H1 directionality → Market Designer revises to two-phase plan.

**Mid-run dynamic adjustment**: Runner observes 3× higher variance in ρ=0.9 condition at trial 15/30 → immediately notifies Analysis Agent → Analysis authorizes doubling trials for ρ=0.9 only → Runner adjusts without stopping the run or escalating to Orchestrator.

**Optimization triangle**: Analysis flags systematic failure on "correlated types" concept across all models → Market Designer argues it may be insufficient information in prompt context, not prompt wording → Prompt Engineer examines raw outputs, finds models acknowledge correlation but misapply its direction → root cause is conceptual gap, not information gap → Prompt Engineer drafts v_new with explicit externality direction explanation → Gate 3.

---

### Structural Principle

```
Pipeline (rejected):       A → B → C → D → E

Collaborative network      Layer 1:  Paper ↔ Extractor ↔ SolverBuilder
(adopted):                               (iterate to Gate 1)

                           Layer 2:  MarketDesigner ↔ Orchestrator
                                          ↕               ↕
                                       Runner         Analysis
                                     (cost est.)   (coverage check)
                                               → Gate 2

                           Layer 3:  Runner ↔ Analysis (live feedback)
                                          ↕
                                      Orchestrator (hypothesis loop)

                           Layer 4:  Analysis ↔ MarketDesigner ↔ PromptEngineer
                                               → Gate 3
```
