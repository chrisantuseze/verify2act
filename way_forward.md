# Verify2Act — RA-L Submission & Way Forward

This document outlines the current tasks, completed items, and timeline for revising the Verify2Act manuscript for submission to **IEEE Robotics and Automation Letters (RA-L)**. 

---

## 🚀 Completed Tasks (Ready for Paper)

The following items have been successfully implemented and integrated into the manuscript (`main.tex` and `refs.bib`):

*   **ODE Multi-Step Fix**: Corrected the erroneous "single Euler ODE step" claim to "5-step Euler ODE solver ($\Delta\tau = 0.2$)" based on the actual codebase implementation (`dynamics.py`).
*   **Baselines Selection**: Removed **DINO-WM** from all baseline comparisons (per discussion, **RLA-WM** is sufficient as the latent-space world model baseline).
*   **Baselines Scope Clarification**: Added detailed explanations for why **GR-1**, **HULC**, and **MoDE** are compared only on the CALVIN benchmark and omitted from Nut Assembly.
*   **Related Work Extension**: Wove discussions of **GR-1**, **HULC**, and **MoDE** directly into Section II-A (*VLMs and Foresight in Robotics*) without creating redundant subsections.
*   **Reference Updates**: Added bibtex references for GR-1 (`wu2024gr1`) and MoDE (`reuss2024mode`) to `refs.bib`.

---

## 📌 Current Action Plan (Pending Items)

### Phase 1 — Experimental Reruns & Evaluation (Priority: 🔴 Critical)
The tables in the manuscript currently contain placeholder values (`--`). We need to run evaluations and fill in these numbers.
- [ ] **Rerun Nut Assembly with the correct model** (50–100 episodes; previously only 25 episodes were evaluated with an incorrect model).
- [ ] **Rerun CALVIN evaluation** (at least 100 sequences; previously only 20 sequences).
- [ ] **Collect baseline numbers for CALVIN**:
    - Extract published performance numbers for **GR-1**, **HULC**, and **MoDE** (no need to rerun locally).
    - Evaluate **VLM-Only**, **RLA-WM**, and **ReflectVLM** under identical evaluation conditions.
- [ ] **Evaluate Ablations** (Nut Assembly SR%):
    - [ ] Base (RLA-WM)
    - [ ] w/o Cross-Attention Grounding
    - [ ] w/o Temporal History
    - [ ] w/o Sparsity Regularization
- [ ] **Fill in Tables 1 & 2** in `main.tex` once evaluations are finished.

### Phase 2 — Writing & Manuscript Refinement (Priority: 🔴 Critical)
- [ ] **Fix Abstract Claims**: Rewrite the abstract and intro to tone down the blanket claim of "significantly outperform SOTA". Frame it realistically: *V2A-WM outperforms latent-space baselines on CALVIN and achieves competitive performance on Nut Assembly while providing higher planning/pruning efficiency (NCR).*
- [ ] **Rewrite Experiments Narrative**: Write the description and interpretation of the results ourselves (avoid delegating this to LLMs).
- [ ] **Generate Qualitative Filmstrip Figure**:
    - Implement/run `compare_imaginations.py` or `visualize_wm.py` to extract actual frame rollouts.
    - Replace the placeholder comment in `main.tex` (around line 312) with a beautiful figure showcasing V2A-WM's early-pruning behavior vs. baselines.

### Phase 3 — Technical Clarifications & Reviewer Concerns (Priority: 🟡 Medium)
- [ ] **Training Data Details**: Add explicit counts and sources of training demonstrations (addresses Reviewer Q1).
- [ ] **Threshold Sensitivity Analysis**: Add a discussion/analysis on how the critic thresholds ($\theta_{\text{conf}}$, $\theta_c$, $\theta_p$) affect performance (addresses Reviewer Q2).
- [ ] **Solver Speed Justification**: Add a brief discussion on the Euler step solver choice (5-step solver chosen as a sweet spot for planning latency vs. trajectory accuracy) (addresses Reviewer Q3).
- [ ] **Wall-Clock Latency**: Add latency comparison against ReflectVLM (addresses Reviewer Q4).
- [ ] **Failure Mode Analysis**: Document common failure modes observed in Nut Assembly (e.g., threshold sensitivity, aggressive pruning) (addresses Reviewer Q5).

### Phase 4 — High-Value Additions (Priority: 🟢 Low / Optional)
- [ ] **Calibration Curves**: Plot confidence calibration curves for the critic thresholds.
- [ ] **SR@N Distribution**: Plot Success Rate vs. Chain Length ($N$) for the CALVIN benchmark.
- [ ] **Apply for VLM API Credits**: Apply for Anthropic, OpenAI, or Google research API credits to offset costs.

---

## 📅 Estimated Timeline

| Phase / Task | Effort (Days) | Priority |
| :--- | :--- | :--- |
| **Nut Assembly & CALVIN Reruns** | 2–3 Days | 🔴 Critical |
| **Component Ablation Runs** | 1–2 Days | 🔴 Critical |
| **Abstract & Experiments Rewrite** | 1 Day | 🔴 Critical |
| **Technical Clarifications (Q1–Q5)** | 1 Day | 🟡 Medium |
| **Qualitative Filmstrip Figure** | 0.5 Day | 🟡 Medium |
| **Final Proofreading & Polish** | 0.5 Day | 🟢 Low |

**Total Estimated Effort: ~6–8 days of focused work.**
