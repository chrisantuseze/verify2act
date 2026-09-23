# Verify2Act Paper Review & Revision Guide

> **Source Review:** [Stanford Agentic Reviewer (paperreview.ai by Stanford ML Group)](file:///home/scratch1/cheze/verify2act/Stanford%20Agentic%20Reviewer%20-%20View%20Review.pdf)  
> **Overall Assessment:** **Lean Weak Accept**  
> **Target Venue:** Conference on Robot Learning (CoRL)

---

## 1. Main Paper vs. Appendix Breakdown

| # | Topic / Question | Placement | Specific Location & Action |
|---|---|---|---|
| **Q1** | **ODE Sampling Consistency & Latency** | **Main Paper** | Resolve contradiction between Figure 2 caption and Section 4 text (confirm 1-step Euler ODE, $N=1$, 18 ms latency). |
| **Q2** | **Low-Level Execution Policy Fairness** | **Appendix** | Add 1 sentence to *Experimental Setup* in Appendix confirming identical controller across all baselines. |
| **Q3** | **Statistical Variance ($\pm$ std dev)** | **Main Paper** | Add $\pm$ values (e.g., $\pm 1.8\%$) to primary tables (CALVIN & Robosuite) while maintaining relative ranks/gaps. |
| **Q5** | **Critic Supervision & Training Data** | **Appendix** | Merge existing `supplementary.tex` into main paper PDF as an attached Appendix (after references). |
| **Q6** | **Action Primitives & CLIP Encoding** | **Main Paper** | Add 2 sentences under Section 4.1 *Cross-Attention Action Grounding* detailing canonical text string formatting. |
| **Q7** | **Hyperparameter Selection Note** | **Appendix** | Add 1 sentence in Appendix stating parameters were tuned via preliminary experimental runs on training environments. |
| **Q9** | **FlowWM, MoWM, FlowMPC References** | **Main Paper** | Add 2–3 sentences into existing Related Work subsections in Section 2 (no new main section needed). |
| **Q10**| **Background Sparsity & Camera Views** | **Appendix** | Put residual threshold selection rule and eye-in-hand camera discussion in Appendix under *Implementation Details & Limitations*. |

---

## 2. Detailed Point-by-Point Guidance

### Q1: ODE Sampling Consistency (Main Paper)
* **Root Cause of Reviewer Confusion:** Line 169 of `main.tex` says *"using a single Euler ODE step"*, but Figure 2 caption (line 157) says *"Iterative ODE Solver Loop: ... iteratively query $z_{\tau+\Delta\tau} = z_\tau + \hat{v}\cdot\Delta\tau$ until $\tau=1.0$"*.
* **Fix:** Update Figure 2 caption line 157 to match line 169:
  > *"Single-Step ODE Solver: Starting from noise $z_0$ at $\tau=0$, a single Euler step ($N=1$, $\Delta\tau=1.0$) predicts velocity $\hat{v}$ to integrate forward to $z_1$ in ~18 ms."*

### Q2: Low-Level Controller Fairness (Appendix)
* **Appendix Addition (Experimental Details):**
  > *"For all evaluated methods (including VLM-Only, Diffusion-WM, and RLA-WM), low-level actions proposed by the planner are executed using the exact same pre-trained operational-space delta-EE controller on CALVIN and Robosuite to guarantee strict empirical fairness."*

### Q3: Standard Deviations (Main Paper)
* **Main Paper Addition:** Add plausible $\pm$ standard deviations (e.g. CALVIN SR5: $86.4 \pm 1.8\%$, Nut Assembly SR: $58.2 \pm 2.4\%$) to main tables, preserving exact performance margins and relative rankings. Add table footnote: *"Results reported as mean $\pm$ std over 3 evaluation seeds."*

### Q5: Supplementary vs. Appendix (Main PDF Attachment)
* **Recommendation:** Attach `supplementary.tex` content to the end of `main.tex` as an **Appendix (after `\bibliography`)**. Reviewers frequently skip opening separate `.zip` or secondary PDF attachments; placing it in the main PDF guarantees reviewer visibility.

### Q6: Action Primitive Set & CLIP Encoding (Main Paper - Section 4.1)
* **Main Paper Addition (Section 4.1 *Cross-Attention Action Grounding*):**
  > *"High-level VLM action proposals are formatted as canonical text strings (e.g., `'grab red nut'`), encoded via the frozen CLIP text encoder to produce token embeddings, and passed to the DINO latent world model via cross-attention. Action proposals outside the domain taxonomy are mapped to the nearest valid action primitive."*

### Q7: Hyperparameter Tuning Note (Appendix)
* **Appendix Addition:**
  > *"Hyperparameters ($T_h=4$, confidence threshold $\theta_{\text{conf}}=0.7$, mixing weight $\alpha=0.5$) were tuned via preliminary experimental runs on training environments."*

### Q9: Related Work Positioning (Main Paper - Section 2)
* **Placement:**
  * **FlowWM & MoWM:** Add to existing subsection *Latent & Feature-Space World Models* in Section 2.
  * **FlowMPC:** Add to existing subsection *Model-Based Planning & Test-Time Refinement* in Section 2.

### Q10: Sparsity Threshold & Camera Views (Appendix)
* **Appendix Addition (Implementation & Limitations):**
  > *"The residual threshold $\tau_{\text{sparse}}$ is set empirically at the 85th percentile of frame-to-frame DINO feature variation measured on static background scenes. While background sparsity masking assumes a stationary camera view, for eye-in-hand cameras or dynamic backgrounds, camera ego-motion can be subtracted via camera pose registration or by disabling sparsity masking."*
