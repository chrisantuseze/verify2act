# RA-L Submission Plan — Verify2Act

> **Goal:** Produce a strong, credible RA-L submission that properly showcases Verify2Act's core benefits.
> **Priority order:** correctness → evaluation depth → baselines → new tasks (if feasible).

---

## ✅ Task Checklist

### Phase 1 — Critical Fixes (Must Do)
- [ ] Fix abstract to match actual results
- [ ] Fix experiment section (remove 25-episode language, update all numbers)
- [ ] Rewrite claims to be precise and not use "significantly outperform" without qualification

### Phase 2 — Evaluation Depth (High Impact, Low Cost)
- [ ] **CALVIN**: Increase evaluation to ≥100 sequences (ABC→D split, 5-task chains)
- [ ] **Nut Assembly**: Run full eval with 100–200 episodes, report mean ± std over seeds
- [ ] Report **chain-length degradation curve** (tasks 1→5 per-method) — this is V2A's "money figure"

### Phase 3 — Baselines (CALVIN only)
- [ ] Add **HULC** to CALVIN table (run with existing weights from calvin repo)
- [ ] **Cite GR-1 numbers** from their paper directly (zero compute cost)
- [ ] **Cite SuSIE numbers** from their paper directly
- [ ] Report **HULC-only** performance (low-level policy without V2A planning layer)
- [ ] Report **MoDE-only** performance (diffusion low-level policy without V2A planning layer)

### Phase 4 — Ablations (High Value for Novelty Argument)
- [ ] Run ablation table on CALVIN: w/o Cross-Attention, w/o Temporal History, w/o Sparsity
- [ ] Report critic precision / NCR metrics in main table or appendix

### Phase 5 — Optional New Task (Only If Time Permits)
- [ ] Evaluate on **Block Stacking** (Robosuite Stack) — see feasibility estimate below
- [ ] OR: Keep Nut Assembly but improve framing (see narrative strategy below)

---

## 📊 Baseline Strategy

### CALVIN Table — What to Include

| Method | Source | Run it? | Notes |
|---|---|---|---|
| VLM-Only (GPT-4o) | Your system | ✅ Already have | Keep as anchor |
| ReflectVLM | Your system | ✅ Already have | Pixel-space baseline |
| DINO-WM (CEM) | Your system | ✅ Already have | Latent WM baseline |
| RLA-WM | Your system | ✅ Already have | Direct predecessor |
| **HULC** | Official weights | ✅ **Run eval** | Popular CALVIN baseline; included in calvin repo |
| **GR-1** | Their paper | 📄 **Cite from paper** | Top CALVIN method; cite ABC→D numbers directly |
| **SuSIE** | Their paper | 📄 **Cite from paper** | Pixel-space subgoal; cite ABC→D numbers |
| HULC-only (your env) | Your system | ✅ Run | Shows V2A adds value over raw HULC |
| MoDE-only (your env) | Your system | ✅ Run | Shows V2A adds value over raw MoDE |
| **Verify2Act (Full)** | Your system | ✅ Already have | Your method |
| Ablations (3 rows) | Your system | ✅ Run on CALVIN | Directly address novelty gap |

> [!IMPORTANT]
> **Baselines cited from paper (GR-1, SuSIE) only work if the evaluation protocol matches.** 
> Confirm they use `task_ABC_D` split and report the same 1→5 chain metric.
> If they use a different split or metric, you must note this explicitly and either skip them 
> or add a footnote explaining the discrepancy.

### Nut Assembly Table — What to Include

| Method | Notes |
|---|---|
| VLM-Only (GPT-4o) | Anchor |
| DINO-WM | Latent WM comparison |
| RLA-WM | Direct predecessor |
| ReflectVLM | Pixel-space |
| Verify2Act (Ablations) | 3 ablation rows |
| **Verify2Act (Full)** | Your method |

No new baselines needed on Nut Assembly. The scripted-primitive constraint makes cross-system comparison unfair and should be stated explicitly in the paper.

---

## 🔑 Narrative Strategy: Making V2A Shine

The key insight: **CALVIN's 5-task chain metric is the best showcase of Verify2Act's benefit.**

### Why the Chain-Length Degradation Curve Is the "Money Figure"

- Baseline methods degrade steeply as chain length increases (1-task → 5-task success rate drops)
- Verify2Act's critic prunes incorrect sub-plans *before execution*, so compounding errors are caught early
- Expected result: **V2A degrades more slowly than baselines across the 1→5 chain**
- This directly proves the core claim: "latent verification prevents cascading failures"

### Paper Narrative Adjustment

Reframe the comparison not just as "outperforms baselines" but:

> *"Verify2Act maintains high success rates on long chains where purely reactive methods fail — the performance gap widens with horizon length, confirming that critic-guided verification prevents the cascade failures that limit existing approaches."*

This is a claim you can make even if V2A doesn't dominate at chain length 1 (where all methods do well), as long as the gap opens up at lengths 3–5.

### Nut Assembly Narrative Fix

Instead of claiming general SOTA, frame it as:
> *"Verify2Act achieves [X]% success on Nut Assembly, demonstrating effective latent verification for precision peg-insertion tasks with scripted primitives. Comparison with end-to-end visuomotor policies is left as future work since those systems operate over continuous action spaces rather than discrete primitive sets."*

This is a **respected and accepted** scope statement in robotics papers.

---

## 🧱 New Task: Robosuite Block Stacking — Feasibility Estimate

### What You Already Have

You already have `robosuite/run_stack.py` with a complete `HeuristicStackPolicy` covering Stack, Stack3, and Stack4 environments. This is the **largest time-saving factor** in the estimate below.

### Time Estimate (With Existing Infrastructure)

| Component | Effort | Notes |
|---|---|---|
| Stack env wrapper (like NutAssemblyEnvWrapper) | 1.5–2 days | You can clone `env_wrapper.py` and adapt it; state machine primitives for "pick cube X, stack on Y" are simpler than nut assembly |
| VLM prompt adaptation | 0.5 days | Block stacking has a cleaner language interface than nuts |
| Data collection (heuristic policy demo collection) | 0.5–1 day | ~10k–30k transitions; `HeuristicStackPolicy` already works, just add trajectory recording |
| Autoencoder + WM training | 2–3 days | Training from scratch on new environment data; same code, different dataset |
| Critic training | 1 day | Contrastive critic on new data |
| Inference pipeline integration | 1 day | Adapt `inference.py` to use new env wrapper |
| Evaluation runs | 1 day | 100 episodes per method |
| **Total** | **~8–11 days** | Assumes no major debugging surprises |

> [!WARNING]
> **This is a best-case estimate for someone with your existing infrastructure.** 
> If the heuristic policy needs tuning for stable data collection, or if the WM needs more
> training epochs, add 3–5 days. Without existing infra, this would be 3–4 weeks.

### Should You Do It?

**Honest recommendation: Only if you have ≥2 weeks and the Nut Assembly results are strong.**

Block stacking offers better V2A showcasing because:
- Multiple cubes = ambiguous stacking order (VLM may propose wrong order)
- Critic verifies which stacking sequence is physically coherent
- Occlusion during stacking tests the temporal history head directly

But it's not necessary if:
- Your CALVIN chain-length curve tells the story clearly
- Your Nut Assembly numbers are solid
- You're on a tight RA-L timeline

**Alternative:** Use Stack as a *qualitative* demo in the supplementary video (it's already integrated) without adding it as a full quantitative benchmark.

---

## 📋 Revised Paper Table Structure

### Table 1: CALVIN (ABC→D, 5-task chain)

```
Category          | Method              | T1  | T2  | T3  | T4  | T5  | Avg.
------------------+---------------------+-----+-----+-----+-----+-----+-----
Pixel-Space       | VLM-Only (GPT-4o)   |     |     |     |     |     |
                  | ReflectVLM [cite]   |     |     |     |     |     |
                  | SuSIE [paper cite]  |     |     |     |     |     |  ← from paper
Latent Feature    | DINO-WM (CEM)       |     |     |     |     |     |
                  | RLA-WM              |     |     |     |     |     |
                  | GR-1 [paper cite]   |     |     |     |     |     |  ← from paper
Policy-Only       | HULC (standalone)   |     |     |     |     |     |  ← run eval
                  | MoDE (standalone)   |     |     |     |     |     |  ← run eval
Verify2Act (Ours) | w/o Cross-Attention |     |     |     |     |     |
                  | w/o Temporal History|     |     |     |     |     |
                  | w/o Sparsity        |     |     |     |     |     |
                  | V2A + HULC (Full)   |     |     |     |     |     |  ← main result
                  | V2A + MoDE          |     |     |     |     |     |  ← additional result
```

> [!NOTE]
> **Key new insight from having HULC and MoDE integrated:** You can now show two things simultaneously:
> 1. The V2A *planning layer* adds value over raw policy execution (HULC-only vs V2A+HULC)
> 2. The modular design works with multiple low-level policies (V2A+HULC vs V2A+MoDE)
> This is a much stronger paper than a pure ablation approach — it demonstrates **generalizability** of the verification framework.

### Table 2: Nut Assembly

```
Category          | Method              | Single Nut (%)  | Two Nuts (%)
------------------+---------------------+-----------------+--------------
Pixel-Space       | VLM-Only            |                 |
                  | ReflectVLM          |                 |
Latent Feature    | DINO-WM             |                 |
                  | RLA-WM              |                 |
Verify2Act (Ours) | w/o Cross-Attention |                 |
                  | w/o Temporal History|                 |
                  | w/o Sparsity        |                 |
                  | Verify2Act (Full)   |                 |
```

---

## 🚀 Execution Timeline (Suggested Order)

```
Week 1:
  Days 1–2:  CALVIN eval increase (100+ sequences, all existing methods)
  Days 3–4:  HULC standalone eval + HULC-only vs V2A+HULC comparison on CALVIN
  Day 5:     MoDE-only vs V2A+MoDE comparison on CALVIN
  
Week 2:
  Days 1–2:  CALVIN ablation runs (3 ablation rows)
  Days 3–4:  Nut Assembly rerun (100–200 episodes)
  Day 5:     Buffer / debugging

Week 3:
  Days 1–3:  Paper rewrite (abstract, intro, experiment section)
  Days 4–5:  Figure updates, table population, response to reviewer concerns

Week 4 (optional):
  If Block Stacking: begin env wrapper + data collection
  Otherwise: polish, supplementary video, submission
```

---

## 🗒️ Key Caveats for Paper Text

1. **On citing GR-1/SuSIE from paper:** Add footnote — *"Numbers for GR-1 and SuSIE are reported from the original papers using the same `task_ABC_D` evaluation split."*

2. **On scripted primitives:** Add one sentence to experimental setup — *"Verify2Act operates over a fixed set of high-level primitives; comparison with end-to-end visuomotor policies trained over continuous action spaces is deferred to future work."*

3. **On HULC/MoDE integration:** Emphasize in the intro that this modularity is a **feature** — the verification layer is policy-agnostic and improves any low-level executor.

4. **On API cost:** No longer an issue for the expanded eval if you apply for research credits (Anthropic, OpenAI, Google all have PhD programs).
