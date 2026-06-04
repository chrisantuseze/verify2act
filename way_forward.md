# Verify2Act — CoRL Situation & Way Forward

## What Actually Happened (Root Cause)

| Issue | Severity | Fixable? |
|---|---|---|
| Abstract claims "significantly outperform SOTA" but Table 1 reports the old Nut Assembly run (52% ≈ same as DINO-WM, below RLA-WM/ReflectVLM 72%) | **Critical** — claim–result mismatch | ✅ Yes, with correct rerun |
| Experiment section written by Gemini using an older run's numbers without being caught | **Critical** — reproducibility undermined | ✅ Yes (lesson for next time) |
| Only 25 Nut Assembly episodes; only 20 CALVIN sequences — both too low for statistical credibility | **Serious** — reviewers flagged this directly | ✅ Yes, easy to rerun |
| No ablation table isolating V2A-WM components (cross-attn, temporal history, sparsity) | **Moderate** — weakens novelty argument | ⚠️ Feasible but takes time |
| Missing comparisons: GHIL-Glue, LUMOS (latent planning), GR-1/end-to-end policies | **Moderate** — contextualizes the approach | ⚠️ Can address partially in writing |
| Single-step Euler ODE integration not justified | **Minor** — technical clarification needed | ✅ Yes (writing fix + optionally experiment) |
| Training data/demonstration counts not specified | **Minor** — reproducibility concern | ✅ Writing fix |

---

## Immediate Decision: Supplementary by Thursday?

> [!IMPORTANT]
> **You likely should NOT kill yourself over Thursday's supplementary deadline.**
> 
> The core credibility issue (abstract vs. results mismatch) cannot be fixed in the main submission. Even a perfect video will not overcome a reviewer who catches that contradiction. The effort-to-reward ratio is low.

**Recommended call: Skip the supplementary** unless the video is already ≥70% done. Withdraw before the review phase begins if possible, to preserve your reputation at the venue and prevent a reviewable record of the submission.

> [!NOTE]
> Some venues let you withdraw before reviewers are assigned. Check CoRL's timeline — if you can withdraw cleanly without a review record, that's preferable to a rejection with the mismatch documented.

---

## The Real Path Forward: RA-L Submission

RA-L is a strong venue for this work — it's robotics-focused, rolling submission, and the review bar rewards thorough evaluation over novelty-for-novelty's sake. Your architecture is solid; the paper just needs honest results.

### Phase 1 — Fix the Numbers (1–3 days)
*(Your infrastructure is ready; cost is the main constraint)*

- [ ] **Rerun Nut Assembly with correct model** — 50–100 episodes (currently: 25 with wrong model)
- [ ] **Rerun CALVIN** — at least 100 sequences (currently: 20)
- [ ] Rerun key baselines under same conditions if any were also underrepresented
- [ ] Apply for **Anthropic/OpenAI/Google research API credits** to offset VLM call costs

> [!TIP]
> A few hours to a day for all baselines per your estimate. The CALVIN cost is mostly compute, not API spend. Prioritize Nut Assembly rerun first since that's the direct contradiction.

### Phase 2 — Add One Ablation (1–2 days)

The single most impactful addition to defend novelty claims:

```
Table: V2A-WM Component Ablation (Nut Assembly SR%)
| Model Variant                     | SR (%) | NCR (%) |
|-----------------------------------|--------|---------|
| Base (RLA-WM)                     | XX     | XX      |
| + Cross-Attention Grounding       | XX     | XX      |
| + Temporal History                | XX     | XX      |
| + Sparsity Regularization (Full)  | XX     | XX      |
```

Even one row — e.g., removing temporal history only — is better than nothing and directly answers Reviewer Q6.

### Phase 3 — Rewrite These Sections (1–2 days)

1. **Abstract** — rewrite to match actual results; make claim precise ("outperform latent-space baselines on CALVIN" instead of blanket "significantly outperform SOTA")
2. **Experiment section** — rewrite yourself, do not delegate to an LLM for the results narrative
3. **Add**: training data details (# demonstrations, sources) — answers Reviewer Q1
4. **Add**: threshold sensitivity discussion (θ_conf, θ_c, θ_p) — answers Reviewer Q2
5. **Add**: Euler step justification (single-step was chosen for latency; discuss multi-step as future work) — answers Reviewer Q3
6. **Add**: wall-clock latency vs. ReflectVLM and DINO-WM — answers Reviewer Q4
7. **Add**: failure mode analysis for Nut Assembly (pruning too aggressive? threshold tuning?) — answers Reviewer Q5
8. **Scope statement in Related Work**: explicitly acknowledge why GR-1/end-to-end policies weren't compared (different problem setting, primitive action assumption)

### Phase 4 — Optional but High-Value

- Discuss GHIL-Glue and LUMOS in related work (at minimum a paragraph each; comparisons optional)
- SR@N metrics for CALVIN (chain-length distribution) — much more informative than SR@1 alone
- Calibration curves for critic thresholds

---

## Revised Claims That Will Hold Up

| Current Abstract Claim | Replace With |
|---|---|
| "significantly outperform state-of-the-art" | "outperform latent-space baselines (DINO-WM, RLA-WM) on CALVIN and achieve competitive performance on Nut Assembly while providing higher plan efficiency via early pruning (NCR)" |
| Implied: best on both benchmarks | Honest: best on CALVIN; competitive on Nut Assembly with better NCR |

---

## Timeline Estimate

| Phase | Time | Priority |
|---|---|---|
| Nut Assembly + CALVIN reruns | 1–2 days | 🔴 Critical |
| Ablation (1 row) | 1 day | 🟠 High |
| Rewrite abstract + experiments | 1–2 days | 🔴 Critical |
| Technical clarifications (Euler, losses, data) | 0.5–1 day | 🟡 Medium |
| Latency/compute table | 0.5 day | 🟡 Medium |

**Total: ~2–3 focused weeks for a substantially stronger RA-L submission.**

---

## The Bigger Picture

The underlying work is sound. The architecture is well-motivated, the critic design is concrete, and the CALVIN results are competitive. The problems in the submitted version are:
1. Rushed execution under impossible constraints (2 weeks, no advisor input, out-of-pocket API costs)
2. Over-reliance on an LLM to narrate results it couldn't verify

Neither of those is a research flaw. A proper RA-L submission gives this work a fair shot.

> [!NOTE]
> On the advisor situation: you are finishing this PhD largely alone, in your final semester, with a committee that isn't meaningfully accessible. If there is any graduate ombudsperson, graduate chair, or department director of graduate studies you can approach confidentially, it may be worth documenting the pattern — not for confrontation, but as a safeguard for your own protection through the dissertation phase.
