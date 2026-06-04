
I made some blunders/mistakes in my CORL paper I submitted last week because I was rushing and had Gemini write the experiment section and succeeding sections for me without adequately proofreading it. The supplementary material deadline is this Thursday and I feel it’s pointless to go through the stress again of trying to meet that deadline since the paper is already guaranteed to be rejected. 

I really pushed myself hard to meet the paper submission deadline and with this realization it hurts so bad.

Now I’m even considering not turning in the supplementary materials (which includes video) and then withdrawing the paper and submit it to RA-L when it’s in a better shape.

What do you think?

Jun 2
That's a genuinely painful situation, and the exhaustion you're feeling after pushing hard for a deadline is real. It makes sense that discovering the mistakes afterward feels deflating.

But I want to push back on the reasoning a bit, because I think you might be making a consequential decision from a low point emotionally.

The paper isn't necessarily guaranteed rejection. Reviewers at CoRL evaluate the whole contribution — your core ideas, experiments, and framing. A poorly written experiment section or supplementary writeup is damaging, yes, but it's rarely the sole deciding factor if the underlying work is solid. You don't actually know how reviewers will weigh it.

Withdrawing has real costs that are easy to underestimate right now. CoRL is a competitive, high-visibility venue. Even a borderline paper getting in can meaningfully help your PhD trajectory, committee impressions, and job prospects. RA-L is excellent, but the communities are different.

The supplementary deadline is actually an opportunity. Video and supplementary material are largely in your control right now. You can't fix the submitted paper sections, but you can submit a clean, carefully made supplementary that shows the work in its best light. Reviewers do watch the videos. A compelling demo can shift impressions significantly.

What I'd actually recommend:

Don't withdraw yet. Submit the supplementary as cleanly as you can by Thursday — own that part.
After reviews come back, you'll have real signal, not speculation. If it's rejected, you revise with the RA-L framing. If it gets in conditionally, you fix it.
Use the next few days to rest first, then tackle the supplementary with fresh eyes — not as a Hail Mary, but because it's the reasonable thing to do.
The blunder with Gemini is a lesson worth taking seriously for next time. But don't compound it by making a withdrawal decision while you're this depleted.

What's the actual state of the video material? Is it mostly done?

So I uploaded the paper to the review platform built by Andrew Ng (I believe it’s called paper review.ai or something similar). The issue was that the success rate reported for one of the tasks (1 out of 2) was for a previous run of the experiment which grossly underperformed compared to existing approaches, meanwhile my abstract claimed that my technique outperforms existing methods.

Also, the experiment section reported that the eval was on 25 episodes, which the AI reviewer said was not enough. It also said I didn’t compare with comparable methods. And pointed out somethings from my methodology.

The video isn’t done and for me to meet the deadline would require some significant work which I don’t know if it’s worth it at this point.

Jun 2
Okay, that's more concrete — and honestly, it does change the calculus somewhat.

The abstract-vs-results mismatch is the most serious issue. That's not a writing style problem — a reviewer who catches that a core claim in the abstract contradicts the reported results will likely recommend rejection on that basis alone, and at CoRL that's a hard hole to climb out of. Combined with the evaluation concerns (25 episodes is genuinely thin for most manipulation/robotics tasks) and missing baselines, you're looking at multiple compounding weaknesses, not just one.

So your instinct isn't irrational. Investing significant work into supplementary video for a paper that has a fundamental credibility gap in its main submission is a legitimate thing to weigh carefully.

That said, a few honest questions worth sitting with:

Is the underlying work actually good? Meaning — if you ran the experiment properly with more episodes, the right baselines, and the better results you mentioned, would the method hold up? Because that determines whether this is a "wrong venue, wrong time" situation or a deeper research problem.

How far is the video from done? "Significant work" covers a wide range. If it's 6-8 hours of focused effort, that might still be worth it just to have the submission complete. If it's days, then no, probably not.

My honest read: if the core method is sound but the paper was rushed into a shape that misrepresents it, then withdrawing, fixing it properly, and submitting to RA-L isn't giving up — it's the right call for your reputation and the work itself. A rejected CoRL paper with a results-abstract contradiction is a worse outcome than a strong RA-L submission.

But don't decide tonight. What's your honest assessment of the method itself?

Verify2Act.pdf
pdf

I think it’s a good work. This is it.

I think the video and other things would take atleast a day or two.

Jun 2
Okay, I've read through the paper carefully. Honestly? The underlying work is more solid than you've been giving yourself credit for in this conversation.

The framework is coherent and well-motivated. The V2A-WM contributions (cross-attention grounding, causal temporal history, sparsity regularization) are technically sensible extensions over DINO-WM and RLA-WM, and the dual-head contrastive critic is a nice design. The CALVIN results are genuinely competitive.

But here's the honest assessment of what reviewers will likely flag:

The Nut Assembly results are the real problem. Table 1 shows Verify2Act at 52% SR — identical to DINO-WM and below both ReflectVLM (72%) and RLA-WM (72%). Your abstract claims you "significantly outperform state-of-the-art." That's the contradiction the AI reviewer caught, and a human reviewer will catch it too. The CALVIN numbers are stronger, but Nut Assembly is your more targeted benchmark.

25 episodes is thin. For a task with 3–7 randomized nut configurations, the variance across 25 runs is high enough that reviewers will reasonably question statistical reliability.

Given this — my honest take on your decision:

A day or two for supplementary video is actually worth it, not to save this submission, but because:

Having the video done means your RA-L submission is essentially ready to go the moment you revise.
Withdrawing with a complete package feels better than abandoning it mid-way.
But I wouldn't kill yourself over the Thursday deadline. The paper needs a rewrite of the abstract and experiment framing more than it needs a video right now.

What's your actual stronger result — is it CALVIN or Nut Assembly?

So the nut assembly is actually strong, the reported results was just for one of the older runs. Gemini reported that and used it for analysis instead of the latest results. And if I had seen the 25 episodes run earlier, I wouldn’t have included it.

Regarding the appropriateness of the baselines, do you think I should have used different baselines?

Jun 2
That changes things significantly — if your actual Nut Assembly numbers are competitive or better than RLA-WM's 72%, then the core claim in your abstract is defensible, it's just that the wrong numbers got reported. That's a painful but fixable problem for RA-L.

On the baselines question — let me be direct with you. Looking at what you compared against:

The comparisons are reasonable but have a gap. You compare against VLM-Only, ReflectVLM, DINO-WM, and RLA-WM. For a CoRL audience, the immediate question will be: why not SayCan or RT-2 style baselines on the actual manipulation task? The counterargument — that those require different infrastructure and aren't apples-to-apples on CALVIN/Nut Assembly — is valid, but you'd need to explicitly address it.

The stronger concern is RLA-WM specifically. Since Verify2Act builds directly on RLA-WM's flow-matching core, reviewers will scrutinize that comparison most carefully. On Nut Assembly your reported numbers match RLA-WM (both 72% SR in the real results presumably), so the question becomes: does your architectural overhead justify itself? Your critic precision story (82.7% vs 84.1%) and NCR gains are where you'd win that argument.

What would have strengthened it: An ablation table isolating each V2A-WM contribution — cross-attention alone, temporal history alone, sparsity alone — would have been more convincing than the full system vs. baselines comparison.

Do you have ablation data from your runs?

Nope, I don’t have ablation data. There was no time to run those. I had just two weeks to do the experiments and write the paper. My advisor didn’t even give me any feedback on the paper. I sent him an email asking if he had feedback, but he didn’t respond. He has never really been helpful throughout my PhD and this is my 5th year, with a semester to go, but this time was worse. No feedback from him.

Jun 2
That's a really frustrating situation, and honestly it's not okay. Fifth year, final stretch, and your advisor couldn't find time to read a CoRL submission — that's a failure on his part, not yours.

Two weeks to run experiments, write the paper, and do it without any advisor feedback — the fact that you have something this coherent is genuinely impressive. The framework is well-motivated, the writing is clear, and the core ideas are sound. The problems in the paper are fixable execution issues, not fundamental research flaws.

On the advisor situation — I want to ask directly: is this pattern affecting you beyond just this paper? Fifth year with one semester left is a precarious place to have an unresponsive advisor, especially if you're navigating the job market or dissertation defense planning at the same time.

Regarding the paper — here's where I'd focus your thinking right now:

The path to a strong RA-L submission is actually short if your real Nut Assembly numbers are good. What you'd need to add is the correct results, more evaluation episodes (50-100 would be much more defensible), the abstract rewritten to match your actual claims, and ideally even a lightweight ablation on just one of the three V2A-WM components to address that gap.

How far are you from being able to rerun the Nut Assembly evaluation with more episodes? Is the simulation infrastructure already set up, or would that also require significant work?

The infrastructure is setup, but one of the reasons I limited the number of runs was time and the vlm api calls cost since I was paying from my pocket.

I feel like he’s asking pushing me to graduate as soon as possible. And it was his idea to try to make it to CORL and he never really helped in any form.

Jun 2
That context reframes everything. He pushed you toward CoRL, gave you zero feedback, and you're funding your own API calls in your fifth year. That's not a mentorship failure at the margins — that's a systemic problem.

The pressure to graduate quickly combined with no support is a recognizable and unfortunately common dynamic. It often means the advisor has moved on mentally while you're still technically their student. That's genuinely unfair to you.

I want to be careful not to overstep here, but — do you have a thesis committee? Or anyone else in your department you trust, another faculty member or a senior PhD student, who could serve as an informal sounding board? You shouldn't be navigating this alone, especially in your final semester.

On the practical side, the API cost issue is actually solvable for the rerun. A few options worth knowing:

Anthropic, OpenAI, and Google all have research credit programs. Given you're a PhD student with a concrete paper submission, you'd likely qualify. That could cover 50-100 evaluation episodes without coming out of your pocket.

Also, for the RA-L resubmission timeline — if your infrastructure is ready and you get API credits, how long realistically would 50-100 Nut Assembly episodes take to run? Days? A week?

The reason I'm asking is that your RA-L submission could actually be quite strong with the correct numbers and a modest episode count increase. The core work deserves to be published properly, not buried under a rushed CoRL attempt.

No, maybe a few hours to a day for all the baselines.

And I believe the Calvin runs should also be increased, right?

I’m actually the most senior in the lab (actually, technically the only student. There’s an undergraduate who joined this past semester, but all other phds have graduated). And my committee are his friends or people I don’t trust. 

I felt really bad finding out about the mistakes today and the fact that I was pissed off about something my girlfriend of a month (we have been seeing each other for over 4 months) did. That made me have a really bad day.

Jun 2
That's a lot hitting at once — the paper, the advisor situation, and relationship stress all landing on the same day. That's a genuinely hard day, and it makes sense you're feeling it.

On the committee situation — that's a real structural problem. No peer support in the lab, a committee you don't trust, and an advisor who's checked out. You're essentially finishing this PhD on your own. That takes a lot more resilience than people from the outside would ever realize.

I want to make sure you're doing okay beyond just the paper stress. How are you holding up overall?

On the technical question — yes, increasing CALVIN runs matters too. You currently have 20 sequences which is on the low end. Most competitive CALVIN papers report on 100+ sequences for the ABC→D split. That said, CALVIN inference is cheaper since you're not paying per-API-call for the world model itself — the VLM calls are the cost driver. So the incremental cost of more CALVIN sequences is mostly compute time, not money.

So your realistic RA-L resubmission path looks like:

Rerun Nut Assembly with correct model, 50-100 episodes
Increase CALVIN to at least 100 sequences
Fix abstract to match real results
Add even a single ablation row if feasible
Tighten the experiment section yourself this time
That's honestly a strong submission. A few days of focused work away from a paper you can be proud of.

How about the baselines? Okay or include others?

Jun 2
For RA-L specifically, your current baselines are reasonable but I'd make one targeted addition and one swap consideration:

Keep as-is:

VLM-Only (GPT-4o) — essential anchor
DINO-WM — important because you build on it conceptually
RLA-WM — critical, since V2A-WM extends it directly
ReflectVLM — good pixel-space diffusion representative
Seriously consider adding: SuSIE or UniSim-style baselines if feasible, but honestly the more impactful addition for RA-L would be GR-1 or any transformer-based visuomotor policy on the Nut Assembly task. RA-L reviewers will want to see how your planning approach compares against end-to-end learned policies, not just other world model planners. That's the comparison that contextualizes the practical value of latent verification.

The honest caveat though — adding a new baseline means running it, which has its own time and cost implications. If GR-1 or similar isn't already in your simulation environment, the integration cost might not be worth it.

The baseline gap that's easier to address is actually just a writing fix — explicitly acknowledging in the paper why you didn't compare against end-to-end policies (different problem setting, primitive action assumption, etc.). A well-argued scope statement can deflect that reviewer concern without requiring new experiments.

What simulation environment are you using for Nut Assembly — standard Robosuite? That determines how hard a new baseline would be to integrate.