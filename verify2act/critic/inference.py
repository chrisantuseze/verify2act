from dataclasses import dataclass


@dataclass
class CriticDecision:
    action: str
    reason: str


# ── DINOv2DualHeadCritic — sequential early-abort logic ──────────────────────

def check_rollout_consistency(
    consistency_score: float,
    threshold: float,
    uncertainty: float = 0.0,
) -> CriticDecision:
    """Head 2 gate — called *per imagined frame*.

    A low inter-frame cosine similarity signals a diffusion artifact (the world
    model hallucinated an implausible transition).  Abort the rollout early and
    request a new sample from the world model.

    Uncertainty (predictive std from MC sampling) is logged in the reason string
    but does not change the requery decision for temporal consistency — even an
    uncertain low-consistency score warrants a fresh world-model sample.

    Parameters
    ----------
    consistency_score : float
        Mean cosine_sim(head2(e_t), head2(e_{t+1})) from MC sampling ∈ [-1, 1].
    threshold : float
        theta_c — frames below this are considered incoherent.
    uncertainty : float
        Predictive std of the MC similarity estimate (from
        ``temporal_sim_with_uncertainty()``).  Reported in reason string.

    Returns
    -------
    CriticDecision  with action ``"requery"`` or ``"continue"``
    """
    if consistency_score < threshold:
        unc_tag = f"unc={uncertainty:.3f}" if uncertainty > 0 else ""
        reason = f"low_temporal_consistency {unc_tag}".strip()
        return CriticDecision(action="requery", reason=reason)
    return CriticDecision(action="continue", reason="consistent")


def decide_from_proximity(
    proximity_score: float,
    threshold: float,
    uncertainty: float = 0.0,
    confidence_threshold: float = 0.15,
) -> CriticDecision:
    """Head 1 gate — called once after a *complete* imagined rollout passes.

    A low cosine similarity between the final imagined frame and the goal image
    means the plan is unlikely to succeed.  However, if the critic is uncertain
    about this estimate (high predictive std), avoid immediately triggering a
    full reflection — the world model may have produced a hallucinated state
    that looks unlike any training distribution.

    Decision logic
    --------------
    * proximity_score >= threshold                       → continue
    * proximity_score <  threshold AND
      uncertainty     <  confidence_threshold            → reflect  (confident failure)
    * proximity_score <  threshold AND
      uncertainty     >= confidence_threshold            → requery  (uncertain; get new sample)

    Parameters
    ----------
    proximity_score : float
        Mean cosine_sim(head1(e_H), head1(e_goal)) from MC sampling ∈ [-1, 1].
    threshold : float
        theta_p — final frames below this are considered plan failures.
    uncertainty : float
        Predictive std from ``goal_sim_with_uncertainty()``.
    confidence_threshold : float
        Maximum allowed uncertainty to trigger a reflect decision.
        Default 0.15 — calibrate from validation uncertainty histograms.

    Returns
    -------
    CriticDecision  with action ``"reflect"``, ``"requery"``, or ``"continue"``
    """
    if proximity_score >= threshold:
        return CriticDecision(action="continue", reason="high_goal_proximity")
    # Below threshold: check whether the critic is confident
    if uncertainty >= confidence_threshold:
        return CriticDecision(
            action="requery",
            reason=f"low_proximity_but_uncertain prox={proximity_score:.3f} unc={uncertainty:.3f}",
        )
    return CriticDecision(
        action="reflect",
        reason=f"low_goal_proximity prox={proximity_score:.3f} unc={uncertainty:.3f}",
    )
