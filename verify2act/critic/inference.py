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
    confidence_threshold: float = 0.02,
) -> CriticDecision:
    """Head 2 gate — called *per imagined frame*.

    A low inter-frame cosine similarity signals a diffusion artifact (the world
    model hallucinated an implausible transition).  Abort the rollout early and
    request a new sample from the world model.

    Uncertainty is checked *first*: if the critic cannot confidently estimate
    the consistency score, the score itself is untrustworthy regardless of
    whether it clears the threshold, so we requery immediately.

    Decision logic
    --------------
    * uncertainty     >= confidence_threshold                  → requery  (unreliable estimate)
    * uncertainty     <  confidence_threshold AND
      consistency     <  threshold                            → requery  (confident incoherence)
    * otherwise                                               → continue

    Parameters
    ----------
    consistency_score : float
        Mean cosine_sim(head2(e_t), head2(e_{t+1})) from MC sampling ∈ [-1, 1].
    threshold : float
        theta_c — frames below this are considered incoherent.
    uncertainty : float
        Predictive std of the MC similarity estimate (from
        ``temporal_sim_with_uncertainty()``).
    confidence_threshold : float
        Maximum allowed uncertainty to trust the consistency score.
        Default 0.02 — calibrate from validation uncertainty histograms.

    Returns
    -------
    CriticDecision  with action ``"requery"`` or ``"continue"``
    """
    if uncertainty >= confidence_threshold:
        return CriticDecision(
            action="requery",
            reason=f"uncertain_consistency_estimate unc={uncertainty:.3f}",
        )
    if consistency_score < threshold:
        return CriticDecision(
            action="requery",
            reason=f"low_temporal_consistency score={consistency_score:.3f}",
        )
    return CriticDecision(action="continue", reason="consistent")


def decide_from_proximity(
    proximity_score: float,
    threshold: float,
    uncertainty: float = 0.0,
    confidence_threshold: float = 0.02,
) -> CriticDecision:
    """Head 1 gate — called once after a *complete* imagined rollout passes.

    A low cosine similarity between the final imagined frame and the goal image
    means the plan is unlikely to succeed.  However, if the critic is uncertain
    about this estimate (high predictive std), avoid immediately triggering a
    full reflection — the world model may have produced a hallucinated state
    that looks unlike any training distribution.

    Decision logic
    --------------
    * uncertainty     >= confidence_threshold            → requery  (unreliable estimate)
    * uncertainty     <  confidence_threshold AND
      proximity_score >= threshold                       → continue
    * uncertainty     <  confidence_threshold AND
      proximity_score <  threshold                       → reflect  (confident failure)

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
        Default 0.02 — calibrate from validation uncertainty histograms.

    Returns
    -------
    CriticDecision  with action ``"reflect"``, ``"requery"``, or ``"continue"``
    """
    if uncertainty >= confidence_threshold:
        return CriticDecision(
            action="requery",
            reason=f"uncertain_proximity_estimate unc={uncertainty:.3f} prox={proximity_score:.3f}",
        )
    if proximity_score >= threshold:
        return CriticDecision(action="continue", reason="high_goal_proximity")
    return CriticDecision(
        action="reflect",
        reason=f"low_goal_proximity prox={proximity_score:.3f} unc={uncertainty:.3f}",
    )
