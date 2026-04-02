from dataclasses import dataclass


@dataclass
class CriticDecision:
    action: str
    reason: str


# ── DINOv2DualHeadCritic — sequential early-abort logic ──────────────────────

def check_rollout_consistency(
    consistency_score: float,
    threshold: float,
) -> CriticDecision:
    """Head 2 gate — called *per imagined frame*.

    A low inter-frame cosine similarity signals a diffusion artifact (the world
    model hallucinated an implausible transition).  Abort the rollout early and
    request a new sample from the world model.

    Parameters
    ----------
    consistency_score : float
        cosine_sim(head2(e_t), head2(e_{t+1}))  ∈ [-1, 1]
    threshold : float
        theta_c — frames below this are considered incoherent.

    Returns
    -------
    CriticDecision  with action ``"requery"`` or ``"continue"``
    """
    if consistency_score < threshold:
        return CriticDecision(action="requery", reason="low_temporal_consistency")
    return CriticDecision(action="continue", reason="consistent")


def decide_from_proximity(
    proximity_score: float,
    threshold: float,
) -> CriticDecision:
    """Head 1 gate — called once after a *complete* imagined rollout passes.

    A low cosine similarity between the final imagined frame and the goal image
    means the plan is unlikely to succeed even if it is physically plausible.
    Ask the VLM to reflect and revise.

    Parameters
    ----------
    proximity_score : float
        cosine_sim(head1(e_H), head1(e_goal))  ∈ [-1, 1]
    threshold : float
        theta_p — final frames below this are considered plan failures.

    Returns
    -------
    CriticDecision  with action ``"reflect"`` or ``"continue"``
    """
    if proximity_score < threshold:
        return CriticDecision(action="reflect", reason="low_goal_proximity")
    return CriticDecision(action="continue", reason="high_goal_proximity")
