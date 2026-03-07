from dataclasses import dataclass


@dataclass
class CriticDecision:
    action: str
    reason: str


def decide_replan(
    mean_feasibility: float,
    uncertainty: float,
    feasibility_threshold: float,
    uncertainty_threshold: float,
) -> CriticDecision:
    """
    3-way control from Section 3:
    - continue
    - reflect (confident infeasible)
    - requery (low-feasibility but high-uncertainty)
    """
    if mean_feasibility < feasibility_threshold and uncertainty < uncertainty_threshold:
        return CriticDecision(action="reflect", reason="confident_infeasible")
    if mean_feasibility < feasibility_threshold and uncertainty >= uncertainty_threshold:
        return CriticDecision(action="requery", reason="low_feasibility_high_uncertainty")
    return CriticDecision(action="continue", reason="feasible_or_not_confident_failure")
