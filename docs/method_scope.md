# Method and inference scope

The repository distinguishes three tasks:

1. **Plausibility screening:** in a real application, candidate conceptual models should be tested against the same observations and weights. Cases with unacceptable residual structure, implausible parameter compensation, numerical failure, or contradiction with established evidence should be rejected.
2. **Forecast-directed deterministic screening:** evidence-consistent hypotheses are tested against decision quantities. Cases that duplicate an existing management regime or have negligible forecast influence are not escalated.
3. **Within-scenario stochastic refinement:** the retained conceptual cases are refined using a common pilot-point architecture.

The present benchmark implements the third task as a retrospective perfect-model experiment. Therefore, scenario-specific Phi values are diagnostic within a scenario only and cannot be used as AIC, BIC, KIC, likelihood evidence, or posterior scenario probabilities.

The saved-ensemble sensitivity analysis uses Spearman rank associations. It is an exploratory diagnostic, not a formal derivative-based PREDVAR/PREDUNC analysis or an observation-worth calculation.
