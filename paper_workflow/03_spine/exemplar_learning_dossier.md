# Exemplar Learning Dossier

This analysis is limited to metadata, screened abstracts, and source notes recorded in the project. It extracts argument architecture rather than importing empirical claims or results.

## Exemplar Inventory

| Title | Venue | Year | Why selected |
|---|---|---:|---|
| *Determination of the optimal defrosting initiating time point for an ASHP unit based on the minimum loss coefficient in the nominal output heating energy* | *Energy* | 2020 | Closest precedent for framing defrost initiation as a cycle-level optimization problem. |
| *A novel defrosting initiating method for air source heat pumps based on the optimal defrosting initiating time point* | *Energy and Buildings* | 2020 | Connects an estimated optimum to a prospective initiating method. |
| *Deep learning-based image recognition method for on-demand defrosting control to save energy in commercial energy systems* | *Applied Energy* | 2022 | Representative bridge from visual recognition to demand-based control. |
| *Optimal defrost initiation for air-source heat pumps: Evaluating the improvement potential of common defrosting controllers* | *Energy* | 2025 | Recent controller-comparison framing centred on timing sensitivity and improvement potential. |
| *Common pitfalls and recommendations for using machine learning to detect and prognosticate for COVID-19 using chest radiographs and CT scans* | *Nature Machine Intelligence* | 2021 | Exemplar for a warning-and-recommendation argument around grouped images and uncertainty. |
| *Optimal Strategies for Reject Option Classifiers* | *Journal of Machine Learning Research* | 2023 | Formal exemplar for treating abstention and coverage as part of the prediction problem. |

## Structural Patterns

The heat-pump exemplars form a recurring problem-to-decision sequence. Wang et al. (2020) place an explicit optimization target in the title, making the decision variable and loss criterion visible immediately. Li et al. (2020) then exemplify the next translational move: an optimum is not the endpoint but the basis of an initiating method. Chen et al. (2022) use a sensing-to-control bridge, coupling an image-recognition method to on-demand operation and an energy objective. Klingebiel et al. (2025) broaden the arc from proposing a method to benchmarking existing controllers against improvement potential.

The validation exemplars supply a complementary evidence contract. Roberts et al. (2021) organize the contribution around failure modes followed by recommendations, a reusable structure for explaining why experiment-level separation and explicit uncertainty are necessary rather than decorative. Franc et al. (2023) elevate rejection from an afterthought to a formal strategy, suggesting that ambiguous samples, retained coverage, and predictive risk should be presented together.

Combined, these patterns support a four-move paper spine: define a policy-conditional cycle cost; convert its surface into pointwise regret states; test whether RGB images identify those states under experiment-held-out evaluation; and report abstention or ambiguity as a measured operating trade-off. This structure also keeps retrospective label construction separate from any prospective control or savings claim.

## Rhetorical Patterns

The strongest openings name a practical decision, then expose the inadequacy of the current decision signal. They narrow quickly to a measurable object—initiation time, controller gap, grouped-image evaluation, or risk–coverage—rather than opening with generic claims about artificial intelligence. Closings should mirror that narrowing: state what the proposed supervision-and-validation bridge establishes, then explicitly withhold claims of global optimality, deployable online control, comfort improvement, or prospective savings.

## Language Patterns

The register is technical, objective-led, and scope-conscious. Effective wording pairs concrete nouns with operational verbs: “determine initiation time,” “track accumulation,” “evaluate improvement potential,” and “classify with rejection.” For this paper, prefer “policy-conditional,” “cycle-specific,” “experiment-held-out,” “relative regret,” and “high-confidence coverage.” Reserve “optimal” for the defined empirical objective, and avoid unsupported priority claims such as “first,” “novel,” or “state of the art.”
