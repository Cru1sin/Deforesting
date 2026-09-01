# Motivation spine

## Problem

Air-source heat-pump controllers must decide when accumulated frost has become costly enough to justify defrosting, yet historical controllers expose only the action they actually took rather than a decision label for every preceding frost image.

## Consequence

Image models trained on elapsed time, heuristic frost grades or the observed defrost event can reproduce controller habits or temporal drift without learning whether the visible frost state is economically consequential.

## Knowledge gap

Energy-based timing objectives and image-based frost recognition have both been studied, but the available evidence does not provide an auditable, uncertainty-aware mapping from a cycle-specific cost surface to every timestamped RGB frame, followed by a matched test of visual information beyond time alone.

## Why existing work is insufficient

A single argmin creates contradictory hard labels across broad or disconnected low-regret regions. Frame-random splits inflate performance because adjacent images and illumination conditions recur within experiments. High RGB accuracy alone also cannot exclude a cycle-time shortcut.

## Approach

Integrate unsmoothed approximately 1-s water-side and electrical measurements under an explicit fixed-policy renewal-cost objective; export candidate-level relative regret; map images to pre-, near- or post-optimal states; exclude near-optimal frames for the locked high-confidence binary task; and compare RGB, retrospective time-only and RGB-plus-time models in identical leave-one-experiment-out folds.

## Evidence

The cost stage audits 77 catalogued cycles, identifies 47 complete candidate domains and 37 interior empirical minima, and quantifies broad low-regret regions. The image stage contains 105,178 metadata rows and 57,213 candidate-domain images across 45 cost-valid cycles. Publication evidence will use the completed streamed cohort, paired experiment-level uncertainty and class-balanced misclassification regret.

## Contribution

An auditable bridge from policy-conditional system cost to ambiguity-aware visual supervision, together with a validation contract that distinguishes frost appearance from temporal shortcuts.

## Boundary conditions

The study is retrospective and associational. The empirical minimum is conditional on the observed fixed-duration ticket and offline reference construction. Thermal-service shortfall is not measured occupant comfort. No prospective energy saving, causal global optimum or online deployment claim is permitted without intervention data.
