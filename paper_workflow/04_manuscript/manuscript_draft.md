# Cost-derived visual frost states for policy-conditional heat-pump defrost timing

## Abstract

Defrost initiation in air-source heat pumps is an economic decision: continued heating avoids a defrost interruption but progressively loses useful heat as frost accumulates. Historical records, however, contain only the action selected by the installed controller and provide no decision label for the preceding frost images. Here we introduce a retrospective framework that converts unsmoothed, approximately 1-s water-side and electrical measurements into a policy-conditional renewal-cost surface and maps its pointwise relative regret to timestamped RGB images. Among 77 catalogued cycles, 47 had complete candidate domains and 37 of these exhibited an interior empirical minimum. The median minimum preceded the observed defrost action by 58.8 min, but the median envelope spanning candidates within 1% of the minimum was 33.0 min, showing that a precise timestamp was often weakly identified. The resulting image dataset comprised 105,178 metadata rows, including 57,213 images within cost-valid candidate domains. Five-model development favoured a high-confidence binary task that excludes low-regret images over direct pre/near/post classification. In developmental leave-one-experiment-out evaluation of 5,289 sampled images from 45 cycles and 11 experiments, all-view RGB achieved a balanced accuracy of 0.951, but its paired increment over a retrospective time-only control was uncertain (0.024, 95% interval −0.012 to 0.069). A front view showed an exploratory increment of 0.038 (0.006 to 0.081) and reduced error-associated regret by 0.00113 (0.00005 to 0.00277). These results establish an auditable bridge from system cost to ambiguity-aware visual supervision and show that incremental visual information is camera dependent. They do not establish a causal global optimum, prospective energy savings, occupant comfort improvement or online deployment.

## Introduction

Frost accumulation on the outdoor heat exchanger of an air-source heat pump restricts airflow and heat transfer, while defrosting interrupts useful heating and consumes electricity. The controller must therefore decide when the cost of continuing a frosting cycle exceeds the cost of interrupting it. Previous studies have formulated energy-loss criteria for an optimal defrost initiation point, implemented model-based initiation methods, tracked frost accumulation and quantified the improvement potential of common controllers [1–3,6]. These studies establish that defrost timing can be treated as an optimization problem rather than a fixed-time maintenance action.

Visual sensing provides a complementary route because frost is directly observable. Image recognition and image-grey methods have already been connected to on-demand defrost control and heating-performance evaluation [4,5]. The unresolved problem is therefore not whether images can detect frost, nor whether an energy-based timing objective can be defined. It is how to construct a defensible decision label for every historical image when only one defrost action was observed and the objective may remain nearly flat over a long part of a cycle.

A single mathematical argmin is an inadequate label when several candidate times have nearly equivalent cost. It assigns opposite classes to visually similar images on either side of an arbitrary timestamp and conceals cycles in which low-regret candidates are broad or disconnected. Repeated images create a second problem: adjacent frames, camera appearance and experimental conditions are strongly correlated, so frame-random partitioning can produce severe information leakage [7]. Finally, even a correctly grouped image model may exploit elapsed time or systematic illumination drift rather than visible frost. A visual claim therefore requires a matched time-only control.

We address these problems with a cost-to-image supervision framework. First, unsmoothed operational measurements define a cycle-specific empirical renewal-cost surface under the observed fixed-duration defrost policy. Second, the relative regret of every candidate is joined to timestamped images, retaining a low-regret region rather than forcing a unique label. Third, whole experiments are held out during evaluation and identical folds compare RGB features, a retrospective time-only control and their combination. The contribution is this auditable translation and validation contract, not a first optimal-defrost or first image-defrost method. The study is retrospective; prospective interventions are required before energy or comfort benefits can be claimed.

## Results

### Unsmoothed measurements yielded auditable empirical cost surfaces

We retained the original approximately 1-s observations for all cost integrations. Water-side heating capacity was calculated as

\[
Q_h(t)=1.161\,\dot V_w(t)\,[T_{out}(t)-T_{in}(t)],
\]

where water flow is in m³ h⁻¹, temperature difference is in K and capacity is in kW. A 60-s median was used only to estimate stable clean anchors; no rolling, low-pass, wavelet or monotonic smoother was applied to the integrated signal. Across 74 valid anchors, the median clean coefficient of performance was 2.487, yielding an equivalent thermal-service conversion coefficient \(\lambda_Q=0.402\).

Defrost and recovery crossed cycle-file boundaries, so temporally adjacent files from the same experiment were linked before integration. Fifty observed events met the ticket contract. Their mean equivalent cost was 1.018 kWh-eq. and their mean duration was 13.60 min. The ticket combined measured electrical use with a thermal-service shortfall; the latter is not a measurement of PMV, PPD or occupant comfort.

For candidate defrost time \(\tau\), the heating-stage cost was

\[
C_H(\tau)=\int_{t_0}^{\tau}\left(P_{el}(t)+\lambda_Q[Q_{ref}(t)-Q_h(t)]_+\right)dt,
\]

and the renewal-average objective was

\[
\rho_i(\tau)=\frac{C_{H,i}(\tau)+\bar K_D}{T_{H,i}(\tau)+\bar T_D}.
\]

Of 77 catalogued cycles, 47 retained a complete candidate domain from 10 min after stable heating to the observed defrost boundary. Thirty-seven of these had an interior minimum and 10 were minimized at the observed right boundary; none was minimized at the left boundary. The empirical minimum preceded the observed action by a median of 58.8 min (Fig. 1). This difference is descriptive and policy-conditional because the historical record does not reveal the defrost and recovery ticket that would have followed every earlier candidate action.

### Cost valleys replaced a universal timing window with pointwise regret

The relative regret of candidate \(\tau\) was defined as

\[
r_i(\tau)=\frac{\rho_i(\tau)}{\rho_i(t_i^*)}-1.
\]

Near-optimal timing was not represented by one fixed number of minutes. Median envelopes from the earliest to latest candidates satisfying the threshold were 33.0, 52.0, 95.0 and 102.8 min at 1%, 2%, 5% and 10% regret, respectively. Some low-regret sets were disconnected, so all downstream labels used pointwise regret rather than filling the full earliest-to-latest envelope (Fig. 2).

Ticket and reference sensitivity identified a stable majority and a sensitive minority. Replacing the mean defrost ticket with its median left the optimum unchanged in 26 of 47 cycles and shifted it by no more than 5 min in 83.0%, although the 90th percentile absolute shift was 36.2 min. Replacing two-anchor interpolation with the current clean anchor alone produced a median absolute shift of 1.0 min and kept 76.6% of cycles within 5 min, but 12.8% shifted by more than 30 min. These results preclude treating every point estimate as equally sharp.

### Regret generated cycle-safe RGB states

Timestamp interpolation joined the candidate curves to 105,178 image metadata rows. Images outside a valid candidate domain were excluded, leaving 57,213 eligible images across 45 cost-valid cycles. At each audited threshold, an image was `near-optimal` only when its own interpolated regret did not exceed the threshold. Remaining images were labelled `pre-optimal` or `post-optimal` relative to the earliest argmin. Whole experiments, rather than frames, defined data partitions, producing 7/2/2 train/validation/test experiments containing 25/8/12 cycles.

Five development models—colour logistic regression, colour random forest, colour RBF support-vector machine, a small convolutional neural network and a pretrained ResNet18 linear probe—were compared on common sampled images. The strongest three-class validation result was a balanced accuracy of 0.617, whereas excluding the 1% low-regret region yielded 0.816 for the colour RBF-SVM in the high-confidence binary task. These numbers selected the compact model family and task for the streamed cohort; they are development results, not publication-level performance estimates.

### RGB was tested against retrospective cycle time in unseen experiments

The completed cohort contained 5,289 sampled images from 45 cycles and 11 experiments. In developmental leave-one-experiment-out evaluation, the pre-locked all-view analysis gave experiment-macro balanced accuracies of 0.951 (95% interval 0.930–0.971) for RGB, 0.926 (0.885–0.963) for retrospective time alone and 0.954 (0.933–0.974) for RGB plus time (Fig. 3). Corresponding RGB macro-F1, AUROC and class-balanced misclassification regret were 0.939 (0.912–0.965), 0.963 (0.941–0.984) and 0.00097 (0.00047–0.00153). The paired RGB-minus-time balanced-accuracy difference was 0.024 (−0.012 to 0.069), and the RGB-plus-time-minus-time difference was 0.028 (−0.004 to 0.068). Their regret differences were −0.00081 (−0.00250 to 0.00043) and −0.00090 (−0.00249 to 0.00020), respectively. Thus, the pre-locked pooled-camera analysis did not establish a stable visual increment beyond temporal progression.

The result depended on camera position. The front view gave RGB, time-only and RGB-plus-time balanced accuracies of 0.965, 0.927 and 0.966, respectively. Its paired RGB-minus-time difference was 0.038 (0.006–0.081), accompanied by a regret difference of −0.00113 (−0.00277 to −0.00005). RGB plus time also exceeded time alone for the pooled top pair by 0.034 (0.005–0.075), although RGB alone did not (0.023, −0.003 to 0.054). These camera-specific comparisons were observed after inspection of the complete cohort and are exploratory, not pre-specified confirmatory findings. Simply pooling every camera did not outperform the strongest single view.

The 1% threshold retained 65.6% of eligible images and all 11 experiments. Raising the threshold to 2%, 5% and 10% reduced absolute coverage to 48.6%, 19.8% and 8.6%, while the number of experiments supporting balanced accuracy fell from 11 to 11, 7 and 2. The apparent increase in accuracy at the strictest thresholds therefore reflects an easier and progressively narrower task and cannot justify replacing the pre-locked 1% analysis.

The normalized candidate-progress feature uses the future observed cycle boundary and is therefore a strong retrospective diagnostic rather than a deployable online feature. Its purpose is to test whether cost-state labels can be reconstructed from temporal order alone. Camera groups pool images from the named views; they do not perform synchronous multi-view fusion.

### Experiment-level failures bounded the visual claim

Across held-out experiments, all-view RGB balanced accuracy ranged from 0.883 to 0.997 and class-balanced error regret from 0.00004 to 0.00298 (Fig. 4). Of 5,289 high-confidence images, 388 were misclassified. Errors concentrated near the exclusion boundary: 283 of 1,781 images with 1–2% regret were misclassified (15.9%), compared with 97 of 2,242 at 2–5% regret (4.3%) and 8 of 1,266 above 5% regret (0.6%). Nevertheless, cycle-level auditing identified concentrated failures: cycle 68 had the largest mean misclassification regret (2.65%) and a 73.6% error rate, followed by cycles 61 and 76. These failures show why frame-pooled accuracy alone is insufficient. All uncertainty intervals were obtained by resampling experiments, not frames. Because historical test outputs were inspected during development, the complete leave-one-experiment-out analysis is developmental cross-validation rather than a blind test.

## Discussion

This study connects two established lines of defrost research: system-level timing objectives and visual frost sensing. Its central advance is an explicit transformation from raw operational measurements to a regret-valued label for every image. The transformation preserves uncertainty that a single optimal timestamp would discard and makes the assumptions behind the visual target inspectable.

The broad low-regret regions alter how “optimal defrost time” should be interpreted. A numerical argmin exists in each complete cycle, but several candidate actions can be practically equivalent under the present objective. Selective classification provides a natural learning analogue: images inside the low-regret region can be withheld while the model learns the more defensible pre/post states [8,9]. This abstention is not missing data; it is a statement that the cost model does not support a sharp class there.

Experiment-level isolation and the time-only control are equally important. Adjacent images are not independent observations, and a high frame-level score can reflect experiment identity, camera exposure or elapsed time. The all-view model classified states accurately, but its paired increment over time alone remained compatible with no increment. The front camera provided the only RGB-alone comparison in which both balanced-accuracy gain and regret reduction excluded zero; because this view was selected after complete-cohort inspection, it is a hypothesis for a prospective camera-design study rather than confirmation that RGB generally adds decision value. The result also shows that additional views can dilute rather than strengthen the useful signal.

Four limitations set the current boundary. First, earlier defrost candidates are counterfactual: their heating-stage history is observed, but their subsequent defrost and recovery ticket is replaced by a cohort estimate. The resulting optimum is conditional on that fixed observed policy. Second, the reference uses a future clean anchor in the primary offline reconstruction, which is unavailable at decision time and materially affects a minority of cycles. Third, the thermal-service term is not direct thermal-comfort measurement. Fourth, all visual results are retrospective and the historical test subset was inspected during method development.

The decisive next experiment is therefore prospective rather than a more complex image network. Selected cycles should be randomized or otherwise assigned to several candidate initiation regions inside and outside the estimated low-regret set, with direct measurements of defrost electricity, recovery, delivered heat and indoor conditions. Such a study would test whether the empirical surface transports to changed actions and whether image-triggered decisions improve outcomes.

## Methods

### Study object and cohort

The unit of analysis was one heating-to-defrost cycle. All 77 catalogued cycles remained in the audit table. A cycle received an optimum only when catalog status, stable-heating and observed-defrost boundaries, clean anchors and at least 95% raw-data coverage of the candidate domain were valid. Twelve cycles lacked an observed defrost boundary, nine were catalogued as invalid, five contained long gaps that truncated the candidate domain and four lacked a valid clean anchor.

### Raw integration and clean reference

Invalid raw observations were removed before trapezoidal integration. Adjacent valid intervals of at most 5 s were accepted; longer gaps were not bridged. Coverage was the accepted duration divided by the full candidate duration. The primary clean reference linearly joined stable 60-s anchors from the current and following cycles. Pointwise thermal-service shortfall was \([Q_{ref}(t)-Q_h(t)]_+\), converted to electrical-equivalent power using the reciprocal of the cohort median clean COP.

### Observed defrost ticket and candidate optimization

Each ticket began at observed defrost start and ended when heating capacity remained above 90% of the following clean anchor for 30 s. Electrical energy, equivalent service shortfall and elapsed duration were integrated over this window. Primary candidate curves used the cohort arithmetic mean cost \(\bar K_D\) and duration \(\bar T_D\); median-ticket and current-anchor-only alternatives were retained as sensitivity analyses. Candidates were evaluated at 1-min spacing from 10 min after stable heating to the observed boundary, which was always included. The earliest minimizing candidate defined \(t_i^*\).

### Image labels and partitions

Image timestamps were joined to candidate regret by within-cycle interpolation. Images outside the candidate domain or without valid regret were excluded. Thresholds of 1%, 2%, 5% and 10% were audited. The locked high-confidence binary task removed images at or below 1% regret and mapped remaining pre/post states to 0/1. All images from one experiment remained in one partition. The original fixed partition was used for model development; the final developmental estimate used leave-one-experiment-out evaluation across completed experiments.

### Streamed image processing and models

Cloud image archives were materialized one cycle at a time. The exact cycle archive was copied to a temporary local directory, paths and image decoding were checked, and at most 12 images were sampled uniformly for each cost-state × camera-role group. Forty compact colour, gradient and camera-role features were stored as a cycle-level Parquet shard. The temporary archive and extracted directory were then removed locally; cloud objects were neither modified nor deleted.

The five development models were logistic regression, random forest, radial-basis SVM, a small CNN and a pretrained ResNet18 linear probe. After task and family selection, the RBF-SVM used standardized features, \(C=2\) and balanced class weights. Camera analyses comprised each single view, pooled top/top-close images, pooled left/left-close images and all pooled camera images. Pooled groups were not synchronized image fusion.

### Temporal control and statistical analysis

The retrospective time-only control used elapsed minutes from the candidate start and normalized progress from candidate start to the observed candidate boundary. RGB-plus-time appended these two variables to the 40 RGB features. All modalities used identical held-out experiments and the same SVM specification.

Balanced accuracy, macro-F1 and AUROC were calculated within each held-out experiment. Class-balanced misclassification regret was

\[
L_r=\frac{1}{2}\sum_{y\in\{0,1\}}\frac{1}{n_y}\sum_{j:y_j=y}r_j\,\mathbf{1}(\hat y_j\ne y_j),
\]

so errors far from the empirical optimum contributed more than errors close to the excluded boundary. Primary summaries macro-averaged experiments. Percentile 95% intervals resampled experiments with replacement 5,000 times using a fixed seed. Modality differences were paired by held-out experiment.

## Data availability

Source tables supporting the figures, the 77-cycle cohort audit, candidate cost curves, ticket summaries, image-label metadata and experiment-level model outputs will be deposited with the manuscript subject to the existing data-sharing permissions. Raw high-volume RGB archives remain in the authorized cloud store and are processed cycle by cycle. The final repository record and accession or persistent link must be inserted before submission.

## Code availability

Analysis code for raw cost reconstruction, regret labelling, streamed feature extraction, experiment-held-out evaluation and publication figures is version controlled in the project repository. A release tag, environment specification and archival DOI must be added before submission.

## References

1. Wang, Z. et al. Determination of the optimal defrosting initiating time point for an ASHP unit based on the minimum loss coefficient in the nominal output heating energy. *Energy* (2020). https://doi.org/10.1016/j.energy.2019.116505
2. Li, Y. et al. A novel defrosting initiating method for air source heat pumps based on the optimal defrosting initiating time point. *Energy Build.* (2020). https://doi.org/10.1016/j.enbuild.2020.110064
3. Chung, Y. et al. A determination method of defrosting start time with frost accumulation amount tracking in air source heat pump systems. *Appl. Therm. Eng.* (2021). https://doi.org/10.1016/j.applthermaleng.2020.116405
4. Chen, X. et al. Deep learning-based image recognition method for on-demand defrosting control to save energy in commercial energy systems. *Appl. Energy* (2022). https://doi.org/10.1016/j.apenergy.2022.119702
5. Wang, Z. et al. Space heating performance analysis of air source heat pump integrated with image gray recognition-based defrosting controller. *Appl. Therm. Eng.* (2024). https://doi.org/10.1016/j.applthermaleng.2023.121536
6. Klingebiel, J. et al. Optimal defrost initiation for air-source heat pumps: Evaluating the improvement potential of common defrosting controllers. *Energy* (2025). https://doi.org/10.1016/j.energy.2025.135871
7. Roberts, M. et al. Common pitfalls and recommendations for using machine learning to detect and prognosticate for COVID-19 using chest radiographs and CT scans. *Nat. Mach. Intell.* (2021). https://doi.org/10.1038/s42256-021-00307-0
8. El-Yaniv, R. & Wiener, Y. On the foundations of noise-free selective classification. *J. Mach. Learn. Res.* (2010). https://www.jmlr.org/papers/v11/el-yaniv10a.html
9. Franc, V., Prusa, D. & Voracek, V. Optimal strategies for reject option classifiers. *J. Mach. Learn. Res.* (2023). https://www.jmlr.org/papers/v24/21-0048.html

## Draft integrity notes

- References 1–9 have verified metadata or official abstract records; detailed comparative wording still requires full-text audit before submission.
- Complete-cohort values are drawn from the committed experiment-level summaries and paired bootstrap outputs; the camera-specific findings remain exploratory.
- The title and claims deliberately avoid “first”, “causal optimum”, “energy saving”, “comfort improvement” and “online control”.
