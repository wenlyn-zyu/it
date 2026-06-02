# Implementation Notes for `rd-semantic-repro`

## 1. Relation to the original paper
The paper models a source as a pair `(S, X)`:
- `S`: the intrinsic state, interpreted as the semantic content;
- `X`: the extrinsic observation, interpreted as the observable appearance.

The main object is the **state-observation rate-distortion function** (SORDF), which asks for the minimum rate required to jointly control:
- semantic distortion between `S` and `\hat S`;
- appearance distortion between `X` and `\hat X`.

Our implementation mirrors the paper section by section instead of jumping directly to a neural-network system. That choice is deliberate: this project is for an information theory course, and the numerical code should remain visibly tied to the mathematics.

## 2. Why we implement the binary case first
The binary classification case in the paper is the most educational numerical entry point.

Why:
1. It already captures the key idea of semantic communication: good task performance does not require perfect reconstruction of the observation.
2. The quantities are numerically manageable.
3. The figures are easy to interpret in a report.
4. The same code path can later support the multiclass extension.

In plain language: before handling large matrices or high-dimensional Gaussian decompositions, we want a simple example where the meaning of semantic distortion is unmistakable.

## 3. Why the code is split into `src/` and `experiments/`
We separate reusable mathematics from script-level experiments.

- `src/` contains reusable routines such as density evaluation, entropy terms, numerical integration, and plotting helpers.
- `experiments/` contains scripts that produce one figure family at a time.

This split is important for the report. It makes it easy to say:
- here is the mathematical routine we implemented;
- here is the exact experiment that generated the figure.

That transparency helps when writing the technical section of the EE142 report.

## 4. Why multiclass is the preferred extension
The original paper handles binary classification in detail. A natural next question is: what changes when semantic meaning is not binary?

The multiclass extension is attractive because:
- it preserves the intrinsic/extrinsic modeling philosophy;
- it remains close to classification tasks used in machine learning;
- it can produce new plots without changing the project into a completely different topic.

In other words, it is ambitious enough to count as a new idea, but conservative enough to stay faithful to the original paper.

## 5. Expected outputs for the report
This project should eventually provide:
- replicated curves from the binary classification case;
- replicated or matched trends for the Gaussian case;
- at least one new multiclass result;
- concise explanations that connect numerical behavior back to the SORDF formulation.

## 6. How this supports the course requirements
The EE142 guideline asks for:
- a professional English report;
- clear technical description;
- an original idea if possible.

This implementation supports those requirements as follows:
- reproduction gives technical credibility;
- section-by-section numerical reconstruction forces us to explain the proof ideas clearly;
- the multiclass extension gives us a modest but real innovation.


## 7. Why the first implemented behaviors are boundary cases
We started the code with two boundary behaviors and tested them first.

### Boundary 1: D_s >= 1/2 implies zero rate
In the binary model with equal priors, 1/2 corresponds to a trivial guesser. If the allowed semantic distortion is already this large, the encoder does not need to communicate anything to satisfy the semantic requirement. This is the cleanest sanity check for R(D_s, \infty).

### Boundary 2: D_s < Q(A/\sigma) is infeasible
The paper states that the binary case only makes sense for Q(A/\sigma) <= D_s <= 1/2. The lower endpoint is the Bayes classification error. So if the user asks for a smaller distortion than the best classifier can achieve from the observation, the numerical routine should reject the input rather than silently produce nonsense.

These two tests are not just software conveniences. They encode the mathematical domain of the theorem into executable checks.
