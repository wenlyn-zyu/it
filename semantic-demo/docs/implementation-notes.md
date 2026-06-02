# Implementation Notes for `semantic-demo`

## 1. Relation to the original paper
The original paper is not a deep learning paper. It is an information-theoretic paper that formalizes semantic information through the pair `(S, X)` and studies the minimum rate needed to control semantic and appearance distortion.

This demo does not claim to compute the exact SORDF. Instead, it translates the paper's core insight into a small machine learning experiment:
- the task label acts as a practical analogue of the intrinsic semantic state;
- the input sample acts as the observable appearance;
- a communication bottleneck acts as a rough rate proxy.

So this project is **inspired by** the original theory rather than being a literal theorem reproduction.

## 2. Why we include this demo at all
For a course project, theory alone can be rigorous but visually abstract. A small engineering demo helps in three ways:
1. it makes the semantic-vs-appearance tradeoff intuitive;
2. it gives a simple experimental story for a poster;
3. it connects information theory with modern learning-based communication ideas.

## 3. Why the baseline is task-agnostic compression
A fair contrast requires a baseline that tries to preserve the raw input rather than optimize directly for the downstream task. That is the engineering analogue of an appearance-oriented objective.

Examples include:
- generic dimensionality reduction;
- autoencoder reconstruction bottlenecks;
- standard compression then classification.

## 4. Why the alternative is task-oriented representation learning
The original paper argues that semantic relevance matters. In an engineering setting, the natural translation is to optimize the transmitted representation directly for task success, subject to a bottleneck.

In plain language: if the destination wants a label, we should not spend all our communication resources preserving details that do not affect the label.

## 5. Why this track remains secondary
This demo is intentionally auxiliary because the EE142 project sits in an information theory course. The theory-first reproduction remains the main contribution. This demo is there to strengthen intuition and presentation quality, not to replace the original mathematics.


## 6. Why the first baseline uses PCA + logistic regression
We intentionally start the demo with a very small classical baseline instead of a neural network.

Why:
1. it has almost no infrastructure cost;
2. it runs fast on the server;
3. it gives us a concrete notion of a communication bottleneck through the reduced feature dimension;
4. it provides a clean task-agnostic baseline before we introduce a task-oriented learned representation.

This is not meant to be the final semantic demo. It is the control experiment. Later, we can compare it against a learned task-oriented bottleneck under a similar dimension budget.
