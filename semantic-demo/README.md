# semantic-demo

Companion engineering demo for the EE142 project.

## 1. Why this project exists
This is the secondary track. The main track reproduces the mathematics of the paper. This demo track exists to make the same core idea visible in an engineering experiment:

> if the destination mainly cares about a task, then a task-oriented representation can outperform a generic appearance-preserving representation under the same bottleneck budget.

This is not a theorem-proof track. It is an intuition-building and presentation-support track.

## 2. How it relates to the original paper
The original paper studies a source (S, X) where:
- S is the semantic state;
- X is the observation.

In the demo:
- the class label is the practical analogue of S;
- the image or feature vector is the practical analogue of X;
- a feature bottleneck is used as a rough communication-budget proxy.

So this demo does not compute the exact SORDF. Instead, it illustrates the same semantic principle in a lightweight machine learning setting.

## 3. Why we start with PCA + logistic regression
We deliberately start with a very small baseline:
- PCA acts as generic, task-agnostic compression;
- logistic regression acts as a simple downstream classifier.

This choice is good for the project because it is:
1. easy to run on the server;
2. easy to explain in the report;
3. a clear control group for later task-oriented models.

## 4. Planned comparison
The intended comparison is:
- **baseline**: generic compressed representation then classification;
- **task-oriented**: representation learned directly for the label prediction task under a similar bottleneck budget.

If the task-oriented method performs better at the same reduced dimension, that supports the semantic-communication intuition behind the paper.


## 5. Current comparison result
We now generate a small comparison table across multiple bottleneck dimensions.

The current result already shows the desired qualitative trend:
- at small bottleneck sizes, the task-oriented supervised projection clearly outperforms the task-agnostic PCA baseline;
- as the bottleneck grows, the gap narrows, but the task-oriented method remains competitive or slightly better.

This is a useful supporting result for the project because it translates the semantic-information intuition into a compact empirical story: under constrained representation budgets, task-aware compression can be more effective than generic compression.


## 6. Current figure artifact
The current main C-track figure is 
esults/baseline_comparison.png, which plots classification accuracy against requested bottleneck dimension.

Its present qualitative message is straightforward:
- in the low-budget regime, the task-oriented supervised projection outperforms the task-agnostic PCA baseline;
- as the bottleneck grows, both methods improve, and the gap narrows.

This is exactly the kind of compact visual evidence we want from the engineering side of the project.
