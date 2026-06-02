# rd-semantic-repro

Main theory-first project for reproducing the paper **A Rate-Distortion Framework for Characterizing Semantic Information**.

## 1. Why this project is implemented this way
This project is the main EE142 course-project track. The course asks for a professional report, clear technical explanation, and preferably a new idea. For that reason, we do **not** start from a large engineering system. We start from the mathematics of the paper itself.

The original paper is an information-theoretic paper. Its main contribution is not a software system, but a formal rate-distortion framework for a source modeled as (S, X):
- S: intrinsic state, representing the semantic content;
- X: extrinsic observation, representing the observable appearance.

The paper asks: what is the minimum rate needed to reproduce both semantic content and appearance under two separate distortion constraints?

That question is encoded by the **state-observation rate-distortion function** (SORDF). Since the central object is mathematical, our implementation also has to be mathematical first. Otherwise, the project would drift away from the original paper.

So the implementation strategy is:
1. reproduce the paper's basic numerical claims;
2. turn theorems and boundary conditions into executable tests;
3. only after that, extend the model with one modest new idea.

## 2. How this project is implemented
The repository is split into:
- src/: reusable numerical routines;
- 	ests/: theorem-inspired checks and boundary conditions;
- experiments/: scripts that will generate report figures;
- docs/: explanatory notes for the report and poster.

We intentionally use **test-driven development** for the numerical routines. In a mathematics-heavy project, tests are not just software checks; they are a way to encode the domain of validity of the formulas.

For example, in the binary classification case, we first implemented and tested:
- the Gaussian Q-function;
- the infeasible region D_s < Q(A/\sigma);
- the zero-rate boundary at D_s = 1/2;
- the one-bit boundary at D_s = Q(A/\sigma).

These are meaningful mathematical facts from the paper, not arbitrary implementation details.

## 3. Detailed mathematical relation to the paper
### 3.1 Source model
The paper models the source as a random pair (S, X).
- S is hidden and semantic;
- X is visible and observable.

The encoder sees only X, not S. The decoder produces both \hat S and \hat X.

This matters because semantic reconstruction is **indirect**: the decoder must infer S through information carried by X.

### 3.2 Two distortion constraints
The paper uses two distortions:
- d_s(s, \hat s): semantic distortion;
- d_a(x, \hat x): appearance distortion.

Hence the objective is not ordinary compression. It is not enough to reproduce the appearance well, and it is also not enough to do the semantic task well if the appearance must also be preserved. The interesting part is the tradeoff between the two.

### 3.3 State-observation rate-distortion function
The paper shows that the SORDF is
R(D_s, D_a) = min I(X; \hat S, \hat X)
subject to the semantic and appearance constraints.

This formula is the center of the project. It says the rate is controlled by how much information about X must be kept to support both reconstructed outputs.

### 3.4 Why the binary classification case is the first target
The binary case in the paper is:
- S is Bernoulli with equal prior;
- X | S=0 ~ N(A, \sigma^2);
- X | S=1 ~ N(-A, \sigma^2).

Semantic distortion is Hamming loss. Appearance distortion is squared error.

This is the best first reproduction target because:
1. it captures the meaning of semantic distortion very clearly;
2. the Bayes classification error has a clean boundary Q(A/\sigma);
3. the paper gives explicit formulas and an achievable upper bound;
4. the same setup can later be generalized to multiclass semantics.

### 3.5 Meaning of the current implementation
At the current stage, the code already encodes several theorem-level facts:
- if D_s < Q(A/\sigma), the target is infeasible;
- if D_s = Q(A/\sigma), the rate sits at the most demanding classification boundary in our current staged implementation;
- if D_s >= 1/2, zero rate is enough;
- between these endpoints, the rate should decrease as larger semantic distortion is allowed.

The present interior formula is still a staged implementation step. It gives us the right monotonic and boundary behavior while we continue building the fuller numerical procedure.

## 4. Planned next steps
1. replace the current staged interior formula with a closer numerical implementation of the paper's binary-case expression;
2. generate the first R(D_s, \infty) curve;
3. reproduce the achievable upper bound for R(D_s, D_a);
4. move to the Gaussian cases;
5. add a multiclass extension.

## 5. Why this helps the final report
This structure helps the report because it lets us explain the project in layers:
- first the model;
- then the theorem;
- then the numerical reproduction;
- then our extension.

That is exactly the kind of progression that makes an information theory project readable and persuasive.
