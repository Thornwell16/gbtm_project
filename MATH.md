# AutoTraj Mathematical Reference

This document is the complete technical reference for every formula implemented in AutoTraj. It serves as the mathematical appendix for the validation paper. All notation follows Nagin (1999, 2005) except where extensions are noted explicitly.

**V3.0 note:** Sections 1–4 below describe the model as of V3.0, which generalizes the mixing
proportions to depend on subject-level baseline covariates and generalizes the trajectory linear
predictor to include time-varying covariates (TVC). Setting the number of baseline covariates
$P=0$ and the number of TVCs $Q=0$ recovers the V1.5.0 model **exactly** (same parameter count,
same values) — V3.0 is a strict superset, not a breaking change.

**V4.0 note:** Section 1 also describes an optional per-subject survey/sampling weight $w_i$,
scaling each subject's contribution to the log-likelihood (and hence the gradient). Unlike V3.0,
this adds **no new parameters** to $\theta$ — it is a pure likelihood-level device. Setting
$w_i \equiv 1$ for all subjects recovers the unweighted (V3.0) model **exactly**.

**V5.0 note:** Section 9 (additive — does not modify §1-§8) describes a wholly separate model
family, the Nagin-style **joint dual-trajectory** model: two outcomes Y and Z, each with its own
independent single-outcome GBTM structure, linked by a joint latent-class probability matrix
instead of assuming independence. It does not compose with V3.0's mixing-covariates/TVC or V4.0's
survey weights in this pass (both are deferred future extensions).

---

## Table of Contents

1. [Model Overview](#1-model-overview)
2. [Parameter Vector Layout](#2-parameter-vector-layout)
3. [Log-Likelihood by Distribution](#3-log-likelihood-by-distribution)
4. [Gradient Derivations](#4-gradient-derivations)
5. [Hessian and Standard Error Computation](#5-hessian-and-standard-error-computation)
6. [BIC and AIC Formulas](#6-bic-and-aic-formulas)
7. [Model Adequacy Metrics](#7-model-adequacy-metrics)
8. [References](#8-references)
9. [Joint Dual-Trajectory Model (V5.0)](#9-joint-dual-trajectory-model-v50)

---

## 1. Model Overview

### Finite Mixture Representation

AutoTraj estimates a **Group-Based Trajectory Model (GBTM)** as a finite mixture of $K$ latent trajectory groups. Each subject $i = 1, \dots, N$ is assumed to belong to exactly one group $g \in \{0, 1, \dots, K-1\}$ with unknown probability $\pi_g$.

### Mixing Proportions: Intercept-Only or Covariate-Dependent

In the base model, mixing proportions are subject-invariant constants satisfying:

$$\pi_g \geq 0, \qquad \sum_{g=0}^{K-1} \pi_g = 1$$

**V3.0 extension.** When $P \geq 1$ baseline (time-invariant) covariates are supplied, the mixing
proportion becomes a function of subject $i$'s covariate row $x_i = (1, x_{i,1}, \dots, x_{i,P})$
(intercept prepended), via a multinomial logit:

$$\theta_g(x_i) = \Gamma_{g,0} + \sum_{p=1}^{P} \Gamma_{g,p}\, x_{i,p}, \qquad \theta_0(x_i) \equiv 0$$

$$\pi_g(x_i) = \frac{\exp(\theta_g(x_i))}{\sum_{j=0}^{K-1} \exp(\theta_j(x_i))}$$

With $P=0$, $\theta_g(x_i)$ reduces to a subject-invariant scalar $\theta_g$ and $\pi_g(x_i)$
reduces to the constant $\pi_g$ of the base model — every formula below is written in the general
$\pi_g(x_i)$ form, which specializes automatically.

### Subject-Level Joint Likelihood

Let $\mathbf{y}_i = (y_{i1}, \dots, y_{iT_i})$ denote the observed trajectory for subject $i$ over $T_i$ time points. Conditional on group membership $g$, the observations are assumed **independent across time**:

$$P(\mathbf{y}_i \mid g) = \prod_{t=1}^{T_i} P(y_{it} \mid g, t)$$

where $P(y_{it} \mid g, t)$ is the group- and distribution-specific likelihood contribution at time $t$ (detailed in Section 3), evaluated at the linear predictor $\eta_{igt}$ which (as of V3.0) may include time-varying covariate terms in addition to the polynomial-in-time terms.

### Marginal Likelihood

Marginalising over the latent group variable:

$$P(\mathbf{y}_i) = \sum_{g=0}^{K-1} \pi_g(x_i) \cdot P(\mathbf{y}_i \mid g) = \sum_{g=0}^{K-1} \pi_g(x_i) \prod_{t=1}^{T_i} P(y_{it} \mid g, t)$$

### Total Log-Likelihood

The total log-likelihood summed over all $N$ subjects is:

$$\ell(\theta) = \sum_{i=1}^{N} \log P(\mathbf{y}_i) = \sum_{i=1}^{N} \log \left[ \sum_{g=0}^{K-1} \pi_g(x_i) \prod_{t=1}^{T_i} P(y_{it} \mid g, t) \right]$$

This is the objective function maximised by the BFGS optimizer.

**V4.0 extension — survey/sampling weights.** When each subject $i$ carries a survey/sampling
weight $w_i > 0$ (e.g. an inverse-probability-of-selection weight), the objective generalizes to
the **weighted** total log-likelihood:

$$\ell_w(\theta) = \sum_{i=1}^{N} w_i \cdot \log P(\mathbf{y}_i)$$

By linearity of differentiation, $\partial \ell_w / \partial\theta = \sum_i w_i \cdot \partial \ell_i/\partial\theta$
— every gradient formula in Section 4 applies **completely unchanged** to each subject's own
contribution $\ell_i$; only the per-subject contribution (both to the objective and to the
gradient) is scaled by $w_i$ before summing. Setting $w_i \equiv 1$ for all subjects recovers
$\ell(\theta)$ exactly. This is a weighted **pseudo-MLE** (Binder, 1983) — see Section 5d for the
corresponding variance estimator, which is the *required* basis for inference under weighting, not
merely an optional cross-check. AutoTraj does not model stratification or clustering (PSU) design
effects — only independent-observations inverse-probability weighting is supported; this is a
documented limitation, not a solved problem.

### Posterior Group Probabilities

By Bayes' theorem, the posterior probability that subject $i$ belongs to group $g$ given their observed trajectory is:

$$P(g \mid i) = \frac{\pi_g(x_i) \cdot P(\mathbf{y}_i \mid g)}{\sum_{g'=0}^{K-1} \pi_{g'}(x_i) \cdot P(\mathbf{y}_i \mid g')}$$

These posterior probabilities are used in gradient computations and all post-estimation adequacy metrics.
Note that $P(g\mid i)$ itself does **not** depend on $w_i$ — survey weights only affect
*estimation* of $\theta$ (via the weighted objective above), not the Bayes'-rule posterior formula
evaluated at a given $\theta$.

---

## 2. Parameter Vector Layout

The optimizer operates on a flat real-valued vector $\theta \in \mathbb{R}^D$. The layout is fixed and described below. Let $K$ be the number of groups, $p_g$ be the polynomial degree for group $g$ (so group $g$ has $p_g + 1$ beta coefficients), $P$ be the number of baseline (mixing) covariates, and $Q$ be the number of time-varying covariates (TVCs).

$$
\theta = \big[\, \underbrace{\Gamma}_{\text{2a}} \;\big|\; \underbrace{\beta}_{\text{2b}} \;\big|\; \underbrace{\delta}_{\text{2c}} \;\big|\; \underbrace{\gamma_{\text{drop}}}_{\text{2d, optional}} \;\big|\; \underbrace{\text{raw\_}\sigma \text{ or } \zeta}_{\text{2e/2f, optional}} \,\big]
$$

### 2a. Mixing Weight Covariate Parameters (Gamma)

$$\theta[0 \,\dots\, (K-1)(P+1) - 1]$$

For each non-reference group $g = 1, \dots, K-1$, a length-$(P+1)$ block $[\Gamma_{g,0}, \Gamma_{g,1}, \dots, \Gamma_{g,P}]$ (intercept first, then baseline covariates in a fixed, user-supplied order), stored **group-major**. The reference group has $\Gamma_0 \equiv \mathbf{0}$ (a vector of zeros), exactly as $\theta_0 \equiv 0$ in the base model. The mixing proportions are recovered via the **softmax** transformation described in Section 1.

For $K = 1$ there are no Gamma parameters. With $P = 0$, this block has exactly $K-1$ scalars — identical in size and meaning to the base model's theta block.

### 2b. Trajectory Beta Coefficients

$$\theta\!\left[(K-1)(P+1) \;\dots\; (K-1)(P+1) + \sum_{g=0}^{K-1}(p_g + 1) - 1\right]$$

Unchanged from the base model. Beta coefficients are stored in **group-major order**: all coefficients for group 0, then all for group 1, etc. Within each group block the polynomial coefficients are ordered from intercept to highest degree:

$$[\beta_{g,0},\; \beta_{g,1},\; \dots,\; \beta_{g,p_g}]$$

**Note:** internally, times are rescaled to $t' = t / s$ where $s$ is a scale factor chosen at fit time (typically the maximum observed time). Betas in $\theta$ are therefore in scaled-time units; the unscaling matrix $D$ (Section 5b) converts SEs back to original-time units. TVC and mixing-covariate coefficients (2a, 2c) are **not** subject to this rescaling — see the note in Section 5b.

### 2c. Time-Varying Covariate (TVC) Deflection Parameters (Delta)

$$\theta[\delta_{\text{start}} \;\dots\; \delta_{\text{start}} + KQ - 1], \qquad \delta_{\text{start}} = (K-1)(P+1) + \sum_{g=0}^{K-1}(p_g+1)$$

For **every** group $g = 0, \dots, K-1$ (no reference-group exclusion here — these are trajectory-shape parameters like $\beta$, not log-ratio parameters like $\Gamma$), a length-$Q$ block $[\delta_{g,1}, \dots, \delta_{g,Q}]$, stored group-major, placed immediately after the last beta block. With $Q = 0$ this block has zero width and the layout reduces exactly to the base model's.

### 2d. Dropout Gamma Parameters (when `use_dropout=True`)

$$\theta[\gamma_{\text{start}} \;\dots\; \gamma_{\text{start}} + 3K - 1], \qquad \gamma_{\text{start}} = \delta_{\text{start}} + KQ$$

where $\gamma_{\text{start}}$ immediately follows the TVC block (or the last beta block if $Q=0$). For each group $g$ there are three parameters:

$$[\gamma_{g,0},\; \gamma_{g,1},\; \gamma_{g,2}]$$

stored in group-major order. The dropout sub-model does **not** depend on TVCs or mixing covariates — see the boundary note in Section 3e.

### 2e. CNORM Log-Sigma (CNORM distribution only)

$$\theta[-1] = \text{raw\_}\sigma = \log \sigma$$

A single scalar appended at the end of $\theta$. Unchanged from the base model — this and the ZIP block below are indexed from the *end* of the vector, which is why inserting the new 2a/2c blocks earlier in the layout requires no change to their indexing.

### 2f. ZIP Zero-Inflation Logits (ZIP distribution only)

$$\theta[-K \;\dots\; -1] = [\zeta_0, \zeta_1, \dots, \zeta_{K-1}]$$

One logit-scale parameter per group, appended after the beta/TVC/dropout blocks (and before raw_$\sigma$ if both were present, though in practice CNORM and ZIP are mutually exclusive). Unchanged from the base model.

---

## 3. Log-Likelihood by Distribution

All distributions share the linear predictor (generalized as of V3.0 to include TVC terms):

$$\eta_{igt} = \underbrace{\sum_{p=0}^{p_g} \beta_{g,p} \, t^p}_{\text{polynomial-in-time}} \;+\; \underbrace{\sum_{q=1}^{Q} \delta_{g,q} \, z_{i,q,t}}_{\text{TVC deflection}}$$

where $z_{i,q,t}$ is the value of TVC $q$ for subject $i$ at time $t$. With $Q=0$ the second term vanishes and $\eta_{igt}$ reduces to the base model's polynomial-only predictor. This $\eta_{igt}$ (or $\mu_{igt}$ for CNORM) is substituted, unchanged in functional form, into each of the four distribution-specific log-likelihoods below — none of their equations change shape, only the definition of their argument.

---

### 3a. LOGIT — Binary Longitudinal Outcomes

**Linear predictor:** $\eta_{igt}$ as defined above.

**Conditional probability of $y = 1$:**

$$P(y_{it} = 1 \mid g, t) = \sigma(\eta_{igt}) = \frac{1}{1 + e^{-\eta_{igt}}}$$

**Log-likelihood contribution per observation** (log-sum-exp stable form):

$$\ell_{igt}(y) = y \cdot \eta - \log(1 + e^{\eta})$$

To avoid overflow/underflow, the numerically stable implementation evaluates:

$$\ell_{igt}(y) = \begin{cases} y \cdot \eta - \eta - \log(1 + e^{-\eta}) & \text{if } \eta \geq 0 \\ y \cdot \eta - \log(1 + e^{\eta}) & \text{if } \eta < 0 \end{cases}$$

Both branches are equivalent to $y \eta - \log(1 + e^\eta)$ but avoid evaluating $e^\eta$ when $\eta$ is large positive or $e^{-\eta}$ when $\eta$ is large negative.

---

### 3b. CNORM — Censored Normal (Tobit Model)

**Linear predictor (mean):** $\mu_{igt} = \eta_{igt}$ as defined above.

**Residual standard deviation** (shared across groups and time):

$$\sigma = \exp(\text{raw\_}\sigma), \qquad \text{raw\_}\sigma \in \mathbb{R}$$

**Standardised residual:**

$$z = \frac{y - \mu}{\sigma}$$

Let $\Phi(\cdot)$ denote the standard Normal CDF and $\phi(\cdot)$ the standard Normal PDF. The data are censored at a lower bound $y_{\min}$ and an upper bound $y_{\max}$.

**Log-likelihood contribution** (three cases):

$$\ell_{igt}(y) = \begin{cases} \log \Phi(z_{\min}) & \text{if } y \leq y_{\min} \quad (\text{left-censored}) \\[6pt] \log \phi(z) - \log \sigma & \text{if } y_{\min} < y < y_{\max} \quad (\text{interior}) \\[6pt] \log\!\bigl(1 - \Phi(z_{\max})\bigr) & \text{if } y \geq y_{\max} \quad (\text{right-censored}) \end{cases}$$

where:

$$z_{\min} = \frac{y_{\min} - \mu}{\sigma}, \qquad z_{\max} = \frac{y_{\max} - \mu}{\sigma}$$

The **Inverse Mills Ratio (IMR)** arises in the gradients:

$$\text{IMR}^{-}(z) = \frac{\phi(z)}{\Phi(z)}, \qquad \text{IMR}^{+}(z) = \frac{\phi(z)}{1 - \Phi(z)}$$

where $\text{IMR}^{-}$ applies to left-censored observations and $\text{IMR}^{+}$ to right-censored observations.

---

### 3c. POISSON — Count Outcomes (Log Link)

**Linear predictor:** $\eta_{igt}$ as defined above.

**Conditional mean (rate):**

$$\lambda_{igt} = \exp(\eta_{igt})$$

**Log-PMF contribution:**

$$\ell_{igt}(y) = y \cdot \eta - e^{\eta} - \log(y!)$$

This is the canonical Poisson log-likelihood under a log link. The term $\log(y!)$ is constant with respect to $\theta$ and contributes only to the absolute value of $\ell$, not to the gradient.

---

### 3d. ZIP — Zero-Inflated Poisson

**Structural zero probability** (per group, time-constant):

$$\omega_g = \sigma(\zeta_g) = \frac{1}{1 + e^{-\zeta_g}}$$

where $\zeta_g$ is the logit-scale parameter for group $g$.

**Rate:**

$$\lambda_{igt} = \exp(\eta_{igt})$$

**Mixture PMF:**

$$P(y_{it} \mid g, t) = \begin{cases} \omega_g + (1 - \omega_g)\, e^{-\lambda} & \text{if } y = 0 \\[6pt] (1 - \omega_g)\, \dfrac{e^{-\lambda} \lambda^y}{y!} & \text{if } y > 0 \end{cases}$$

**Log-likelihood contribution:**

$$\ell_{igt}(y) = \begin{cases} \log\!\bigl[\omega_g + (1 - \omega_g)\, e^{-\lambda}\bigr] & \text{if } y = 0 \\[6pt] \log(1 - \omega_g) + y \eta - \lambda - \log(y!) & \text{if } y > 0 \end{cases}$$

Define $p_0 \equiv \omega_g + (1 - \omega_g) e^{-\lambda}$ for notational convenience in the gradient section.

---

### 3e. Informative Dropout — MNAR Model

When `use_dropout=True`, AutoTraj augments the likelihood with a **logistic dropout sub-model** that accounts for Missing Not At Random (MNAR) attrition. For each group $g$ and each time point $t > t_0$ (where $t_0$ is the first observed time), the probability of dropout is:

$$P(\text{drop}_{it} = 1 \mid g, t, y_{i,t-1}) = \sigma\!\left(\gamma_{g,0} + \gamma_{g,1} \cdot t + \gamma_{g,2} \cdot y_{i,t-1}\right)$$

**V3.0 boundary note:** this dropout hazard is defined purely in terms of $t$ and the lagged outcome — it does **not** depend on $\eta_{igt}$, TVCs, or mixing covariates. This is a deliberate scope boundary for V3.0, not an oversight: allowing TVCs to also deflect the dropout hazard is a reasonable future extension but is out of scope here.

Let $d_{it}$ denote the dropout indicator ($d_{it} = 1$ if $t$ is the last observed time for subject $i$, $d_{it} = 0$ for all preceding observed times).

**Log-likelihood contribution of the dropout process:**

$$\ell^{\text{drop}}_{igt} = \begin{cases} \log\!\bigl(1 - P(\text{drop}_{it})\bigr) & \text{if } d_{it} = 0 \quad (\text{subject not yet dropped}) \\ \log P(\text{drop}_{it}) & \text{if } d_{it} = 1 \quad (\text{last observed time}) \end{cases}$$

The total log-likelihood becomes:

$$\ell(\theta) = \sum_{i=1}^{N} \log \left[ \sum_{g=0}^{K-1} \pi_g(x_i) \prod_{t} P(y_{it} \mid g, t) \cdot \prod_{t > t_0} P(\text{drop}_{it} \mid g, t, y_{i,t-1})^{1} \right]$$

---

## 4. Gradient Derivations

All gradients are computed **analytically** (not by finite differences) and passed to BFGS as the Jacobian. The derivations use the chain rule through the log-sum-exp marginal likelihood.

Define the **per-subject log-sum** in numerically stable form:

$$L_i = \log \sum_{g=0}^{K-1} \exp\!\left(\log \pi_g(x_i) + \sum_t \ell_{igt}\right)$$

Gradients of $L_i$ with respect to $\theta$ propagate through the softmax and the per-observation likelihoods via the posterior weights $P(g \mid i)$.

**V4.0 note:** every gradient formula below is a formula for $\partial \ell_i/\partial\theta$ (a
single subject's contribution). Under survey weighting (Section 1), the total gradient is
$\sum_i w_i \cdot \partial \ell_i/\partial\theta$ — each subject's *entire* gradient row (every
block: $\Gamma$, $\beta$, $\delta$, dropout $\gamma$, $\text{raw}\_\sigma$/$\zeta$) is scaled by
that subject's $w_i$ once, after being fully assembled; no per-block formula below changes.

---

### 4a. Mixing Covariate (Gamma) Gradient

The per-subject softmax-derivative identity holds pointwise for each subject regardless of what $\theta_g$ is a function of:

$$\frac{\partial \ell_i}{\partial \theta_g(x_i)} = P(g \mid i) - \pi_g(x_i)$$

Applying the chain rule through $\theta_g(x_i) = \Gamma_{g,0} + \sum_p \Gamma_{g,p} x_{i,p}$ (with $x_{i,0} \equiv 1$), for $g > 0$ and $p = 0, \dots, P$:

$$\frac{\partial \ell_i}{\partial \Gamma_{g,p}} = \bigl[P(g \mid i) - \pi_g(x_i)\bigr] \cdot x_{i,p}$$

$$\frac{\partial \ell}{\partial \Gamma_{g,p}} = \sum_{i=1}^{N} \bigl[P(g \mid i) - \pi_g(x_i)\bigr] \cdot x_{i,p}$$

Setting $p=0$ (the intercept, $x_{i,0}=1$) recovers the base model's theta gradient exactly:
$\partial \ell / \partial \Gamma_{g,0} = \sum_i [P(g\mid i) - \pi_g(x_i)]$. At the MLE, this equals zero for every $p$, generalizing the intuitive base-model result (estimated mixing proportion equals mean posterior probability) to a proper multinomial-logit score-equation condition.

---

### 4b. Beta (Trajectory Polynomial) Gradient

$$\frac{\partial \ell_i}{\partial \beta_{g,p}} = \sum_{t=1}^{T_i} P(g \mid i) \cdot \varepsilon_\mu^{(g,t)} \cdot t^p$$

where $\varepsilon_\mu^{(g,t)}$ is the **distribution-specific score with respect to the linear predictor** $\eta$:

| Distribution | Observation type | $\varepsilon_\mu^{(g,t)}$ |
|---|---|---|
| LOGIT | any | $y - \sigma(\eta)$ |
| CNORM | interior ($y_{\min} < y < y_{\max}$) | $(y - \mu)/\sigma^2$ |
| CNORM | left-censored ($y \leq y_{\min}$) | $-\,\text{IMR}^{-}(z_{\min})/\sigma$ |
| CNORM | right-censored ($y \geq y_{\max}$) | $+\,\text{IMR}^{+}(z_{\max})/\sigma$ |
| Poisson | any | $y - \lambda$ |
| ZIP | $y = 0$ | $-\,(1-\omega_g)\,e^{-\lambda}\,\lambda \;/\; p_0$ |
| ZIP | $y > 0$ | $y - \lambda$ |

This table is unchanged from the base model — $\varepsilon_\mu^{(g,t)}$ is a function of $\eta$, not of what $\eta$ is composed of, so it applies identically whether or not TVC terms are present.

---

### 4c. TVC (Delta) Gradient

Since $\partial \eta_{igt} / \partial \delta_{g,q} = z_{i,q,t}$, and $\varepsilon_\mu^{(g,t)}$ is exactly the same score residual already computed for the beta gradient (Section 4b) — no new per-distribution derivation is required:

$$\frac{\partial \ell_i}{\partial \delta_{g,q}} = \sum_{t=1}^{T_i} P(g \mid i) \cdot \varepsilon_\mu^{(g,t)} \cdot z_{i,q,t}$$

This is structurally identical to the beta gradient (4b) with the regressor $t^p$ replaced by $z_{i,q,t}$. Implementations should compute $\varepsilon_\mu^{(g,t)}$ once per (group, observation) and reuse it for both the beta and delta gradient accumulations.

---

### 4d. CNORM Raw-Sigma Gradient

Because $\sigma = \exp(\text{raw\_}\sigma)$, the chain rule introduces a factor of $\sigma$:

$$\frac{\partial \ell_{igt}}{\partial \text{raw\_}\sigma} = \varepsilon_{\text{aux}}^{(g,t)}$$

where:

| Observation type | $\varepsilon_{\text{aux}}^{(g,t)}$ |
|---|---|
| Interior | $-1 + z^2$ |
| Left-censored | $-z \cdot \text{IMR}^{-}(z_{\min})$ |
| Right-censored | $+z \cdot \text{IMR}^{+}(z_{\max})$ |

The total gradient is:

$$\frac{\partial \ell}{\partial \text{raw\_}\sigma} = \sum_{i=1}^{N} \sum_{g=0}^{K-1} P(g \mid i) \sum_{t=1}^{T_i} \varepsilon_{\text{aux}}^{(g,t)}$$

---

### 4e. ZIP Zeta Gradient (Per Group)

Let $\omega_g = \sigma(\zeta_g)$ so $\partial \omega_g / \partial \zeta_g = \omega_g(1 - \omega_g)$.

**For $y = 0$:**

$$\frac{\partial \log p_0}{\partial \zeta_g} = \frac{(1 - e^{-\lambda})}{p_0} \cdot \omega_g(1 - \omega_g)$$

**For $y > 0$:**

$$\frac{\partial \log p_{y>0}}{\partial \zeta_g} = -\omega_g$$

(Since $\partial \log(1-\omega_g)/\partial \zeta_g = -\omega_g$.)

The subject-level gradient contribution is:

$$\frac{\partial \ell_i}{\partial \zeta_g} = P(g \mid i) \sum_{t=1}^{T_i} \frac{\partial \log P(y_{it} \mid g, t)}{\partial \zeta_g}$$

---

### 4f. Dropout Gamma Gradient

Let $q_{igt} = P(\text{drop}_{it} = 1 \mid g, t, y_{i,t-1}) = \sigma(\gamma_{g,0} + \gamma_{g,1} t + \gamma_{g,2} y_{i,t-1})$.

Define the **dropout score** $\varepsilon_{\text{drop}}^{(g,t)}$:

$$\varepsilon_{\text{drop}}^{(g,t)} = \begin{cases} -q_{igt} & \text{if } d_{it} = 0 \quad \text{(not dropped; penalise dropout probability)} \\ 1 - q_{igt} & \text{if } d_{it} = 1 \quad \text{(dropped; reward dropout probability)} \end{cases}$$

The gradients with respect to the three gamma parameters for group $g$ are:

$$\frac{\partial \ell_i}{\partial \gamma_{g,0}} = P(g \mid i) \sum_{t > t_0} \varepsilon_{\text{drop}}^{(g,t)} \cdot 1$$

$$\frac{\partial \ell_i}{\partial \gamma_{g,1}} = P(g \mid i) \sum_{t > t_0} \varepsilon_{\text{drop}}^{(g,t)} \cdot t$$

$$\frac{\partial \ell_i}{\partial \gamma_{g,2}} = P(g \mid i) \sum_{t > t_0} \varepsilon_{\text{drop}}^{(g,t)} \cdot y_{i,t-1}$$

As noted in Section 3e, these dropout gradients do not involve $\Gamma$ or $\delta$ at all — the dropout sub-model is unaffected by the V3.0 extension.

---

## 5. Hessian and Standard Error Computation

### 5a. Numerical Hessian via Central Finite Differences

AutoTraj approximates the Hessian of the **negative** log-likelihood using the gradient (Jacobian) already available analytically. The central finite-difference approximation of the Hessian's $j$-th column is:

$$H_{\cdot j} \approx \frac{\nabla f(\theta + \varepsilon_j \mathbf{e}_j) - \nabla f(\theta - \varepsilon_j \mathbf{e}_j)}{2\varepsilon_j}$$

where the adaptive step size is:

$$\varepsilon_j = \max\!\left(10^{-5} \cdot |\theta_j|,\; 10^{-8}\right)$$

After building the full $D \times D$ matrix, it is **symmetrised**:

$$H \leftarrow \frac{H + H^\top}{2}$$

This is the Hessian of $-\ell(\theta)$, so the model-based covariance is $H^{-1}$.

---

### 5b. Time-Scale Unscaling Matrix $D$

Because betas are estimated in **scaled time** $t' = t/s$, a polynomial coefficient $\beta_{g,p}$ in scaled time corresponds to $\beta_{g,p} / s^p$ in original time. The unscaling matrix $D$ is diagonal:

$$D_{jj} = \begin{cases} s^{-p} & \text{if parameter } j \text{ is } \beta_{g,p} \text{ (polynomial coefficient of degree } p\text{)} \\ 1 & \text{for all other parameters (Gamma, delta, gamma\_drop, raw\_}\sigma\text{, zeta)} \end{cases}$$

where $s$ is the time scale factor used at fit time. **V3.0 note:** mixing-covariate ($\Gamma$) and TVC ($\delta$) coefficients get $D_{jj}=1$ — they multiply arbitrary user-supplied covariates, not powers of rescaled time, so no unscaling applies. Continuous covariates on very different numeric scales should be standardized by the user before fitting to keep the Hessian well-conditioned (see identifiability guidance in the implementation notes).

---

### 5c. Model-Based Covariance and Standard Errors

$$V_{\text{model}} = D \cdot H^{-1} \cdot D$$

where $H^{-1}$ is the Moore-Penrose pseudoinverse (computed via `numpy.linalg.pinv` to handle near-singular cases).

$$\mathrm{SE}_{\text{model}} = \sqrt{\left|\operatorname{diag}(V_{\text{model}})\right|}$$

The absolute value is taken element-wise to guard against small negative diagonal entries due to numerical imprecision.

**V4.0 note:** under survey weighting, $H$ is the Hessian of the *weighted* NLL
$-\ell_w(\theta)$, so $V_{\text{model}}$ is a weighted information-matrix inverse — but this
naive "model-based" SE is **not** a valid/consistent variance estimator under weighting (Binder,
1983; Skinner, Holt & Smith, 1989), since it ignores the weighting design entirely. It is
retained here only for reference/diagnostic display; see Section 5d for the required estimator
once weights are used.

---

### 5d. Huber-White Sandwich Estimator (Robust SEs)

Let $\mathbf{g}_i = \nabla_\theta \ell_i(\hat\theta)$ be the score vector (gradient of the log-likelihood) for subject $i$ evaluated at the MLE.

The **"meat"** of the sandwich is:

$$G = \sum_{i=1}^{N} \mathbf{g}_i \mathbf{g}_i^\top$$

The **sandwich covariance** in original-time units is:

$$V_{\text{robust}} = D \cdot H^{-1} \cdot G \cdot H^{-1} \cdot D$$

$$\mathrm{SE}_{\text{robust}} = \sqrt{\operatorname{diag}(V_{\text{robust}})}$$

The sandwich estimator is consistent under misspecification of the within-subject correlation structure and heteroskedasticity (White, 1980).

**V4.0 extension — weighted sandwich variance.** Under survey weighting (Section 1), the
implementation's per-subject score $\mathbf{g}_i$ is *already* the weighted score
$w_i \cdot \nabla_\theta \ell_i(\hat\theta)$ (Section 4's weighting note), so the same formula
above automatically yields:

$$G = \sum_{i=1}^{N} (w_i \mathbf{g}_i)(w_i \mathbf{g}_i)^\top = \sum_{i=1}^{N} w_i^2\, \mathbf{g}_i \mathbf{g}_i^\top$$

— exactly the standard Binder-type weighted-pseudo-MLE sandwich ("meat"), with **no separate
computation required**. This is the *required* basis for valid inference once weights are used
(see the Section 5c note); AutoTraj does not implement the additional stratum/PSU Taylor-series
linearization terms of a full complex-survey design (`svydesign`-style variance) — only
independent-observations IPW is supported.

---

## 6. BIC and AIC Formulas

AutoTraj reports **two parallel conventions**. They are equivalent for model selection (same ordering) but differ in sign and scaling.

Let:
- $\ell = \ell(\hat\theta)$ — maximised log-likelihood (or $\ell_w(\hat\theta)$, the weighted
  log-likelihood, when survey weights are used — see V4.0 note below)
- $p$ — total number of free parameters (dimension of $\theta$; as of V3.0 this includes the $\Gamma$ and $\delta$ blocks when present)
- $N$ — number of **subjects** (not observations)

**V4.0 note:** under survey weighting, AutoTraj plugs the *weighted* $\ell_w$ directly into the
formulas below with $N$ and $p$ left as the raw (unweighted) subject count and parameter count —
a simplification consistent with common practice in weighted-GBTM software, not a fully resolved
theoretical question (an "effective sample size" adjustment, e.g. Kish's design effect, is a
documented open alternative AutoTraj does not implement). BIC/AIC-based model selection under
weighting should be interpreted with this caveat in mind.

---

### Nagin Convention (higher is better)

Used as the primary model-selection criterion in AutoTraj, following the convention of Nagin's SAS procedure (Jones & Nagin, 2001):

$$\text{BIC}_N = \ell - \frac{1}{2} \cdot p \cdot \log N$$

$$\text{AIC}_N = \ell - p$$

A model with a larger (less negative) $\text{BIC}_N$ is preferred.

---

### Standard Statistical Convention (lower is better)

The standard textbook/software convention (e.g., R's `BIC()`, Stata's `estat ic`):

$$\text{BIC}_S = -2\ell + p \cdot \log N$$

$$\text{AIC}_S = -2\ell + 2p$$

A model with a smaller $\text{BIC}_S$ is preferred.

The two conventions are related by:

$$\text{BIC}_S = -2 \cdot \text{BIC}_N - p \log N \quad \text{(not a simple sign flip)}$$

Both are displayed in AutoTraj output to facilitate comparison with other software.

---

## 7. Model Adequacy Metrics

These metrics assess the quality of group separation after estimation. They do not affect optimisation.

---

### Average Posterior Probability (AvePP)

The AvePP for group $g$ is the mean posterior probability of belonging to group $g$ among subjects **assigned** to group $g$ (i.e., subjects for whom $g$ is the modal group):

$$\text{AvePP}_g = \frac{1}{N_g} \sum_{i:\, \hat{g}_i = g} P(g \mid i)$$

where $\hat{g}_i = \arg\max_g P(g \mid i)$ is the modal assignment and $N_g = |\{i: \hat{g}_i = g\}|$.

**Adequacy threshold:** $\text{AvePP}_g \geq 0.70$ (Nagin, 2005).

---

### Odds of Correct Classification (OCC)

$$\text{OCC}_g = \frac{\text{AvePP}_g / (1 - \text{AvePP}_g)}{\pi_g / (1 - \pi_g)}$$

This is the ratio of the estimated odds of correct classification to the odds expected under random assignment. It equals 1.0 if the model provides no improvement over chance. **V3.0 note:** when mixing covariates are present, $\pi_g$ here is evaluated at the sample-average covariate profile (i.e., the marginal/average mixing proportion across subjects), since OCC is defined relative to a single reference "chance" rate per group.

**Adequacy threshold:** $\text{OCC}_g \geq 5.0$ (Nagin, 2005).

---

### Relative Entropy

The relative entropy measures the sharpness of the posterior distribution across subjects. Perfect assignment (each $P(g \mid i) \in \{0, 1\}$) gives $H_{\text{rel}} = 1$; completely flat posteriors give $H_{\text{rel}} = 0$:

$$H_{\text{rel}} = 1 + \frac{1}{N \log K} \sum_{i=1}^{N} \sum_{g=0}^{K-1} P(g \mid i) \cdot \log P(g \mid i)$$

The inner double sum equals the total entropy of the posterior distribution (which is non-positive), normalised by $N \log K$ (the maximum entropy for $K$ groups).

**Range:** $[0, 1]$. Values $\geq 0.50$ indicate adequate group separation.

---

## 8. References

- Nagin, D.S. (1999). Analyzing developmental trajectories: A semiparametric, group-based approach. *Psychological Methods*, 4(2), 139–157.

- Jones, B.L., & Nagin, D.S. (2001). A SAS procedure for group-based trajectory modeling. *Sociological Methods & Research*, 29(3), 374–393.

- Nagin, D.S. (2005). *Group-Based Modeling of Development*. Harvard University Press.

- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817–838.

- Tobin, J. (1958). Estimation of relationships for limited dependent variables. *Econometrica*, 26(1), 24–36.

---

## 9. Joint Dual-Trajectory Model (V5.0)

This section is **additive** — it describes a separate model family and does not modify §1-§8.
Two outcomes Y and Z, each with its own independent GBTM structure (own group count, polynomial
orders, distribution, dropout toggle), linked by a **joint latent-class probability matrix**
$\pi_{gh}$ ($K_Y \times K_Z$) instead of assuming the two outcomes' group memberships are
independent — the standard "dual trajectory" approach (Nagin & Tremblay). Does not compose with
V3.0's mixing-covariates/TVC or V4.0's survey weights in this pass (deferred future extensions).

### 9a. Conditional Independence Assumption

Given joint class $(g,h)$, Y's trajectory and Z's trajectory are assumed conditionally independent:

$$P(y_i, z_i \mid g, h) = P(y_i \mid g) \cdot P(z_i \mid h)$$

where each factor is computed by the **same** single-outcome machinery in §3 — Y's factor uses
outcome Y's own distribution/parameters exactly as a standalone single-outcome model would, and
likewise for Z. No new per-observation likelihood formulas are introduced by V5.0.

### 9b. Parameter Vector Layout

$$\theta_{\text{joint}} = \big[\, \Theta_{\text{joint}} \;\big|\; \text{Y-BLOCK} \;\big|\; \text{Z-BLOCK} \,\big]$$
$$\text{Y-BLOCK} = [\, \beta_Y \;|\; \gamma_{\text{drop},Y} \text{ (optional)} \;|\; \text{tail}_Y \text{ (optional)} \,]$$
$$\text{Z-BLOCK} = [\, \beta_Z \;|\; \gamma_{\text{drop},Z} \text{ (optional)} \;|\; \text{tail}_Z \text{ (optional)} \,]$$

- **$\Theta_{\text{joint}}$** ($K_Y K_Z - 1$ params): the $K_Y \times K_Z$ joint-class grid
  flattened row-major ($g$ outer, $h$ inner), skipping the reference cell $(0,0)$, which is
  implicitly $\theta_{0,0} \equiv 0$.
- **Y-BLOCK / Z-BLOCK**: each is exactly a "solo" single-outcome parameter vector — group-major
  $\beta$, then optional 3-per-group dropout $\gamma$, then an optional CNORM raw-$\sigma$ or ZIP
  $\zeta$ tail — identical in form to §2b-2e/2f with no Gamma/delta blocks (V5.0 doesn't use
  mixing covariates or TVCs). **Y's tail sits immediately before Z's block starts** (not at the
  vector's absolute end) — this is what makes $\theta_{\text{joint}}$'s Y-slice and Z-slice each
  individually look like a standalone single-outcome vector with its own tail "at the end" of
  that slice, letting both reuse the identical per-outcome subroutines described in 9d.

### 9c. Joint Likelihood and Posteriors

$$\pi_{gh} = \frac{\exp(\theta_{gh})}{\sum_{g',h'} \exp(\theta_{g'h'})}, \qquad \theta_{0,0} \equiv 0$$

$$P(y_i, z_i) = \sum_{g=0}^{K_Y-1} \sum_{h=0}^{K_Z-1} \pi_{gh} \cdot P(y_i \mid g) \cdot P(z_i \mid h)$$

computed via a 2-D log-sum-exp over the flattened $K_Y \cdot K_Z$ grid (same stability pattern as
§1's marginal likelihood). Total log-likelihood: $\ell(\theta_{\text{joint}}) = \sum_i \log P(y_i, z_i)$.

Joint posterior:

$$P(g,h \mid i) = \frac{\pi_{gh} \cdot P(y_i\mid g) \cdot P(z_i\mid h)}{P(y_i,z_i)}$$

**Marginal posteriors** — these are what the per-outcome gradients need (derived, not assumed, in 9d):

$$P(g\mid i) = \sum_{h=0}^{K_Z-1} P(g,h\mid i), \qquad P(h\mid i) = \sum_{g=0}^{K_Y-1} P(g,h\mid i)$$

The estimated $\pi_{gh}$ matrix itself, and the **conditional probabilities**
$P(h\mid g) = \pi_{gh} / \sum_{h'}\pi_{gh'}$ (row-normalize) and $P(g\mid h)$ (column-normalize),
are the key applied deliverables — they answer "do these two behaviors co-develop?", the standard
question a dual-trajectory analysis is run to answer.

### 9d. Gradient Derivations

**$\Theta_{\text{joint}}$ gradient.** The softmax-derivative identity from §4a
($\partial \ell_i/\partial\theta_g = P(g\mid i) - \pi_g$) holds pointwise for each subject
regardless of what the categorical index represents. Treating the flattened joint class $(g,h)$ as
a single categorical index of size $K_Y K_Z$ transfers the identity unchanged:

$$\frac{\partial \ell_i}{\partial \theta_{gh}} = P(g,h\mid i) - \pi_{gh}, \qquad (g,h) \neq (0,0)$$

**$\beta_Y$ gradient — derived explicitly** (not assumed). Only the $g'=g$ terms of the joint
log-sum depend on $\beta_{Y,g,p}$:

$$\frac{\partial \ell_i}{\partial \beta_{Y,g,p}} = \frac{\partial L_{Y,i}(g)}{\partial \beta_{Y,g,p}} \cdot \frac{\sum_h \pi_{gh}\exp(L_{Y,i}(g)+L_{Z,i}(h))}{P(y_i,z_i)} = \frac{\partial L_{Y,i}(g)}{\partial \beta_{Y,g,p}} \cdot \sum_h P(g,h\mid i) = \frac{\partial L_{Y,i}(g)}{\partial \beta_{Y,g,p}} \cdot P(g\mid i)$$

Since $L_{Y,i}(g)$ is *exactly* the same function of $\beta_Y$ as the single-outcome model's
$L_{ig}$, $\partial L_{Y,i}(g)/\partial\beta_{Y,g,p} = \sum_t \varepsilon_\mu^{(Y,g,t)}\cdot t^p$
unchanged from §4b. Therefore:

$$\boxed{\frac{\partial \ell_i}{\partial \beta_{Y,g,p}} = P(g\mid i)\cdot\sum_t \varepsilon_\mu^{(Y,g,t)}\cdot t^p}$$

— **identical in form** to the existing single-outcome beta gradient (§4b), with the joint
model's **marginal** $P(g\mid i)$ substituted for the single-outcome posterior. The same argument
(nothing outside its own outcome's sum depends on that outcome's parameters) applies verbatim to
$\beta_Z$ (weight $P(h\mid i)$), and to $\gamma_{\text{drop},Y/Z}$ and $\text{raw}\_\sigma$/$\zeta$
tails — every per-outcome parameter block's gradient is the existing single-outcome formula (§4b-4f)
with the appropriate marginal posterior substituted for the posterior. This is exactly what
justifies reusing the single-outcome gradient-accumulation code unchanged for both outcomes.

### 9e. Standard Errors, BIC/AIC

Same recipe as §5 (finite-difference Hessian, model-based and Huber-White sandwich SEs) applied to
the wider joint parameter vector, with **two independent time-unscaling factors** — each outcome
rescales only its own $\beta$ block by its own $s_Y$/$s_Z$ scale factor; $\Theta_{\text{joint}}$
stays dimensionless ($D=1$), same convention as Gamma/delta in §5b. BIC/AIC (§6) use the joint
$\ell$, the shared subject count $N$, and $p$ = the full joint vector's dimension.

### 9f. Label-Switching Is Two-Dimensional

Both outcomes' groups can each be independently relabeled by the optimizer, so the ascending-
intercept resort (§ analogous to the base model's group resort) must run **independently per
outcome** and then **consistently re-permute $\pi_{gh}$ across both axes**:

1. Compute $\text{sortY}$, $\text{sortZ}$ (ascending-intercept permutations) independently for Y
   and Z.
2. Permute $\beta_Y/\gamma_{\text{drop},Y}/\text{tail}_Y$ and
   $\beta_Z/\gamma_{\text{drop},Z}/\text{tail}_Z$ mechanically, same per-block logic as the
   existing single-outcome resort.
3. $\Theta_{\text{joint}}$'s raw logits **cannot** be permuted directly — they are only meaningful
   relative to the *old* reference cell $(0,0)$. Instead: reconstruct the full $\pi$ matrix via
   softmax (uniquely defined), permute **both axes simultaneously**
   ($\pi_{\text{new}} = \pi_{\text{old}}[\text{sortY}, \text{sortZ}]$), then re-derive
   $\theta'_{gh} = \log(\pi_{\text{new}}[g,h]) - \log(\pi_{\text{new}}[0,0])$. This full recompute
   runs whenever *either* axis needs resorting, since the reference cell can change even if only
   one axis actually permutes.
4. $\Theta_{\text{joint}}$ SE carryover (mapping each new cell back to its old cell) is an
   **approximation** — same caveat as the base model's Gamma SE carryover (§2a's note). Y-BLOCK/
   Z-BLOCK beta/dropout/tail SEs are exact (pure mechanical permutation).

This is the highest-risk correctness surface in the V5.0 implementation. It is verified by a
dedicated invariance test: resorting a hand-built parameter vector must reproduce the identical
NLL on the same data before vs. after the resort (label permutation is a likelihood-invariant
symmetry — the strongest available check), analogous to the equivalent 1-D invariance test added
for the base model's `sort_groups_by_intercept`.

### 9g. Backward-Compatibility Framing

Unlike V3.0's "$P=0,Q=0$" and V4.0's "$w_i\equiv 1$" invariants (which reduce a wider model back
to the *exact same* prior model), V5.0 is a wholly new capability with no default that reduces to
an existing single-outcome fit. The closest analogous guarantee: with $K_Z=1$, $P(z_i\mid h{=}0)$
has no $g$-dependence, so the joint log-likelihood **additively separates**:

$$\log P(y_i,z_i) = \log P(z_i\mid h{=}0) + \log\sum_g \pi_g P(y_i\mid g)$$

— the second term is exactly the single-outcome Y log-likelihood, so the $(\Theta_Y,\beta_Y)$
optimum equals the single-outcome Y MLE exactly. Because it is nonetheless a different (larger,
jointly-optimized) numerical problem, this is checked via **tolerance-based** agreement against an
independent single-outcome fit, not bit-identical equality — a deliberately weaker guarantee than
V3.0/V4.0's invariants, stated as such rather than oversold.
