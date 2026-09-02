# S1 — Row-level audit log (closing audit, 31 August 2026)

All 47 entries of the audited table were independently re-verified
against their primary texts, in six family-level passes (A–F below),
each verdict counter-checked at source before any correction was
applied. Eleven cells were corrected; none of the corrections concerns
the kernel-tractability column. No entry had received a source
revision after 27 August 2026.

Verdict legend: CONFORME = all coded cells confirmed against the
primary text; CORRECTED = the stated cell was amended, with the
primary-source ground.

## Pass A — references; conditional priors and schedules

| Entry | Verdict | Correction and ground |
|---|---|---|
| DDPM (arXiv:2006.11239) | CONFORME | — |
| Score-SDE (arXiv:2011.13456) | CORRECTED (error column) | `—` → `sensitivity`: §4.3 and Fig. 3 study sample quality vs probability-flow-ODE solver tolerance (“the number of function evaluations can be reduced by over 90% without affecting the visual quality of samples”). |
| PriorGrad (arXiv:2106.06406) | CONFORME | — |
| SpecGrad (arXiv:2203.16749) | CONFORME | — |
| MuLAN (arXiv:2312.13236) | CONFORME | ✓ read conditionally on the declared latent z (drawn once per trajectory). |
| Non-isotropic editing / NGDM (NeurIPS 2023) | CONFORME | identification confirmed on the NeurIPS proceedings (Yu, Gu, Liu, Sun). |
| Edge-Preserving Noise (arXiv:2410.01540) | CORRECTED (year) | `2026` → `2024/26`: v1 2 October 2024, v4 16 April 2026. |
| GPMP priors (arXiv:2509.25685) | CONFORME | — |

## Pass B — correlated and frequency-structured noise

| Entry | Verdict | Correction and ground |
|---|---|---|
| Cold Diffusion (arXiv:2208.09392) | CONFORME | — |
| IHDM (arXiv:2206.13397) | CONFORME | — |
| BDM (arXiv:2209.05557) | CONFORME | — |
| Soft Diffusion (arXiv:2209.05442) | CONFORME | — |
| Non-isotropic SDDM (arXiv:2210.12254) | CORRECTED (error column) | `—` → `ablations`: “We train two models on CIFAR10 … with the exact same hyperparameters”, evaluated at 1000/100/50/20/10 steps. |
| Blue Noise (arXiv:2402.04930) | CONFORME | — |
| Whitened Score (arXiv:2505.10311) | CONFORME | — |
| Squeezed (arXiv:2508.14871) | CONFORME | — |

## Pass C — manifold and metric

| Entry | Verdict | Correction and ground |
|---|---|---|
| RSGM (arXiv:2202.02763) | CONFORME | — |
| Riemannian Diffusion Models (arXiv:2208.07949) | CONFORME | — |
| Shortest Path Diffusion (arXiv:2306.00501) | CONFORME | — |
| Geodesic Diffusion (arXiv:2503.00745) | CORRECTED (structure cell) | the paper defines only conditional Gaussian marginals q(x_t|x_0)=N(α_t x_0, σ_t² I); no forward SDE and no process-level local a is specified — cell rewritten accordingly. |
| CDC-FM (arXiv:2510.05930) | CONFORME | — |
| Implicit Manifolds (arXiv:2604.07213) | CORRECTED (error column) | `accumulate (simulator)` → `bound + accumulate (simulator)`: explicit W₂ rates in the proof of Thm 3 (O(h) for the squared distance, O(h^{1/2}) for the distance). |
| RMM (arXiv:2606.14334) | CONFORME | — |
| MaxEnt-CDC (OpenReview U3LfUfaToV) | CONFORME | — |

## Pass D — anisotropic drift, matrix schedules, SPDE, latent routes

| Entry | Verdict | Correction and ground |
|---|---|---|
| Pang et al. (arXiv:2608.15103) | CORRECTED (structure cell) | PSD → SPD: Assumption 1 requires a symmetric positive definite schedule with m_β I ⪯ β(u) ⪯ B_β I. |
| SAGD (arXiv:2510.09660) | CONFORME | — |
| Anisotropic SPDE (arXiv:2605.08976) | CORRECTED (structure cell) | the forward has BOTH a nonlinear anisotropic drift ∇·(g₁(t,∇u)∇u) and gradient-dependent multiplicative noise; a is the state-dependent multiplication operator a(t,u)v=g₂(t,∇u)²v — cell rewritten. |
| GOU (arXiv:2606.03360) | CONFORME | — |
| PTL-Diffusion (arXiv:2606.09816) | CONFORME | — |
| Low-Pass FM (arXiv:2606.02177) | CONFORME | — |
| DDR (arXiv:2412.06981) | CONFORME | — |
| ILDM (arXiv:2608.04827) | CORRECTED (name) | IHLDM → ILDM: the authors’ acronym is “Intrinsic Hybrid Latent Diffusion Model (ILDM)”. |

## Pass E — preconditioning, sampler-error analyses, tools

| Entry | Verdict | Correction and ground |
|---|---|---|
| Girolami–Calderhead (JRSS-B 2011) | CONFORME | — |
| Precond. SGM (arXiv:2302.06504) | CORRECTED (structure cell; also in the representative table of the audited draft, since merged into the full table) | the coefficient ḡ_t²MM^⊤ belongs to the reverse predictor; the target-preserving Langevin corrector (role α, Thm 4.1) has a = ε_t²MM^⊤ — “the reverse diffusion … is different from the Langevin dynamics”. |
| Choi–Fan (arXiv:2506.08337) | CONFORME | — |
| Li–van der Schaar (arXiv:2308.05021) | CONFORME | — |
| Malliavin score (arXiv:2503.16917) | CONFORME | — |
| Khelifa et al. (arXiv:2606.06179) | CONFORME | — |
| Greco (arXiv:2505.13189) | CONFORME | — |
| Gu et al. (arXiv:2604.02751) | CORRECTED (error column) | `identify` → `identify + bound`: Thms 3.4/3.6 prove explicit stability bounds, including the curved-case reach(M) bound. |

## Pass F — generator level, channels, representations

| Entry | Verdict | Correction and ground |
|---|---|---|
| Generator Matching (arXiv:2410.20587) | CONFORME | — |
| UDMM (arXiv:2504.01938) | CONFORME | — |
| Tweedie beyond Gaussian (arXiv:2605.19391) | CONFORME | ✓ read for the instantiated closed-form families (GBM/BESQ/CIR). |
| LPGM (arXiv:2605.20547) | CONFORME | — |
| GPE (arXiv:2501.09876) | CORRECTED (structure cell) | the isometric/weakly α-bi-Lipschitz condition is imposed on the encoder only; the decoder is a reconstruction-trained approximate inverse — cell rewritten. |
| Generalized Score Matching (arXiv:2504.19288) | CONFORME | — |
| Forward-Learned Discrete (arXiv:2605.18204) | CONFORME | — |
