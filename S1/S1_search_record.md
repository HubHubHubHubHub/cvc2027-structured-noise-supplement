# S1 — Exact search record

## Declared search families (used at assembly, 27 August 2026)

Keyword searches on arXiv, OpenReview, and the proceedings of the
major learning and vision venues, combining the following term
families, completed by forward and backward citation snowballing from
the seed papers of each family:

1. anisotropic diffusion model / non-isotropic diffusion model
2. noise schedule
3. matrix-valued schedule
4. Riemannian score-based
5. carré du champ
6. data-dependent noise
7. latent diffusion geometry
8. error analysis of diffusion samplers

## Dated re-execution (31 August 2026, arXiv API)

Exact-phrase queries on the `all:` field of the arXiv export API
(`opensearch:totalResults`); the last family rendered as a conjunction
of phrases. No intermediate screening count is reconstructed
retroactively.

| Declared family | API query | totalResults |
|---|---|---:|
| anisotropic diffusion model | `all:"anisotropic diffusion model"` | 20 |
| non-isotropic diffusion model | `all:"non-isotropic diffusion model"` | 1 |
| noise schedule | `all:"noise schedule"` | 262 |
| matrix-valued schedule | `all:"matrix-valued schedule"` | 1 |
| Riemannian score-based | `all:"Riemannian score-based"` | 2 |
| carre du champ | `all:"carre du champ"` | 4 |
| data-dependent noise | `all:"data-dependent noise"` | 10 |
| latent diffusion geometry | `all:"latent diffusion geometry"` | 1 |
| error analysis of diffusion samplers | `all:"error analysis" AND "diffusion sampler"` | 2 |

Total identified on arXiv by exact-phrase re-execution: 303 records.
OpenReview and proceedings were covered by keyword search without a
counting API; snowballing completed each family.

## Eligibility and curation (as stated in the paper, Section 3, Review scope and methodology)

A work was *eligible for consideration* when it modified a corruption
mechanism in one of the surveyed loci, or gave a quantitative analysis
of error induced or propagated in a diffusion sampler; purely
architectural and dataset contributions were excluded. The corpus is
curated rather than exhaustive: it focuses on structured corruption
for static imaging, treats temporally correlated video noise as a
separate out-of-scope family, and represents the sampler-convergence
and non-local-generator literatures through selected landmarks.
