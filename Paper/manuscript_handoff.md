# Handoff: Spaceflight scRNA-seq Biological-Fidelity Evaluation Manuscript

**Purpose of this document.** Transfer the full state of the manuscript to another agent so it can continue without re-deriving context. Read this top to bottom once, then work from the "What is still needed" backlog. The deliverable file is `manuscript.tex` (single self-contained LaTeX, compiles with `pdflatex`, ready for Overleaf). A compiled `manuscript.pdf` sits beside it.

Target venue: **npj Microgravity** (Nature Portfolio). Lead/corresponding author: **Arsh Suri**. Co-authors: Walter Alvarado, James Casaletto. Research home: NASA Ames AI4LS (AI for Life in Space), advised by Dr. James Casaletto; originated via NASA SLSTP, continued via Georgia Tech VIP.

---

## 1. The one-paragraph thesis

Spaceflight transcriptomics is bottlenecked by tiny cohorts. Deep generative models can augment scarce single-cell RNA-seq (scRNA-seq), but the field judges synthetic data almost entirely by distributional metrics that are blind to biology. This paper introduces a **biological-fidelity evaluation framework** that scores three biological properties those metrics miss, and applies it, for the first time in a peer-reviewed single-cell study, to NASA spaceflight data. Three established generators (CVAE, cfDiffusion, scGAN) are used only as a **testbed** to exercise the framework. The contribution is the evaluation and the application domain, not any model.

---

## 2. Framing: the two pillars (do not let these drift)

1. **Pillar 1, the evaluation framework (primary novelty).** A uniform, biology-aware protocol with three metrics absent from current scRNA-seq benchmarking practice:
   - Condition fidelity: top-50 DEG overlap (real vs synthetic flight-vs-ground contrast).
   - Mechanistic fidelity: fgsea GO-pathway overlap.
   - Phenotypic fidelity: random-forest feature-importance overlap under a train-on-synthetic, test-on-real (TSTR) design.
2. **Pillar 2, the spaceflight application (milestone, not a model flex).** First peer-reviewed application of multi-conditional deep generative augmentation plus biology-aware evaluation to NASA OSDR spaceflight scRNA-seq.

**The models are a testbed.** CVAE, cfDiffusion (Zhang et al., Brief Bioinform 2025), and scGAN (Marouf et al., Nat Commun 2020) are existing, published methods. The paper neither proposes a new generator nor crowns a best one. Where scGAN leads on biological fidelity, that is reported as a **scoping observation on small-sample, multi-conditional data**, never as a general ranking. This de-emphasis is deliberate and was requested explicitly; keep it.

---

## 3. Framing guardrails and known risks (highest priority)

These are the things most likely to be gotten wrong by a fresh agent.

| Guardrail | Why it matters | Current state in the .tex |
|---|---|---|
| **Do not over-claim "first ever."** A 2024 NTRS poster (Golts, Casaletto & Costes) already applied a CVAE to spaceflight snRNA-seq. Casaletto is also a co-author here, so it is the team's own prior gray-lit work. | Reviewer will find it; over-claiming is the fastest credibility hit. | Abstract/intro currently say "first application ... to spaceflight single-cell transcriptomics." **Needs tightening to "first peer-reviewed, first multi-conditional multi-architecture benchmark on NASA spaceflight scRNA-seq."** Flag, do not silently leave as "first ever." |
| **TSTR is AUC-maximizing, not AUC to 0.5.** The scRNA-seq generative field convention (classifier two-sample test) treats AUC near 0.5 (indistinguishability) as the optimum. This project inverts it: high agreement is the goal. | If not stated explicitly, reviewers will think the metric is backwards. | The Methods already contrast it with the classifier two-sample test. Keep and, if anything, sharpen. |
| **Matched-MMD comparison was dropped.** Earlier project notes describe a dual analysis (compare at matched MMD and at converged best MMD). The author decided to drop the matched-MMD arm. | A fresh agent reading old notes may re-introduce it. **Do not.** Only converged-MMD comparison is used. | Already removed from the .tex. |
| **Lead with feature-importance overlap, not TSTR AUC.** Absolute TSTR AUC is modest/poor. | Over-selling AUC invites a correct reviewer attack. | Phrased as preserved predictive-gene structure, AUC reported as secondary. Keep. |
| **Sparsity gap is a stated limitation, not hidden.** All models produce denser-than-real data (diffusion densest). | Honesty is expected and protects the paper. | Present in Results and Discussion. Keep. |
| **MMD plateaus before biology is fully learned.** Early stopping on MMD alone underperformed; this shaped the 30-epoch-patience design. | The MMD-stopping rationale must be "a fair common operating point," with the caveat that MMD convergence is not proof of full biological learning. Do not present MMD as a biology oracle. | Handled implicitly; worth a sentence if a reviewer probes. |
| **Style rules (author-enforced).** No em dashes anywhere in prose. No arrow symbols in prose (write "to" or "approaching"). Substantive section headings, not generic. Avoid vague jargon like "portable framework" unless defined. | The author corrects these every round. | The current .tex already complies. Preserve on every edit. |

---

## 4. The evaluation framework in detail (the heart of the paper)

Four layers. One statistical gate, then three biological layers. All metrics computed on **held-out test data** to avoid optimistic assessment. For every dataset and model, a full **1:1 synthetic replicate** of the real dataset is generated, matching the exact distribution of mission parameters.

### Layer 1, statistical fidelity (the gate, not the point)
- **Primary: MMD at convergence**, computed in a **frozen 50-component PCA space**.
- Complementary distances: cosine, Wasserstein (Sinkhorn-approximated), mean absolute deviation of per-gene means, Mahalanobis (pooled covariance).
- Purpose: establish the generators reached comparable distributional quality so biological differences are not just unequal training.

### Layer 2, condition fidelity
- t-test DEG between flight and ground control, computed separately in real and synthetic data.
- Rank genes by **absolute log fold change (unsigned)**; take **top 50**; measure overlap of gene identity.
- `N=50` is a convention; the metric is defined for any `N` (report sensitivity if asked). **Directional/signed DEG is explicitly future work, not this paper.**

### Layer 3, mechanistic fidelity
- fgsea preranked GSEA against GO gene sets, separately on real and synthetic.
- Measure overlap of significantly enriched pathways.
- Rationale: two datasets can share function without sharing exact gene identities; pathway overlap typically exceeds gene-level overlap (a central, reportable finding).

### Layer 4, phenotypic fidelity (TSTR)
- Train a random forest to classify flight vs ground on **synthetic** data, and another on **real** data.
- Compare the **top-50 most important genes** by overlap (this is the headline phenotypic number).
- TSTR = train-on-synthetic, test-on-real. **AUC-maximizing** (high = good), the opposite of the classifier two-sample test. AUC is reported but is modest, so overlap leads.

### MMD-stopping rule (shared, architecture-agnostic)
- Fit PCA **once on real training data only**; freeze it. Never refit on synthetic or pooled data (refitting makes MMD non-comparable across models/checkpoints).
- Each interval: generate a synthetic replicate of the validation set, project both real-val and synthetic through the frozen PCA, compute MMD with an RBF kernel.
- Kernel bandwidth sigma by the **median heuristic** with **proportional stratified sampling** on training data.
- Stop when MMD does not improve over a **30-epoch window**; otherwise keep training.
- Why external: ELBO, adversarial (WGAN), and denoising losses are incommensurable; GAN loss in particular does not track sample quality, so an external metric is required.

---

## 5. The three models (testbed)

| Model | Family | Framework | Conditioning | Notes for the agent |
|---|---|---|---|---|
| CVAE | VAE | PyTorch | Concatenated one-hot/embedded covariates | Latent ~50-dim; beta ~0.05 (from code, confirm). Standard conditional VAE. |
| cfDiffusion | Latent diffusion + classifier-free guidance | PyTorch | Separate embedders for categorical + continuous, combined by projection | First-stage VAE to 128-dim latent, T=1000, p_drop=0.15 (from code, confirm). Based on Zhang et al. 2025. |
| scGAN | WGAN-GP + library-size normalization | TensorFlow | **Extended from original single categorical (cell type) to the full covariate vector; exact injection mechanism UNKNOWN** | Marouf et al. 2020 cscGAN base. The multi-conditional extension is plumbing for a fair comparison, not a modeling contribution. |

**All three condition on an identical covariate vector** (experimental condition, strain, sex, age at launch, mission duration). This symmetry is the fairness guarantee and should be stated plainly. The scGAN paper is **not** in the project files; describe from canonical Marouf et al. 2020 until the author confirms the conditioning mechanism.

---

## 6. Dataset panel (confirm before submission)

Current roster in Table 1 (all cell counts, tissues, conditions are placeholders):

- Mouse brain, **pooled** OSD-612 / OSD-613 / OSD-352.
- OSD-402, OSD-403, OSD-404, OSD-405 (individual studies).
- OSDR **pooled** cohort (distinct from the pooled brain set? **confirm**).
- CANCER IMMU 10x, out-of-domain robustness control, metastatic vs non-metastatic immune cells, ~120,000+ cells (**confirm count**).

Open question: exact total count and whether both pooled sets are distinct rows. Prior notes phrase it as "5 OSDR + 1 cancer"; the current table lists more rows. Reconcile with the author.

Preprocessing (standard, confirm thresholds): Cell Ranger counts, per-cell QC, library-size normalize to 1e4, log(x+1), remove low-expression genes (a 0.997 / 3-sigma cutoff was used), top 2000 HVGs (Seurat VST), 80/10/10 train/val/test stratified by condition.

---

## 7. Current document state (section by section)

Single file `manuscript.tex`. Mechanics built in:
- `\protname` command renames the protocol globally (currently "biological-fidelity protocol"; no branded acronym chosen yet).
- `\res{...}` renders **red** = every value/decision still to fill. Find-and-replace target.
- `\figplaceholder{}{}` renders a **grey box** = six figure slots; the real `\includegraphics` line is commented directly beneath each, uncomment to swap in art. The file compiles with zero image files present.
- Bibliography is embedded `thebibliography` (no external .bib; compiles in one pass set). ~20 refs.

| Section | Status |
|---|---|
| Title, authors, affiliations | Done. Arsh Suri lead/corresponding. Confirm final author list/order. |
| Abstract | Done, but tighten the "first application" wording (see guardrails). |
| Introduction (4 substantive subsections + Contributions) | Done. Standalone (no reference to any prior manuscript). |
| Results (testbed, MMD stopping, sparsity, 3 biological layers, out-of-domain) | Prose done; **all numbers are `\res{}` placeholders**; Fig 1 (framework, TikZ) drawn, Figs 2-6 are placeholder boxes. |
| Tables 1-3 | Structured; **all cells placeholders**. |
| Discussion (5 subsections incl. limitations) | Done. |
| Methods (datasets, shared conditioning, 3 model subsections, MMD rule, protocol, distances, training, availability) | Done; hyperparameters and scGAN mechanism are `\res{}`. |
| Acknowledgements / contributions / competing interests | Placeholder text. |

The whole thing currently compiles to ~13 pages.

---

## 8. What is still needed (actionable backlog)

### Tier 1, blocking (the scientific core)
Deliver as one CSV/spreadsheet: rows = dataset x model, columns = metrics.
- **Table 1 fields:** confirmed accessions, tissue, cell count (n), number of conditions (k) for every dataset.
- **Table 2 (statistical), per model x dataset:** MMD at convergence; cosine, Wasserstein, MAD, Mahalanobis (whichever were actually computed); TSTR AUC.
- **Table 3 (biological), per model x dataset, as percentages:** top-50 DEG overlap; fgsea pathway overlap; RF predictive-gene (feature-importance) overlap.

### Tier 2, figure image files (PNG or, preferably, vector PDF)
- Fig 2 MMD convergence (MMD vs epoch, one line per model, stop point marked).
- Fig 3 UMAP (real vs synthetic, paneled by model, colored by condition and by real/synthetic).
- Fig 4 gene overlap (bars: DEG overlap and predictive-gene overlap across datasets/models).
- Fig 5 fgsea (enriched-pathway overlap, heatmap or bars).
- Fig 6 sparsity (zero-fraction distribution, real vs each model).

### Tier 3, short inline numbers (prose, not tables)
- `N` = number of spaceflight configurations (abstract + Results).
- Sparsity: real percent-zeros, and percent-zeros for cfDiffusion, CVAE, scGAN.
- Headline overlap: best top-gene overlap value, which model, on how many of `N` datasets it led.
- 2 to 4 named spaceflight pathways fgsea actually recovered.
- scGAN conditioning-injection mechanism (one line).

### Tier 4, Methods hyperparameters (fill or confirm)
- scGAN learning rate, gradient-penalty weight; batch sizes per model; epochs to convergence; classifier-free guidance strength (gamma); DDIM sampling steps; exact HVG/low-expression threshold. (Latent dims, T=1000, p_drop=0.15, beta~0.05 pulled from code; confirm.)

### Tier 5, admin
- Repository URL; grant numbers / VIP acknowledgement; final author-contributions sentence; confirm author list/order.

---

## 9. Prior art and citations to address

Must be cited and positioned (some not yet in the .tex bibliography):
- **NTRS poster (Golts, Casaletto & Costes, 2024)** — CVAE on spaceflight snRNA-seq. The reason "first ever" is not defensible. Position as prior gray-lit from the same group.
- **GLARE (npj Microgravity 2025, DOI 10.1038/s41526-025-00525-5)** — closest same-venue precedent. Add and contrast.
- **GAN-based NASA OSDR biomarker paper (Neuroinformatics 2025, DOI 10.1007/s12021-025-09729-2)** — non-single-cell comparator reviewers may raise. The current intro references an "unpublished bulk RNA-seq GAN" generically; replace that with this specific citation.
- Core method refs already in the .tex: scGAN (Marouf 2020), cfDiffusion (Zhang 2025), scDiffusion (Luo 2024), scVI (Lopez 2018), ACTIVA (Heydari 2022), CFGen (Palma 2025), scGFT (Tedesco 2025), MMD (Gretton 2012; Sutherland 2017), fgsea (Korotkevich 2021), TSTR (Esteban 2017), classifier two-sample test (Lopez-Paz 2017), OSDR (Gebre 2024), Twins Study (Garrett-Bakelman 2019), Inspiration4 (Overbey 2024), NASA AI roadmap (Scott 2023; Ilangovan 2024), Seurat (Hafemeister 2019), 10x (Zheng 2017).

**The fgsea pathway-overlap metric and the utility-maximizing TSTR design are, per the author's review, absent from all major scRNA-seq generative benchmarking papers.** That absence is the crux of the methodological novelty; keep it front and center.

---

## 10. Known tensions / open decisions (surface to the author, do not silently resolve)

1. "First application" wording vs the NTRS poster (use "first peer-reviewed"). **Highest priority.**
2. Author list/order (confirm; the lead was changed to Arsh Suri in-session).
3. Dataset roster: exact count and the two pooled sets.
4. scGAN conditioning mechanism (unknown).
5. Whether to give the protocol a branded name via `\protname` (currently neutral).
6. Which distance metrics were actually computed (drop any that were not).

---

## 11. How the author works (match this)

- Wants direct, honest assessment over validation. Explicitly asked to stress-test novelty claims with gray literature before submission. Do not soften real weaknesses.
- Iterates in multiple correction rounds; expects precise style compliance (no em dashes, no arrow symbols in prose, substantive headings, no empty jargon).
- Treats the evaluation framework and the application as the story; keeps the models in a supporting role. Any drift back toward "cross-architecture benchmark" or "best model" framing will be corrected.

---

## 12. File pointers

- `manuscript.tex` / `manuscript.pdf` — the deliverable (outputs folder).
- Reference PDFs in the project: cfDiffusion (`bbaf071`), scDiffusion (`btae518`). scGAN paper is **not** present.
- There is an earlier base manuscript in the project, but the paper is written **standalone**; do not reference it in the text.
- Project memory (framing, methodology insights, prior art) lives under the project's notes and is the source for sections 2 through 10 above.
