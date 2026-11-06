# RF classifier accuracy vs. class-composition baseline

`lift` = RF accuracy − majority-class share (percentage points). A value near 0 with low F1 means the
classifier is effectively predicting the majority class only — accuracy is an artifact of class imbalance.
`Real` = train-on-real / test-on-real ceiling; others are TSTR (train-on-synthetic, test-on-real).

## OSDR-402 — Spaceflight vs Ground Control  (n=1,534)
Classes: **Ground Control 30.0%** / **Spaceflight 70.0%**  →  majority baseline = **70.0%** (Spaceflight)

| Train source | RF acc (%) | ±std | F1 | lift vs majority (pts) | beats baseline? |
|---|---|---|---|---|---|
| Real (ceiling) | 69.9 | 0.3 | 0.014 | -0.1 | ✗ (~majority) |
| CVAE | 93.2 | 1.4 | 0.881 | +23.2 | ✓ |
| cfDiffusion | 70.1 | 0.1 | 0.003 | +0.0 | ✗ (~majority) |
| scGAN | 70.7 | 1.0 | 0.108 | +0.7 | ✗ (~majority) |

## OSDR-403 — Spaceflight vs Ground Control  (n=1,561)
Classes: **Ground Control 47.3%** / **Spaceflight 52.7%**  →  majority baseline = **52.7%** (Spaceflight)

| Train source | RF acc (%) | ±std | F1 | lift vs majority (pts) | beats baseline? |
|---|---|---|---|---|---|
| Real (ceiling) | 54.0 | 2.3 | 0.381 | +1.4 | ✓ |
| CVAE | 63.7 | 2.4 | 0.610 | +11.0 | ✓ |
| cfDiffusion | 53.9 | 2.8 | 0.428 | +1.2 | ✓ |
| scGAN | 62.3 | 2.1 | 0.520 | +9.6 | ✓ |

## OSDR-404 — Spaceflight vs Ground Control  (n=14,045)
Classes: **Ground Control 38.5%** / **Spaceflight 61.5%**  →  majority baseline = **61.5%** (Spaceflight)

| Train source | RF acc (%) | ±std | F1 | lift vs majority (pts) | beats baseline? |
|---|---|---|---|---|---|
| Real (ceiling) | 62.8 | 0.3 | 0.088 | +1.3 | ✓ |
| CVAE | 77.7 | 0.4 | 0.645 | +16.2 | ✓ |
| cfDiffusion | 61.5 | 0.2 | 0.011 | -0.1 | ✗ (~majority) |
| scGAN | 62.9 | 0.5 | 0.099 | +1.4 | ✓ |

## OSDR-405 — Spaceflight vs Ground Control  (n=3,384)
Classes: **Ground Control 35.3%** / **Spaceflight 64.7%**  →  majority baseline = **64.7%** (Spaceflight)

| Train source | RF acc (%) | ±std | F1 | lift vs majority (pts) | beats baseline? |
|---|---|---|---|---|---|
| Real (ceiling) | 65.0 | 0.2 | 0.017 | +0.3 | ✗ (~majority) |
| CVAE | 79.1 | 1.2 | 0.595 | +14.5 | ✓ |
| cfDiffusion | 64.8 | 0.3 | 0.024 | +0.1 | ✗ (~majority) |
| scGAN | 65.6 | 0.4 | 0.074 | +0.9 | ✗ (~majority) |

## OSD-352/612/613 — Spaceflight vs Ground Control  (n=213,854)
Classes: **Ground Control 54.2%** / **Spaceflight 45.8%**  →  majority baseline = **54.2%** (Ground Control)

| Train source | RF acc (%) | ±std | F1 | lift vs majority (pts) | beats baseline? |
|---|---|---|---|---|---|
| Real (ceiling) | 69.1 | 0.2 | 0.754 | +14.9 | ✓ |
| CVAE | 80.1 | 0.2 | 0.811 | +25.8 | ✓ |
| cfDiffusion | 81.0 | 0.2 | 0.842 | +26.8 | ✓ |
| scGAN | 77.1 | 0.2 | 0.809 | +22.8 | ✓ |

## GSE164522 — Metastasis Tumor vs Normal  (n=96,023)
Classes: **metastasis normal 61.1%** / **metastasis tumor 38.9%**  →  majority baseline = **61.1%** (metastasis normal)

| Train source | RF acc (%) | ±std | F1 | lift vs majority (pts) | beats baseline? |
|---|---|---|---|---|---|
| Real (ceiling) | 81.2 | 0.2 | 0.729 | +20.1 | ✓ |
| CVAE | 95.9 | 0.2 | 0.945 | +34.8 | ✓ |
| cfDiffusion | 85.3 | 0.2 | 0.800 | +24.2 | ✓ |
| scGAN | 86.8 | 0.4 | 0.809 | +25.7 | ✓ |
