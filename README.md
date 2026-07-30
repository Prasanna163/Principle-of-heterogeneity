# Interaction Heterogeneity and Supramolecular Stability

Computational analysis of how interaction heterogeneity relates to the stability of non-covalent molecular complexes.

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Research Question

Many molecular-design strategies focus on maximizing the strength of one dominant interaction. This project tests a broader hypothesis: stability may also depend on the distribution and heterogeneity of interactions across a complex.

The analysis uses the Kulkarni-NCI Fingerprint (KNF) to separate average interaction character from variation within the non-covalent interaction field.

## Dataset and Model

The study analyzes 2,649 molecular complexes. The main regression model includes interaction-strength, heterogeneity, electronic, and geometric terms.

| Quantity | Reported value |
| --- | ---: |
| Molecular complexes | 2,649 |
| Model R2 | 0.515 |
| Mean interaction term, f7 | -0.0157 |
| Residual heterogeneity term | -0.0053 |
| Quadratic heterogeneity term | +0.00235 |
| Wiberg bond-order term, f3 | +0.0185 |

The positive quadratic term indicates a nonlinear relationship between heterogeneity and the target property within this dataset. It does not by itself establish a universal design law.

## Interpretation

The current evidence supports three cautious conclusions:

1. Interaction heterogeneity contains information that is not fully captured by average interaction strength.
2. Its relationship with stability appears nonlinear in the analyzed dataset.
3. Heterogeneity may be useful as a design variable when evaluated together with geometry, electronic structure, and interaction strength.

These findings should be tested on additional molecular families before broad generalization.

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── data/
│   ├── raw/
│   └── processed/
└── Scripts/
```

The repository contains the data and analysis scripts used to evaluate the proposed relationship.

## Reproduction

1. Create a Python environment.
2. Install the required scientific Python packages.
3. Confirm the input paths used by the scripts.
4. Run data preparation before statistical analysis.
5. Compare generated coefficients, diagnostics, and figures with the reported outputs.

Common dependencies include:

```bash
pip install numpy pandas scipy scikit-learn matplotlib statsmodels
```

Exact requirements may vary between scripts.

## Scope and Limitations

- The analysis is observational and dataset-dependent.
- Regression coefficients depend on feature scaling and model specification.
- Correlation does not prove a direct causal mechanism.
- The current dataset is dominated by specific classes of non-covalent complexes.
- Independent validation on additional benchmarks is required.

## Related Work

- [KNF Predictor](https://github.com/Prasanna163/KNF-Predictor)
- [JCIM workflow code](https://github.com/Prasanna163/Supramolecular-Stability-JCIM-Code)
- [Geonit](https://github.com/Prasanna163/Geoinit)

## Citation Status

This repository documents an ongoing research direction. Use the final journal citation when a corresponding peer-reviewed paper becomes available. Until then, cite the repository with the commit hash used in the analysis.

## License

The code in this repository is available under the MIT License. Dataset reuse may be subject to the terms of the original data sources.

## Contact

Prasanna P. Kulkarni  
Institute of Chemical Technology, Mumbai, Marathwada Campus  
Email: prasannakulkarni163@gmail.com
