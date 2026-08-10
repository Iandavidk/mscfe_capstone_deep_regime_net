# Contributors

## Deep-RegimeNet: An Unsupervised Deep Clustering Framework
## for Adaptive Market Regime Identification

**MScFE 690 Capstone Project | Group 16019 | WorldQuant University | 2026**

---

| Name | Role | Contact |
|------|------|---------|
| **Ian Kihara Wangui** | Lead Developer — data pipeline, model architecture, training protocol, GitHub repository management, hyperparameter ablation studies, t-SNE/UMAP visualisations | eandavid6@gmail.com |
| **Grace Nmon Monanyun** | Baseline Models & Results — DTW-KMeans and Sequential AE+KMeans implementation, results tables, full draft assembly, peer review coordination | gracemonah7@gmail.com |
| **Immaculate Wanjiru Kimani** | Mathematical Formulation & Economic Analysis — methodology write-up, regime-conditioned return statistics, historical concordance interpretation, discussion section, presentation narrative | immaculate.kimani@aims.ac.rw |

---

## Acknowledgements

The authors thank Group 16029 (En Chong Lok & Kurt Keissinger McKenzie) for
their thorough and constructive peer review, which materially improved the
evaluation framework, baseline suite, and methodological rigour of this project.
Their identification of the circular evaluation problem, the missing HMM baseline,
and the novelty overstatement led directly to the most important design changes
in the final submission.

---

## External Code and Libraries

This project builds on the following open-source libraries:

- PyTorch (https://pytorch.org)
- scikit-learn (https://scikit-learn.org)
- hmmlearn (https://hmmlearn.readthedocs.io)
- tslearn (https://tslearn.readthedocs.io)
- umap-learn (https://umap-learn.readthedocs.io)
- yfinance (https://github.com/ranaroussi/yfinance)

The Deep Embedded Clustering (DEC) implementation is adapted from:
> Xie, J., Girshick, R., & Farhadi, A. (2016). Unsupervised Deep Embedding for
> Clustering Analysis. ICML 2016, pp. 478–487.

The IDEC joint-loss design (retaining reconstruction loss during joint training) follows:
> Guo, X., et al. (2017). Improved Deep Embedded Clustering with Local Structure
> Preservation. IJCAI 2017, pp. 1753–1759.

The LayerNorm recommendation between recurrent layers follows:
> Ba, J. L., et al. (2016). Layer Normalization. arXiv:1607.06450.
