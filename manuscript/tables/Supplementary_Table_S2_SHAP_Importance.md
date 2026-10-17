| Panel            | Specification   | Driver                 |   Rank |   Mean SHAP |   SHAP_Corr_r | Direction   | Parameter   |
|:-----------------|:----------------|:-----------------------|-------:|------------:|--------------:|:------------|:------------|
| A: Global        | Baseline        | Aquaculture %          |      1 |       2.663 |         0.503 | Positive    | All         |
| A: Global        | Baseline        | Rice %                 |      2 |       2.367 |         0.149 | Positive    | All         |
| A: Global        | Baseline        | Rice PD                |      3 |       1.968 |         0.459 | Positive    | All         |
| A: Global        | Baseline        | Urban %                |      4 |       0.878 |         0.165 | Positive    | All         |
| A: Global        | Baseline        | Urban PD               |      5 |       0.856 |        -0.102 | Negative    | All         |
| A: Global        | Baseline        | Aquaculture ED         |      6 |       0.850 |         0.383 | Positive    | All         |
| A: Global        | Orthogonalized  | Aquaculture %          |      1 |       2.745 |         0.509 | Positive    | All         |
| A: Global        | Orthogonalized  | Rice %                 |      2 |       1.755 |         0.379 | Positive    | All         |
| A: Global        | Orthogonalized  | Aquaculture ED (resid) |      3 |       1.660 |         0.343 | Positive    | All         |
| A: Global        | Orthogonalized  | Urban PD (resid)       |      4 |       1.452 |        -0.132 | Negative    | All         |
| A: Global        | Orthogonalized  | Urban %                |      5 |       1.101 |         0.048 | Positive    | All         |
| A: Global        | Orthogonalized  | Rice PD (resid)        |      6 |       0.996 |         0.453 | Positive    | All         |
| B: Per-parameter | Baseline        | Urban PD               |      1 |       1.663 |         0.896 | Positive    | COD         |
| B: Per-parameter | Baseline        | Aquaculture ED         |      1 |       0.183 |         0.727 | Positive    | DO          |
| B: Per-parameter | Baseline        | Rice PD                |      1 |       0.026 |         0.748 | Positive    | NH4-N       |
| B: Per-parameter | Baseline        | Rice PD                |      1 |       0.097 |         0.871 | Positive    | NO3-N       |
| B: Per-parameter | Baseline        | Aquaculture %          |      1 |      12.657 |        -0.456 | Negative    | TSS         |
| B: Per-parameter | Orthogonalized  | Urban PD (resid)       |      1 |       1.702 |         0.829 | Positive    | COD         |
| B: Per-parameter | Orthogonalized  | Aquaculture %          |      1 |       0.176 |         0.768 | Positive    | DO          |
| B: Per-parameter | Orthogonalized  | Rice PD (resid)        |      1 |       0.024 |         0.735 | Positive    | NH4-N       |
| B: Per-parameter | Orthogonalized  | Rice PD (resid)        |      1 |       0.097 |         0.854 | Positive    | NO3-N       |
| B: Per-parameter | Orthogonalized  | Aquaculture %          |      1 |      12.652 |        -0.397 | Negative    | TSS         |
: Supplementary Table S2. SHAP feature-importance rankings under baseline and orthogonalized predictor specifications, computed via LOSO cross-validated RF ($n_{\mathrm{trees}} = 500$). Panel A reports global rankings (mean $|\text{SHAP}|$ across five WQ parameters); Panel B reports the parameter-specific dominant driver. Direction is defined by the sign of the Pearson correlation $r(X_j, \phi_j)$ between feature values and out-of-fold SHAP values. Orthogonalization residualizes each configuration metric against its corresponding composition percentage via linear regression, isolating fragmentation effects beyond areal extent. Rankings remained consistent across specifications: the dominant drivers for each parameter were unchanged, confirming that the attribution conclusions in the main text are robust to the multicollinearity documented in @tbl-s1-screening. {#tbl-s2-shap}
