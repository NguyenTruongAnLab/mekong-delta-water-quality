**Panel A.** SHAP feature rankings when salinity is included as a 7th predictor.

| Driver         |   Mean_Abs_SHAP |   Mean_SHAP |   SHAP_Corr_r | Direction   |
|:---------------|----------------:|------------:|--------------:|:------------|
| Urban PD       |           1.545 |      -0.027 |         0.856 | Positive    |
| Rice PD        |           0.785 |       0.133 |        -0.695 | Negative    |
| Aquaculture ED |           0.727 |      -0.121 |         0.428 | Positive    |
| Urban %        |           0.659 |      -0.002 |         0.783 | Positive    |
| Salinity       |           0.605 |      -0.038 |         0.631 | Positive    |
| Rice %         |           0.385 |       0.059 |        -0.648 | Negative    |
| Aquaculture %  |           0.162 |       0.063 |        -0.358 | Negative    |

**Panel B.** SHAP rankings across three alternative COD response formulations (concentration, flow-normalized residual, and load).

| Response                 |   Rank | Driver         |   Mean_Abs_SHAP |   Mean_SHAP |   SHAP_Corr_r | Direction   |
|:-------------------------|-------:|:---------------|----------------:|------------:|--------------:|:------------|
| Concentration (mg/L)     |      1 | Urban PD       |           1.547 |      -0.088 |         0.799 | Positive    |
| Concentration (mg/L)     |      2 | Rice PD        |           0.886 |       0.118 |        -0.800 | Negative    |
| Concentration (mg/L)     |      3 | Urban %        |           0.727 |       0.056 |         0.778 | Positive    |
| Concentration (mg/L)     |      4 | Aquaculture ED |           0.520 |      -0.019 |         0.476 | Positive    |
| Concentration (mg/L)     |      5 | Rice %         |           0.495 |       0.037 |        -0.663 | Negative    |
| Concentration (mg/L)     |      6 | Aquaculture %  |           0.151 |       0.005 |         0.193 | Positive    |
| Load (tons/day)          |      1 | Rice %         |         193.496 |     -13.745 |         0.092 | Positive    |
| Load (tons/day)          |      2 | Urban PD       |         161.566 |      -6.735 |         0.763 | Positive    |
| Load (tons/day)          |      3 | Aquaculture ED |          98.721 |      -4.128 |         0.373 | Positive    |
| Load (tons/day)          |      4 | Urban %        |          67.825 |      11.608 |         0.648 | Positive    |
| Load (tons/day)          |      5 | Aquaculture %  |          60.960 |       8.158 |        -0.576 | Negative    |
| Load (tons/day)          |      6 | Rice PD        |          38.786 |       9.883 |        -0.197 | Negative    |
| Flow-normalized residual |      1 | Urban PD       |           1.423 |      -0.113 |         0.766 | Positive    |
| Flow-normalized residual |      2 | Aquaculture ED |           0.682 |      -0.065 |         0.356 | Positive    |
| Flow-normalized residual |      3 | Rice %         |           0.629 |       0.014 |        -0.251 | Negative    |
| Flow-normalized residual |      4 | Rice PD        |           0.530 |       0.102 |        -0.695 | Negative    |
| Flow-normalized residual |      5 | Urban %        |           0.497 |       0.059 |         0.682 | Positive    |
| Flow-normalized residual |      6 | Aquaculture %  |           0.221 |       0.043 |        -0.482 | Negative    |
: Supplementary Table S4. COD attribution robustness checks addressing two potential confounds. Direction is defined by the sign of the Pearson correlation $r(X_j, \phi_j)$ between feature values and SHAP values. Panel A tests whether mixing-zone salinity confounds LULC attribution by including salinity as a 7th predictor alongside six LULC metrics; Urban PD remains the top driver (mean $|\text{SHAP}|$ = 1.54) while salinity ranks 5th (0.60), indicating that the urbanization--COD association persists independently of tidal mixing gradients. Panel B tests sensitivity to response metric choice by comparing SHAP rankings across three COD formulations: concentration (mg L$^{-1}$), load (tons day$^{-1}$, computed using branch discharge), and flow-normalized residual ($C - \hat{C}|_Q$). Urban PD ranked in the top two positions across all formulations, demonstrating that attribution conclusions are not artifacts of the response metric. {#tbl-s4-cod-robustness}
