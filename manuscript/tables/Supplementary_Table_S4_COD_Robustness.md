**Panel A.** SHAP feature rankings when salinity is included as a 7th predictor.

| Driver         |   Mean_Abs_SHAP |   Mean_SHAP |   SHAP_Corr_r | Direction   |
|:---------------|----------------:|------------:|--------------:|:------------|
| Urban PD       |           1.543 |      -0.026 |         0.855 | Positive    |
| Rice PD        |           0.787 |       0.133 |        -0.695 | Negative    |
| Aquaculture ED |           0.727 |      -0.122 |         0.427 | Positive    |
| Urban %        |           0.661 |      -0.001 |         0.785 | Positive    |
| Salinity       |           0.603 |      -0.038 |         0.632 | Positive    |
| Rice %         |           0.385 |       0.057 |        -0.650 | Negative    |
| Aquaculture %  |           0.161 |       0.064 |        -0.351 | Negative    |

**Panel B.** SHAP rankings across three alternative COD response formulations (concentration, flow-normalized residual, and load).

| Response                 |   Rank | Driver         |   Mean_Abs_SHAP |   Mean_SHAP |   SHAP_Corr_r | Direction   |
|:-------------------------|-------:|:---------------|----------------:|------------:|--------------:|:------------|
| Concentration (mg/L)     |      1 | Urban PD       |           1.551 |      -0.092 |         0.800 | Positive    |
| Concentration (mg/L)     |      2 | Rice PD        |           0.882 |       0.117 |        -0.800 | Negative    |
| Concentration (mg/L)     |      3 | Urban %        |           0.726 |       0.059 |         0.774 | Positive    |
| Concentration (mg/L)     |      4 | Aquaculture ED |           0.511 |      -0.018 |         0.459 | Positive    |
| Concentration (mg/L)     |      5 | Rice %         |           0.494 |       0.037 |        -0.662 | Negative    |
| Concentration (mg/L)     |      6 | Aquaculture %  |           0.154 |       0.007 |         0.211 | Positive    |
| Load (tons/day)          |      1 | Rice %         |         193.440 |     -13.534 |         0.093 | Positive    |
| Load (tons/day)          |      2 | Urban PD       |         161.229 |      -6.899 |         0.762 | Positive    |
| Load (tons/day)          |      3 | Aquaculture ED |          99.075 |      -5.091 |         0.369 | Positive    |
| Load (tons/day)          |      4 | Urban %        |          67.779 |      11.510 |         0.647 | Positive    |
| Load (tons/day)          |      5 | Aquaculture %  |          60.882 |       8.848 |        -0.572 | Negative    |
| Load (tons/day)          |      6 | Rice PD        |          38.700 |       9.682 |        -0.199 | Negative    |
| Flow-normalized residual |      1 | Urban PD       |           1.423 |      -0.111 |         0.766 | Positive    |
| Flow-normalized residual |      2 | Aquaculture ED |           0.674 |      -0.067 |         0.356 | Positive    |
| Flow-normalized residual |      3 | Rice %         |           0.629 |       0.014 |        -0.249 | Negative    |
| Flow-normalized residual |      4 | Rice PD        |           0.531 |       0.102 |        -0.694 | Negative    |
| Flow-normalized residual |      5 | Urban %        |           0.500 |       0.060 |         0.685 | Positive    |
| Flow-normalized residual |      6 | Aquaculture %  |           0.219 |       0.045 |        -0.463 | Negative    |
: Supplementary Table S4. COD attribution robustness checks addressing two potential confounds. Direction is defined by the sign of the Pearson correlation $r(X_j, \phi_j)$ between feature values and SHAP values. Panel A tests whether mixing-zone salinity confounds LULC attribution by including salinity as a 7th predictor alongside six LULC metrics; Urban PD remains the top driver (mean $|\text{SHAP}|$ = 1.54) while salinity ranks 5th (0.60), indicating that the urbanization--COD association persists independently of tidal mixing gradients. Panel B tests sensitivity to response metric choice by comparing SHAP rankings across three COD formulations: concentration (mg L$^{-1}$), load (tons day$^{-1}$, computed using branch discharge), and flow-normalized residual ($C - \hat{C}|_Q$). Urban PD ranked in the top two positions across all formulations, demonstrating that attribution conclusions are not artifacts of the response metric. {#tbl-s4-cod-robustness}
