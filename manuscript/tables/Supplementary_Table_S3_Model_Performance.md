**Panel A.** Random Forest leave-one-station-out cross-validation (2013–2020 5-parameter complete-case baseline LULC-only model, $N = 120$).

| Parameter   |   OOB_R2_score |   CV_R2_LOSO |   Spearman_rho_LOSO |   RMSE_LOSO |   N_Perm_Significant |   N_samples |
|:------------|---------------:|-------------:|--------------------:|------------:|---------------------:|------------:|
| COD         |          0.123 |       -0.159 |               0.324 |       7.193 |                    6 |         120 |
| DO          |          0.275 |       -0.120 |               0.356 |       0.962 |                    6 |         120 |
| NH4-N       |         -0.007 |       -0.202 |               0.090 |       0.108 |                    6 |         120 |
| NO3-N       |          0.471 |        0.375 |               0.577 |       0.176 |                    5 |         120 |
| TSS         |          0.053 |       -0.369 |               0.146 |      62.549 |                    6 |         120 |

**Panel B.** Generalization comparison (2005–2020 Ben Tre station-year master, RF + Ridge, 6-feature baseline LULC vs. 11-feature primary augmented feature sets, LOSO metrics).

| Parameter   | Feature_Set   | Model   |   LOSO_R2 |   LOSO_RMSE |   LOSO_Spearman_rho |   N_samples |   N_features |
|:------------|:--------------|:--------|----------:|------------:|--------------------:|------------:|-------------:|
| COD         | Baseline      | RF      |     0.024 |       5.740 |               0.441 |         229 |            6 |
| COD         | Baseline      | Ridge   |     0.068 |       5.610 |               0.382 |         229 |            6 |
| COD         | Augmented     | RF      |     0.423 |       4.416 |               0.561 |         229 |           11 |
| COD         | Augmented     | Ridge   |     0.228 |       5.106 |               0.450 |         229 |           11 |
| DO          | Baseline      | RF      |     0.165 |       0.925 |               0.520 |         152 |            6 |
| DO          | Baseline      | Ridge   |     0.012 |       1.007 |               0.334 |         152 |            6 |
| DO          | Augmented     | RF      |     0.297 |       0.849 |               0.646 |         152 |           11 |
| DO          | Augmented     | Ridge   |     0.080 |       0.971 |               0.460 |         152 |           11 |
| NH4-N       | Baseline      | RF      |    -0.136 |       0.172 |               0.121 |         297 |            6 |
| NH4-N       | Baseline      | Ridge   |    -0.045 |       0.165 |              -0.044 |         297 |            6 |
| NH4-N       | Augmented     | RF      |    -0.038 |       0.165 |               0.250 |         297 |           11 |
| NH4-N       | Augmented     | Ridge   |    -0.062 |       0.167 |              -0.023 |         297 |           11 |
| NO3-N       | Baseline      | RF      |    -0.361 |       0.854 |               0.400 |         236 |            6 |
| NO3-N       | Baseline      | Ridge   |    -0.187 |       0.798 |               0.398 |         236 |            6 |
| NO3-N       | Augmented     | RF      |    -0.796 |       0.981 |               0.562 |         236 |           11 |
| NO3-N       | Augmented     | Ridge   |    -0.264 |       0.823 |               0.353 |         236 |           11 |
| TSS         | Baseline      | RF      |    -0.189 |     119.566 |               0.247 |         294 |            6 |
| TSS         | Baseline      | Ridge   |    -0.223 |     121.284 |              -0.141 |         294 |            6 |
| TSS         | Augmented     | RF      |    -0.198 |     120.031 |               0.317 |         294 |           11 |
| TSS         | Augmented     | Ridge   |    -0.266 |     123.409 |              -0.116 |         294 |           11 |

**Panel C.** COD variance partition (2005–2020 Ben Tre station-year master, $N = 229$: LULC-only [6] vs. LULC+Pop [7] vs. Hydro-only [4] vs. Combined [11]).

| Predictor_Set   |   LOSO_R2 |   LOSO_R2_RF |   LOSO_RMSE |   LOSO_Spearman_rho |   OLS_R2 |   N |   N_features |
|:----------------|----------:|-------------:|------------:|--------------------:|---------:|----:|-------------:|
| LULC-only       |     0.068 |        0.024 |       5.610 |               0.382 |    0.240 | 229 |            6 |
| LULC+Pop        |     0.131 |        0.061 |       5.416 |               0.400 |    0.301 | 229 |            7 |
| Hydro-only      |     0.259 |        0.460 |       5.002 |               0.420 |    0.299 | 229 |            4 |
| Combined        |     0.228 |        0.423 |       5.106 |               0.450 |    0.394 | 229 |           11 |

**Panel D.** RDA cross-network variance explained (RDA1+RDA2).

| Dataset               |   RDA1+RDA2 |
|:----------------------|------------:|
| Ben Tre (LULC only)   |       0.221 |
| Ben Tre (hydro-proxy) |       0.210 |
| MRC (LULC only)       |       0.308 |
: Supplementary Table S3. Model performance and generalization diagnostics. {#tbl-s3-performance}
