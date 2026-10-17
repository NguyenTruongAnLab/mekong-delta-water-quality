| LULC Driver    |   RDA1 |   RDA2 |   Vector Length |    VIF |
|:---------------|-------:|-------:|----------------:|-------:|
| Urban PD       |  0.219 | -0.899 |           0.925 |  7.822 |
| Rice PD        | -0.852 | -0.343 |           0.918 |  1.456 |
| Aquaculture %  | -0.144 | -0.848 |           0.860 | 35.609 |
| Aquaculture ED | -0.171 | -0.832 |           0.850 | 48.749 |
| Urban %        |  0.212 | -0.504 |           0.547 |  5.328 |
| Rice %         | -0.281 | -0.309 |           0.418 |  1.622 |
: Supplementary Table S1. Redundancy analysis (RDA) biplot scores and variance inflation factors (VIF) for six LULC predictors. The two configuration metrics with the longest biplot vectors (Aquaculture edge density [ED] and Urban patch density [PD]) exhibit persistent multicollinearity (VIF = 91.4 and 73.9, respectively), exceeding the conventional threshold of 10. This collinearity reflects the inherent correlation between composition and configuration within the same land-cover class: fragmentation necessarily scales with areal extent. These results motivate the orthogonalization sensitivity test in @tbl-s2-shap; if SHAP rankings shifted substantially after residualizing configuration on composition, the baseline attribution would be unreliable. {#tbl-s1-screening}
