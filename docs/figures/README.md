# Figures

All 23 figures from the submitted notebook
(`notebooks/original_submission.ipynb`), extracted so they can be viewed without
opening it. Filenames describe what each shows; `per_method_*` covers the 30
individual models, `pooled_*` the model trained on all methods together.

| Figure | Shows |
|---|---|
| `test_comparison_r2.png` | **Headline figure.** R² per model, per dataset, on held-out test data |
| `test_comparison_mae.png` | Same, mean absolute error, with the dummy baseline |
| `test_comparison_mse.png` | Same, mean squared error |
| `cv_comparison_r2.png` | R² from nested cross-validation (training half only) |
| `cv_comparison_mae.png` | MAE from nested cross-validation |
| `cv_comparison_mse.png` | MSE from nested cross-validation |
| `pooled_rf_test_predicted_vs_true.png` | Random Forest predictions vs. measurement, pooled test set |
| `pooled_ridge_test_predicted_vs_true.png` | Ridge, same |
| `pooled_lasso_test_predicted_vs_true.png` | Lasso, same |
| `pooled_rf_cv_feature_importance.png` | Random Forest feature importance, pooled model |
| `pooled_ridge_cv_top_features.png` | Ridge standardised coefficients, pooled model |
| `pooled_lasso_cv_zero_coefficients.png` | Features Lasso eliminates, pooled model |
| `pooled_rf_cv_predicted_vs_true.png` | Random Forest fit on the training half |
| `pooled_ridge_cv_predicted_vs_true.png` | Ridge, same |
| `pooled_lasso_cv_predicted_vs_true.png` | Lasso, same |
| `per_method_rf_r2.png` | Random Forest R² across the 30 individual methods |
| `per_method_ridge_r2.png` | Ridge, same |
| `per_method_lasso_r2.png` | Lasso, same |
| `per_method_rf_mean_feature_importance.png` | Which descriptors matter, averaged over methods |
| `per_method_rf_importance_heatmap.png` | Descriptor importance per method — LogP dominates throughout |
| `per_method_ridge_mean_abs_coefficients.png` | Ridge coefficients averaged over methods |
| `per_method_lasso_zero_coefficient_frequency.png` | How often Lasso eliminates each descriptor |
| `per_method_ridge_coefficient_heatmap_KNOWN_BUG.png` | ⚠️ See below |

## ⚠️ `per_method_ridge_coefficient_heatmap_KNOWN_BUG.png`

Every column of this heatmap is identical, because the loop that built it never
advanced its slice index — all 31 columns show the first dataset's five folds. It
appears to demonstrate that feature importance is stable across chromatographic
conditions; it demonstrates nothing of the kind. Kept here because it is part of
the submitted record. See
[`../corrections.md`](../corrections.md#3-the-ridge-coefficient-heatmap-showed-one-dataset-thirty-times).

The Random Forest equivalent, `per_method_rf_importance_heatmap.png`, was built
differently and is correct — it does vary across methods, and is the honest
version of the same comparison.
