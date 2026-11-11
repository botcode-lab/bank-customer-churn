# Bank Customer Churn: Logistic Regression, Decision Trees and Random Forests

A portfolio study of churn classification, leakage-safe preprocessing, feature engineering, model comparison, probability assessment and threshold selection, with MLflow experiment tracking.


## Dataset access

Download the Bank Customer Churn Dataset from:
https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data

Place the CSV at `Data/Raw/bank_churn.csv`.

The dataset is not bundled because its redistribution terms are unclear.


## Outcome

The selected random forest achieved **0.8574 ROC-AUC**, **0.6980 average precision**, and **0.6258 F1** on 2,000 held-out test customers. At a validation-selected threshold of **0.29537626676780526**, it identified **260 of 407 churners (63.88%)**, with **61.32% precision**. It flagged 424 customers (21.2%), including 164 retained customers, and missed 147 churners.

These results describe held-out classification within this dataset. They do not establish future-period performance or the effectiveness of retention interventions.

## Objective and scope

Estimate churn probabilities and assess whether customer attributes can support prioritising customers for retention review. Standard logistic regression provides the initial interpretable model; decision trees and random forests provide nonlinear comparisons.

No deployment, from-scratch optimisation, manual gradients or SGD experiment is included. A technical threshold criterion—maximum validation F1—is used because intervention costs and contact capacity are unknown.

## Data

Source: [Kaggle Bank Customer Churn Dataset](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data).

The inspected CSV contains 10,000 rows and 12 columns. No missing cells, duplicate complete records, repeated customer IDs or identical records excluding ID were found. Binary columns contain only 0 and 1. Zero balances were retained as potentially valid observations.

| Column | Interpretation |
|---|---|
| customer_id | Identifier, excluded from model inputs |
| credit_score | Recorded credit score |
| country | Country of residence |
| gender | Recorded gender category |
| age | Customer age, interpreted in years |
| tenure | Years holding an account, according to source description |
| balance | Account balance; currency unconfirmed |
| products_number | Number of bank products held |
| credit_card | Credit-card indicator |
| active_member | Activity indicator; operational definition unconfirmed |
| estimated_salary | Estimated salary; currency and reporting period unconfirmed |
| churn | Target: 1 = churned, 0 = retained |

The churn horizon, exact outcome definition, predictor measurement times and real/synthetic provenance have not been independently established. Prospective prediction therefore remains unverified.

## Evaluation design and leakage prevention

- Shuffled, stratified 60/20/20 split: 6,000 training, 2,000 validation and 2,000 test customers.
- Two-stage splitting with random_state=42: reserve 20% for testing, then 25% of the remainder for validation.
- Customer ID disjointness checked across partitions; exact split membership recorded in MLflow.
- Training contains 1,223 churners (20.38%). Validation and test each contain 407 churners (20.35%).
- EDA and feature proposals use training data. Hyperparameter searches use five-fold shuffled stratified training CV with random_state=42.
- Scaling, encoding and spline knot estimation occur inside pipelines and are refitted within each CV training fold.
- No oversampling or undersampling. Class weighting was compared for logistic regression.
- Model configurations selected by mean CV average precision; validation used for shortlist assessment, calibration diagnostics, error analysis and threshold selection.
- Model and threshold frozen before final test evaluation. No refit on validation data and no post-test tuning.

Repeated development comparisons and searches can make winning CV and validation scores optimistic. The final test evaluates only the selected forest; it does not establish final-test superiority over all alternatives. The random split assumes independent customer records and is not a temporal backtest.

## Training EDA and feature engineering

Product-count churn rates were 27.86%, 7.31%, 83.95% and 100% for one through four products. The final group contains only 39 training customers. This nonlinear pattern motivated one-hot encoding product count for logistic regression.

Age-group churn rose from 8.03% under 30 to 54.92% at 50–59, then fell to 28.48% at 60+. A cubic spline allowed a smooth nonlinear age response.

Zero-balance customers had 13.14% churn versus 24.37% among positive-balance customers. A zero-balance indicator was tested but showed no clear incremental CV benefit.

These are associations, not causal explanations.

### Logistic regression experiments

| Configuration | Mean training CV average precision |
|---|---:|
| Original features, C=1 | 0.4680 |
| Original features, tuned C=100 | 0.4680 |
| Product count one-hot encoded, C=1 | 0.6218 |
| Product categories plus zero-balance indicator, C=1 | 0.6218 |
| Product categories plus age spline, C=1 | 0.6706 |
| Product categories plus age spline, tuned C=100 | 0.6729 |

Ordinary numeric predictors were standardised, country/gender encoded and binary indicators passed through. The spline used four quantile knots, degree three, include_bias=False and linear extrapolation, followed by scaling. Searches tested C in [0.01, 0.1, 1, 10, 100] and class_weight in [None, balanced].

Feature representation improved ranking much more than regularisation tuning. Balanced weighting increased default-threshold recall but lowered average precision for the original representation.

Coefficient interpretation used category coefficient differences because all one-hot levels were retained. Scaled numeric coefficients describe one training-standard-deviation changes. Spline coefficients were interpreted jointly through partial dependence. The extreme four-product coefficient reflects separation in a small group and weak regularisation, not a reliable population odds ratio.

### Tree and forest experiments

Trees used unscaled numeric inputs, numeric product count and one-hot country/gender; no age spline was needed.

The initial depth-four tree achieved CV AP 0.5799. A depth/leaf search selected max_depth=6 and min_samples_leaf=5, achieving 0.6268. Small-leaf, deep trees showed increasing training AP and falling held-out AP.

The initial 300-tree forest achieved training AP 0.9052 and CV AP 0.6851. The selected forest reduced the gap: training AP 0.7450, CV AP 0.6884.

Selected forest settings: n_estimators=300, max_depth=6, min_samples_leaf=5, max_features=0.7, bootstrap=True, class_weight=None, random_state=42. Depth [6,12,None], leaf size [5,20,50] and max_features [sqrt,0.7] were searched.

## Validation model comparison

These classification metrics use each estimator's default decision rule.

| Model | ROC-AUC | Average precision | Precision | Recall | F1 | Brier | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Engineered logistic | 0.8414 | 0.6531 | 0.7458 | 0.4398 | 0.5533 | 0.1102 | 0.3608 |
| Selected decision tree | 0.8441 | 0.6151 | 0.7680 | 0.3415 | 0.4728 | 0.1130 | 0.4198 |
| Selected random forest | 0.8558 | 0.6732 | 0.7783 | 0.4226 | 0.5478 | 0.1072 | 0.3506 |

The majority-class dummy achieved 79.65% validation accuracy and zero churn recall. A separate constant-probability baseline using training churn prevalence achieved Brier 0.1621 and log loss 0.5052.

Calibration was assessed using ten quantile groups and probability histograms. No large systematic error was apparent; no post-hoc calibrator was fitted. Brier/log loss measure overall probability quality, not calibration alone.

### Threshold selection

Each shortlisted forest/logistic threshold maximised validation F1; ties selected the highest threshold. Full precision was retained.

| Model | Threshold (rounded) | Precision | Recall | F1 | Flagged | Detected | False alarms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Engineered logistic | 0.3476 | 0.6139 | 0.6093 | 0.6116 | 404 | 248 | 156 |
| Random forest | 0.2954 | 0.6095 | 0.6290 | 0.6191 | 420 | 256 | 164 |

These are threshold-selection results, not unbiased performance estimates. The forest's modest advantage in CV/validation AP and probability scores supported its selection; logistic regression remains a competitive interpretable alternative.

### Validation error analysis

Forest recall under 40 was only 15/121; it missed 106 younger churners. For two-product customers it detected 18/69 churners. All three-product customers were flagged, including 11 retained customers. Four-product validation results had only 11 customers, all churners; false-positive rate was undefined.

Errors varied by country and gender. Similar rates on selected metrics do not establish fairness. Subgroup results lacked uncertainty intervals and were not used to introduce group-specific thresholds.

## Final test results

The forest fitted on the original 6,000 training customers was evaluated using the frozen threshold 0.29537626676780526.

| Metric | Result |
|---|---:|
| Accuracy | 0.844500 |
| Precision | 0.613208 |
| Recall | 0.638821 |
| Specificity | 0.897050 |
| F1 | 0.625752 |
| ROC-AUC | 0.857386 |
| Average precision | 0.698029 |
| Brier score | 0.103561 |
| Log loss | 0.341652 |

| Actual / Predicted | Retained | Churned |
|---|---:|---:|
| Retained | 1,429 | 164 |
| Churned | 147 | 260 |

The model flagged 21.2% of customers and identified 63.9% of churners. Test calibration broadly followed the diagonal, with some underestimation in the highest-risk quantile group. No changes were made in response.

## MLflow tracking

Experiment: bank-churn-model-comparison. Local SQLite database: mlflow.db; artifact directory: mlflow_artifacts.

Recorded items include model parameters, fold results, search grids/results, validation/test metrics, threshold comparisons, confusion matrices, calibration/ranking/interpretation plots and subgroup diagnostics. Dataset SHA-256, split customer membership and environment details were recorded with the original logistic baseline. Search and evaluation runs reference their source runs through tags or JSON artifacts.

Selected pipelines were saved and loaded; probability equality was checked on sample customers. Skops serialization required explicitly trusted SciPy spline classes for the spline pipeline and sklearn.tree._tree.Tree for tree models.

The final run, final-random-forest-test-evaluation, contains frozen_configuration.json, the selected saved model URI, threshold and final evaluation artifacts. Local run IDs are in the user's MLflow database, not hard-coded here.

## Reproducing the analysis

This README documents the notebook project developed interactively in VS Code. It does not bundle the user's notebook, CSV, fitted models, package lock file or MLflow database.

1. Place the dataset at data/raw/bank_churn.csv.
2. Use the project's saved requirements.txt and recorded Python version to recreate its environment. Reported package versions were pandas 3.1.1, scikit-learn 1.9.1 and MLflow 3.18.0; use the saved environment artifact as authoritative.
3. Open notebooks/01_bank_data_understanding.ipynb and select the project .venv kernel.
4. Replace the machine-specific PROJECT_ROOT with the local project directory; run notebook cells in order. The original notebook must be supplied with the repository.
5. Preserve the documented splits and seeds. Reuse recorded customer membership when reproducing evaluation exactly.
6. Open MLflow from the project root:

```bash
source .venv/bin/activate
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001
```

Open http://127.0.0.1:5001. Keep the terminal running while reviewing experiments. Preserve both the database and its referenced artifact files when transferring tracking history; the database alone does not contain the fitted models and figures.

The full saved threshold must be applied explicitly to predict_proba output. Calling the pipeline's predict method uses its default decision rule, not the selected threshold.

## Limitations and conclusions

- A single dataset and random split do not demonstrate temporal or external generalisation.
- Predictor availability before churn and the churn horizon remain unconfirmed.
- No experiment established that retention contact prevents churn or creates financial savings.
- F1 is a technical surrogate; operational costs and capacity could justify a different decision policy in a future study.
- Winning CV/validation scores reflect repeated selection. No confidence intervals or formal significance tests were produced.
- Sparse groups, especially three/four-product customers, limit subgroup conclusions.
- Gender was included and diagnosed; a planned with/without-gender sensitivity experiment was not completed. No fairness claim is made.
- Training-size learning curves were discussed but not produced; the tree/forest complexity plots diagnose changes in model complexity rather than training sample size.

The main learning was that feature engineering substantially improved logistic regression, whereas regularisation tuning alone provided little benefit. Random forest delivered a modest additional development advantage and similar final-test performance to its validation estimates. Important missed-customer and subgroup limitations remain.
