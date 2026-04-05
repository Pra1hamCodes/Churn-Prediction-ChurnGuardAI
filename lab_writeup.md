# LAB EXPERIMENT 9: Customer Churn Prediction using Data Science Lifecycle

---

## Aim

To implement a complete, end-to-end Customer Churn Prediction system for telecommunications data using the full Data Science lifecycle — encompassing data collection, preprocessing, exploratory data analysis, feature engineering, model building, evaluation, deployment, and monitoring — and to analyse and compare the performance of multiple machine learning classification algorithms.

---

## Objective

1. To understand and implement all 9 stages of the Data Science lifecycle in a real-world context.
2. To preprocess raw telecom customer data by handling missing values, encoding categorical variables, and engineering new features.
3. To perform comprehensive Exploratory Data Analysis (EDA) and extract actionable business insights from the dataset.
4. To build, train, and evaluate multiple classification models — Logistic Regression, Random Forest, XGBoost, and Support Vector Machine — for churn prediction.
5. To compare model performance using metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
6. To deploy the best-performing model as a web application capable of real-time churn prediction.
7. To design a production monitoring strategy for continuous model performance tracking.

---

## Theory

### The Data Science Lifecycle

The Data Science lifecycle is a structured, iterative methodology that guides the transformation of raw data into actionable insights and deployed intelligent systems. It consists of nine interconnected stages, each building upon the outputs of the previous one, forming a continuous loop of improvement and refinement.

### Stage 1: Problem Definition

The first and arguably most critical stage involves clearly defining the business problem to be solved. In our context, the problem is **customer churn** — the phenomenon where customers discontinue their relationship with a company. For telecom companies, customer acquisition costs are 5-10x higher than retention costs, making churn prediction an extremely valuable business application. The goal is to identify customers who are likely to churn so that proactive retention strategies can be deployed. We frame this as a **binary classification problem**: given a set of customer attributes, predict whether the customer will churn (Yes) or not (No).

### Stage 2: Data Collection

Data collection involves gathering relevant data from appropriate sources. In this experiment, we use the IBM Telco Customer Churn dataset, which contains 7,043 customer records with 21 attributes including demographics (gender, senior citizen status, partner, dependents), account information (tenure, contract type, payment method, billing preferences), service subscriptions (phone, internet, streaming, security, backup, tech support), and financial data (monthly charges, total charges). The dataset represents real-world patterns observed in telecommunications companies.

### Stage 3: Data Preprocessing

Raw data is rarely clean or ready for analysis. Data preprocessing involves transforming raw data into a format suitable for machine learning algorithms. Key preprocessing steps include: (a) **Handling Missing Values** — the TotalCharges column contains blank strings that must be converted to numeric values and imputed using statistical measures like median; (b) **Removing Irrelevant Features** — the customerID column provides no predictive value and is dropped; (c) **Encoding Categorical Variables** — machine learning algorithms require numerical inputs, so categorical variables like gender, contract type, and payment method are encoded using Label Encoding, which assigns a unique integer to each category; (d) **Data Type Conversion** — ensuring all columns have appropriate data types for computation.

### Stage 4: Exploratory Data Analysis (EDA)

EDA is the process of systematically examining the dataset to discover patterns, spot anomalies, test hypotheses, and check assumptions using statistical graphics and summary statistics. In this experiment, we perform five key analyses: (1) **Churn Distribution Analysis** — understanding the class balance through pie charts reveals that approximately 26.5% of customers churned, indicating a moderately imbalanced dataset; (2) **Contract Type Analysis** — bar charts show that month-to-month contracts have significantly higher churn rates compared to annual or biannual contracts; (3) **Tenure Analysis** — histogram and KDE plots reveal that customers with shorter tenure are more likely to churn, while long-tenure customers show strong loyalty; (4) **Correlation Analysis** — heatmaps reveal inter-feature relationships and identify which features are most strongly correlated with churn; (5) **Charges Distribution** — KDE plots show that churned customers tend to have higher monthly charges.

### Stage 5: Feature Engineering

Feature engineering is the art and science of creating new features or transforming existing ones to improve model performance. We apply: (a) **Standard Scaling** — normalizing numerical features (tenure, MonthlyCharges, TotalCharges) to have zero mean and unit variance, which is crucial for distance-based algorithms like SVM; (b) **Feature Creation** — deriving a new feature "ChargesPerMonth" by dividing TotalCharges by (tenure + 1), which captures the average spending intensity; (c) **Train-Test Splitting** — dividing the dataset into 80% training and 20% testing sets with stratification to maintain class proportions.

### Stage 6: Model Building

We train four fundamentally different classification algorithms to compare their approaches: (a) **Logistic Regression** — a linear model that estimates the probability of churn using a logistic (sigmoid) function, providing interpretable coefficients; (b) **Random Forest** — an ensemble method that builds multiple decision trees on random subsets of data and features, aggregating their predictions through majority voting for robust classification; (c) **XGBoost (Extreme Gradient Boosting)** — a sequential ensemble method that builds trees iteratively, with each new tree correcting errors made by previous trees, known for state-of-the-art performance; (d) **Support Vector Machine (SVM)** — finds the optimal hyperplane that maximizes the margin between classes in a high-dimensional feature space, using kernel tricks for non-linear boundaries. Each model is validated using 5-fold Stratified Cross-Validation, which provides a more reliable estimate of generalization performance.

### Stage 7: Model Evaluation

Comprehensive evaluation uses multiple metrics because no single metric tells the complete story: **Accuracy** measures overall correctness but can be misleading with imbalanced data; **Precision** measures what proportion of predicted churners actually churned (minimizing false alarms); **Recall** measures what proportion of actual churners were correctly identified (minimizing missed churners); **F1-Score** provides the harmonic mean of precision and recall, balancing both concerns; **ROC-AUC** measures the model's ability to discriminate between classes across all classification thresholds. Confusion matrices provide a detailed breakdown of true positives, true negatives, false positives, and false negatives.

### Stage 8: Deployment

The best model (selected by F1-Score) is serialized using joblib and deployed as a Flask web application with a REST API. The deployment includes a beautiful web interface where users can input customer attributes and receive real-time churn predictions, including probability scores, risk level assessments, top contributing factors, and actionable retention recommendations.

### Stage 9: Monitoring & Maintenance

Production models require continuous monitoring to detect performance degradation. Key monitoring aspects include: tracking prediction accuracy against actual outcomes, detecting data drift using Population Stability Index (PSI) and statistical tests, establishing retraining triggers based on performance thresholds, and implementing model versioning with rollback capabilities.

---

## Algorithm / Methodology

```
START

1. DATA COLLECTION
   └── Load IBM Telco Customer Churn dataset (CSV)
   └── Display shape, dtypes, first 5 rows

2. DATA PREPROCESSING
   ├── Drop customerID column (non-predictive)
   ├── Convert TotalCharges blank strings → NaN → fill with median
   ├── Verify: zero null values remaining
   └── Encode all categorical columns with LabelEncoder

3. EXPLORATORY DATA ANALYSIS
   ├── Plot 1: Churn distribution pie chart
   ├── Plot 2: Churn vs Contract type bar chart
   ├── Plot 3: Churn vs Tenure histogram + KDE
   ├── Plot 4: Correlation heatmap (all features)
   ├── Plot 5: Monthly Charges KDE by churn status
   └── Print statistical insights (means, correlations)

4. FEATURE ENGINEERING
   ├── Create ChargesPerMonth = TotalCharges / (tenure + 1)
   ├── Scale numerical features with StandardScaler
   ├── Train/Test split: 80/20, stratified, random_state=42
   └── Print feature importance preview

5. MODEL BUILDING
   ├── Initialize: LogisticRegression, RandomForest, XGBoost, SVM
   ├── For each model:
   │   ├── 5-fold Stratified Cross-Validation
   │   ├── Record CV F1 scores (mean ± std)
   │   └── Fit model on full training set
   └── Store all trained models

6. MODEL EVALUATION
   ├── For each model compute:
   │   ├── Accuracy, Precision, Recall, F1-Score
   │   ├── Confusion Matrix → save as heatmap PNG
   │   └── ROC-AUC Score, ROC Curve data
   ├── Plot: All ROC curves on single figure
   └── Print: Comparison table of all models

7. BEST MODEL SELECTION
   ├── Select model with highest F1-Score
   ├── Save model → best_churn_model.pkl (joblib)
   └── Save scaler → scaler.pkl (joblib)

8. DEPLOYMENT
   ├── Define predict_churn() function
   ├── Load saved model + scaler
   ├── Test with 3 sample customers (Low/Medium/High risk)
   └── Print predictions with recommendations

9. MONITORING STRATEGY
   └── Print production monitoring plan

END
```

---

## Expected Output

### Console Output:
- **Step 1**: Dataset shape (7043 rows × 21 columns), data types, and preview of first 5 records
- **Step 2**: Null value counts before (11 in TotalCharges) and after (0) preprocessing, encoding summary for each categorical column
- **Step 3**: Statistical insights including churn rate (~26.5%), tenure and charges statistics by churn group, top 5 correlated features
- **Step 4**: Scaled feature statistics (mean ≈ 0, std ≈ 1), train/test sizes (5634/1409), feature importance rankings
- **Step 5**: Cross-validation F1 scores for each model with mean ± standard deviation
- **Step 6**: Per-model metrics table with Accuracy (~0.79-0.81), Precision (~0.63-0.67), Recall (~0.48-0.55), F1-Score (~0.55-0.60), ROC-AUC (~0.83-0.85)
- **Step 7**: Best model name and saved file paths
- **Step 8**: Predictions for 3 sample customers with probability, risk level, and recommendations
- **Step 9**: Comprehensive production monitoring plan

### Generated Files:
- 5 EDA plot PNGs (pie chart, bar chart, histogram, heatmap, KDE)
- 4 confusion matrix heatmap PNGs (one per model)
- 1 ROC curves comparison PNG (all models overlaid)
- 2 serialized files (model.pkl, scaler.pkl)

### Web Application:
- Dark-themed professional UI at `http://localhost:5000`
- Interactive form with dropdowns and sliders for all customer features
- Animated circular gauge showing churn probability
- Color-coded risk badge (green/yellow/red)
- Top 3 risk factors with explanations
- Personalized retention recommendations

---

## Conclusion

This experiment successfully demonstrates the implementation of a complete Data Science lifecycle for solving a real-world business problem — customer churn prediction in the telecommunications industry. Through systematic execution of all nine lifecycle stages, we have shown how raw customer data can be transformed into an actionable predictive system that delivers tangible business value.

The importance of Data Science in modern business cannot be overstated. In an era where companies generate vast amounts of data daily, the ability to extract meaningful patterns and build predictive systems represents a critical competitive advantage. Customer churn prediction, as demonstrated in this experiment, exemplifies how Data Science bridges the gap between raw data and strategic decision-making.

Our analysis revealed several key findings: month-to-month contracts, short tenure, high monthly charges, lack of bundled services (online security, tech support), and electronic check payments are the strongest predictors of customer churn. These insights enable telecom companies to design targeted retention campaigns — such as offering contract upgrade incentives to at-risk customers, bundling security and support services, and implementing automatic payment options — thereby significantly reducing revenue loss from customer attrition.

The comparison of four machine learning algorithms (Logistic Regression, Random Forest, XGBoost, and SVM) demonstrates that no single algorithm universally dominates; each has strengths depending on the evaluation metric prioritized. The use of F1-Score as the primary selection criterion reflects a balanced consideration of both precision (avoiding unnecessary retention spending on non-churning customers) and recall (identifying as many potential churners as possible).

The deployment of the model as a web application illustrates the critical final step of making data science accessible to non-technical stakeholders. The real-time prediction interface, complete with probability scores, risk assessments, and actionable recommendations, transforms a statistical model into a practical business tool. The monitoring strategy ensures that the deployed model remains reliable over time, with clear triggers for retraining and systematic version control.

This experiment reinforces that Data Science is not merely about building models — it is a holistic discipline encompassing problem formulation, data engineering, statistical analysis, machine learning, software engineering, and business strategy. Mastery of the complete lifecycle, as practiced here, is essential for any aspiring data scientist seeking to create real-world impact.

---

*Experiment completed successfully. All 9 stages of the Data Science lifecycle have been implemented and validated.*
