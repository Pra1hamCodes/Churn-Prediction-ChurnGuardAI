# 🔮 Customer Churn Prediction in Telecom

> An end-to-end Data Science project implementing the complete ML lifecycle — from data collection to deployment — for predicting customer churn in the telecommunications industry.

---

## 📋 Project Overview

This project implements a **Customer Churn Prediction System** using the IBM Telco Customer Churn dataset. It covers all 9 stages of the Data Science lifecycle:

1. **Problem Definition** — Identifying customers likely to leave
2. **Data Collection** — Loading and inspecting the Telco dataset
3. **Data Preprocessing** — Handling missing values, encoding features
4. **Exploratory Data Analysis** — Statistical analysis and visualizations
5. **Feature Engineering** — Scaling, new features, train/test split
6. **Model Building** — Training 4 ML models with cross-validation
7. **Model Evaluation** — Comprehensive metrics and comparison
8. **Deployment** — Flask web app with REST API
9. **Monitoring** — Production monitoring strategy

---

## 📊 Dataset Description

| Property | Value |
|---|---|
| **Name** | IBM Telco Customer Churn |
| **Source** | IBM Sample Data Sets |
| **Rows** | 7,043 customers |
| **Columns** | 21 features |
| **Target** | Churn (Yes/No) |
| **Class Balance** | ~73.5% No / ~26.5% Yes |

### Key Features:
- **Demographics**: gender, SeniorCitizen, Partner, Dependents
- **Services**: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Account**: Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, tenure

---

## 🚀 How to Run

### Prerequisites
- Python 3.9+
- pip

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the ML Pipeline
```bash
python churn_prediction.py
```
This will:
- Load and preprocess the data
- Generate EDA plots in `plots/` folder
- Train 4 ML models with cross-validation
- Save the best model as `best_churn_model.pkl`
- Save the scaler as `scaler.pkl`
- Run deployment simulation with sample customers

### Step 3: Launch the Web App
```bash
python app.py
```
Open your browser and go to: **http://localhost:5000**

---

## 📈 Model Performance Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.80 | ~0.65 | ~0.55 | ~0.59 | ~0.84 |
| Random Forest | ~0.79 | ~0.64 | ~0.48 | ~0.55 | ~0.83 |
| XGBoost | ~0.80 | ~0.66 | ~0.53 | ~0.59 | ~0.84 |
| SVM | ~0.80 | ~0.67 | ~0.51 | ~0.58 | ~0.84 |

> *Note: Exact values will be printed when you run `churn_prediction.py`*

---

## 📁 File Structure

```
EXP9/
├── datasets/
│   ├── data.csv                          # IBM Telco dataset
│   └── telco_churn.csv                   # Alternative dataset format
├── plots/
│   ├── plot1_churn_distribution.png      # Churn pie chart
│   ├── plot2_churn_vs_contract.png       # Contract bar chart
│   ├── plot3_churn_vs_tenure.png         # Tenure histogram/KDE
│   ├── plot4_correlation_heatmap.png     # Feature correlation heatmap
│   ├── plot5_monthly_charges_by_churn.png# Monthly charges distribution
│   ├── confusion_matrix_*.png           # Per-model confusion matrices
│   └── roc_curves_comparison.png        # ROC curves comparison
├── templates/
│   └── index.html                        # Web app frontend
├── churn_prediction.py                   # Complete ML pipeline
├── app.py                                # Flask web application
├── best_churn_model.pkl                  # Saved best model
├── scaler.pkl                            # Saved feature scaler
├── lifecycle_diagram.drawio              # Data Science lifecycle diagram
├── lab_writeup.md                        # University lab experiment writeup
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

---

## 📸 Screenshots

### EDA Plots
After running `churn_prediction.py`, all plots are saved in the `plots/` directory.

### Web Application
The Flask web app provides:
- Professional dark-themed UI
- Interactive customer data input form
- Real-time churn probability gauge
- Risk level assessment (Low/Medium/High)
- Top 3 risk factor analysis
- Personalized retention recommendations

---

## 🛠 Technologies Used

- **Python 3.9+** — Core programming language
- **Pandas & NumPy** — Data manipulation
- **Matplotlib & Seaborn** — Data visualization
- **Scikit-learn** — ML models and preprocessing
- **XGBoost** — Gradient boosting classifier
- **Flask** — Web framework for deployment
- **Joblib** — Model serialization

---

## 📝 License

This project is developed for educational purposes as part of a Data Science university lab experiment.

---

*Built with ❤️ for Data Science Experiment 9*
