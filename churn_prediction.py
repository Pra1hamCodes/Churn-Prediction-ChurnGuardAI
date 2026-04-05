"""
=============================================================================
    CUSTOMER CHURN PREDICTION IN TELECOM
    Complete Data Science Lifecycle Implementation
    
    Dataset: IBM Telco Customer Churn
    Author: Data Science Lab - University Experiment 9
    Date: April 2026
=============================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================
import sys
import os
import warnings

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
import joblib

# Attempt to import xgboost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARNING] XGBoost not installed. Installing now...")
    os.system("pip install xgboost")
    try:
        from xgboost import XGBClassifier
        XGBOOST_AVAILABLE = True
    except ImportError:
        XGBOOST_AVAILABLE = False
        print("[ERROR] Could not install XGBoost. Skipping XGBoost model.")

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
})


def print_header(step_num, title):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  STEP {step_num}: {title}")
    print("=" * 60 + "\n")


# ============================================================================
# STEP 1: DATA COLLECTION
# ============================================================================
def step1_data_collection():
    """
    Load the IBM Telco Customer Churn dataset from local datasets folder.
    Falls back to downloading from GitHub if local file is not found.
    
    Returns:
        pd.DataFrame: Raw dataset
    """
    print_header(1, "DATA COLLECTION")
    
    # Try local dataset first
    local_path_1 = os.path.join(DATASET_DIR, "data.csv")
    local_path_2 = os.path.join(DATASET_DIR, "telco_churn.csv")
    
    if os.path.exists(local_path_1):
        print(f"[INFO] Loading dataset from local file: {local_path_1}")
        df = pd.read_csv(local_path_1)
    elif os.path.exists(local_path_2):
        print(f"[INFO] Loading dataset from local file: {local_path_2}")
        df = pd.read_csv(local_path_2)
    else:
        url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        print(f"[INFO] Local file not found. Downloading from:\n  {url}")
        df = pd.read_csv(url)
        df.to_csv(local_path_1, index=False)
        print(f"[INFO] Dataset saved to: {local_path_1}")
    
    print(f"\n📊 Dataset Shape: {df.shape}")
    print(f"   - Rows: {df.shape[0]}")
    print(f"   - Columns: {df.shape[1]}")
    
    print(f"\n📋 Column Data Types:")
    print(df.dtypes.to_string())
    
    print(f"\n📝 First 5 Rows:")
    print(df.head().to_string())
    
    print(f"\n📈 Dataset Info Summary:")
    print(f"   - Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print(f"   - Numeric Columns: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"   - Categorical Columns: {len(df.select_dtypes(include=['object', 'bool']).columns)}")
    
    return df


# ============================================================================
# STEP 2: DATA PREPROCESSING
# ============================================================================
def step2_data_preprocessing(df):
    """
    Clean and preprocess the dataset:
    - Handle missing values in TotalCharges
    - Drop customerID column
    - Encode categorical columns using LabelEncoder
    
    Args:
        df: Raw dataframe
    
    Returns:
        tuple: (processed_df, label_encoders_dict)
    """
    print_header(2, "DATA PREPROCESSING")
    
    df = df.copy()
    
    # --- Drop unnamed/index columns ---
    cols_to_drop = [c for c in df.columns if 'Unnamed' in str(c)]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"[INFO] Dropped unnamed columns: {cols_to_drop}")
    
    # --- Handle customerID ---
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)
        print("[INFO] Dropped 'customerID' column")
    
    # --- Handle TotalCharges (blank strings → NaN → fill with median) ---
    print("\n--- Handling TotalCharges ---")
    if df['TotalCharges'].dtype == 'object':
        print(f"  Before: TotalCharges dtype = {df['TotalCharges'].dtype}")
        blank_count = (df['TotalCharges'].str.strip() == '').sum()
        print(f"  Blank string values found: {blank_count}")
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    null_before = df.isnull().sum()
    print(f"\n📊 Null Values BEFORE Handling:")
    null_cols = null_before[null_before > 0]
    if len(null_cols) > 0:
        for col, cnt in null_cols.items():
            print(f"   {col}: {cnt} nulls")
    else:
        print("   No null values found")
    
    # Fill TotalCharges nulls with median
    if df['TotalCharges'].isnull().sum() > 0:
        median_val = df['TotalCharges'].median()
        df['TotalCharges'].fillna(median_val, inplace=True)
        print(f"\n[INFO] Filled TotalCharges NaNs with median: {median_val:.2f}")
    
    null_after = df.isnull().sum()
    print(f"\n📊 Null Values AFTER Handling:")
    null_cols_after = null_after[null_after > 0]
    if len(null_cols_after) > 0:
        for col, cnt in null_cols_after.items():
            print(f"   {col}: {cnt} nulls")
    else:
        print("   ✅ No null values remaining!")
    
    # --- Handle Churn column (convert True/False to Yes/No if needed) ---
    if df['Churn'].dtype == bool or set(df['Churn'].unique()).issubset({True, False, 'True', 'False'}):
        df['Churn'] = df['Churn'].map({
            True: 'Yes', False: 'No',
            'True': 'Yes', 'False': 'No'
        })
        print("[INFO] Converted Churn from True/False to Yes/No")
    
    # --- Handle SeniorCitizen column (convert True/False to 0/1 if needed) ---
    if df['SeniorCitizen'].dtype == bool or df['SeniorCitizen'].dtype == 'object':
        df['SeniorCitizen'] = df['SeniorCitizen'].map({
            True: 1, False: 0,
            'True': 1, 'False': 0,
            'Yes': 1, 'No': 0,
            1: 1, 0: 0
        }).fillna(0).astype(int)
        print("[INFO] Standardized SeniorCitizen to 0/1")
    
    # --- Handle blank string values in categorical columns ---
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        blank_mask = df[col].str.strip() == ''
        if blank_mask.sum() > 0:
            df.loc[blank_mask, col] = df[col].mode()[0]
            print(f"[INFO] Filled {blank_mask.sum()} blank values in '{col}' with mode")
    
    # --- Encode categorical columns ---
    print("\n--- Encoding Categorical Columns ---")
    label_encoders = {}
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"  Categorical columns to encode: {cat_cols}")
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"   ✅ {col}: {list(le.classes_)} → {list(range(len(le.classes_)))}")
    
    print(f"\n📊 Final Dataset Shape: {df.shape}")
    print(f"📊 Final Data Types:\n{df.dtypes.to_string()}")
    
    return df, label_encoders


# ============================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
def step3_eda(df, label_encoders):
    """
    Perform comprehensive EDA with 5 visualization plots.
    All plots are saved as PNG files.
    
    Args:
        df: Processed dataframe
        label_encoders: Dictionary of fitted label encoders
    """
    print_header(3, "EXPLORATORY DATA ANALYSIS (EDA)")
    
    # --- Plot 1: Churn Distribution (Pie Chart) ---
    print("📊 Plot 1: Churn Distribution (Pie Chart)")
    fig, ax = plt.subplots(figsize=(8, 8))
    churn_counts = df['Churn'].value_counts()
    
    churn_le = label_encoders.get('Churn')
    if churn_le is not None:
        labels = [churn_le.inverse_transform([i])[0] for i in churn_counts.index]
    else:
        labels = ['No Churn', 'Churn']
    
    colors = ['#00b4d8', '#e94560']
    explode = (0.05, 0.05)
    wedges, texts, autotexts = ax.pie(
        churn_counts.values, labels=labels, autopct='%1.1f%%',
        colors=colors, explode=explode, shadow=True,
        textprops={'fontsize': 14, 'fontweight': 'bold'},
        startangle=90
    )
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax.set_title('Customer Churn Distribution', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plot1_churn_distribution.png'))
    plt.close()
    print("   ✅ Saved: plots/plot1_churn_distribution.png")
    
    # --- Plot 2: Churn vs Contract Type (Bar Chart) ---
    print("📊 Plot 2: Churn vs Contract Type (Bar Chart)")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    contract_le = label_encoders.get('Contract')
    churn_le = label_encoders.get('Churn')
    
    plot_df = df.copy()
    if contract_le is not None:
        plot_df['Contract_Label'] = contract_le.inverse_transform(df['Contract'])
    else:
        plot_df['Contract_Label'] = df['Contract']
    
    if churn_le is not None:
        plot_df['Churn_Label'] = churn_le.inverse_transform(df['Churn'])
    else:
        plot_df['Churn_Label'] = df['Churn']
    
    ct = pd.crosstab(plot_df['Contract_Label'], plot_df['Churn_Label'])
    ct.plot(kind='bar', ax=ax, color=['#00b4d8', '#e94560'], edgecolor='white', width=0.7)
    ax.set_title('Churn by Contract Type', fontsize=16, fontweight='bold')
    ax.set_xlabel('Contract Type', fontsize=13)
    ax.set_ylabel('Number of Customers', fontsize=13)
    ax.legend(title='Churn', fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    for container in ax.containers:
        ax.bar_label(container, fontsize=10, padding=3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plot2_churn_vs_contract.png'))
    plt.close()
    print("   ✅ Saved: plots/plot2_churn_vs_contract.png")
    
    # --- Plot 3: Churn vs Tenure (Histogram/KDE) ---
    print("📊 Plot 3: Churn vs Tenure (Histogram/KDE)")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for churn_val in sorted(df['Churn'].unique()):
        subset = df[df['Churn'] == churn_val]['tenure']
        if churn_le is not None:
            label = churn_le.inverse_transform([churn_val])[0]
        else:
            label = f"Churn={churn_val}"
        color = '#e94560' if label in ['Yes', 1, '1'] else '#00b4d8'
        ax.hist(subset, bins=30, alpha=0.6, label=f'Churn: {label}', color=color, edgecolor='white')
        subset.plot.kde(ax=ax, label=f'KDE - {label}', linewidth=2,
                        color='#ff6b6b' if label in ['Yes', 1, '1'] else '#0077b6',
                        secondary_y=True)
    
    ax.set_title('Customer Tenure Distribution by Churn Status', fontsize=16, fontweight='bold')
    ax.set_xlabel('Tenure (Months)', fontsize=13)
    ax.set_ylabel('Frequency', fontsize=13)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plot3_churn_vs_tenure.png'))
    plt.close()
    print("   ✅ Saved: plots/plot3_churn_vs_tenure.png")
    
    # --- Plot 4: Correlation Heatmap ---
    print("📊 Plot 4: Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(16, 12))
    corr_matrix = df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt='.2f',
        cmap='RdBu_r', center=0, ax=ax,
        linewidths=0.5, linecolor='white',
        annot_kws={'size': 8},
        cbar_kws={'shrink': 0.8, 'label': 'Correlation'}
    )
    ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plot4_correlation_heatmap.png'))
    plt.close()
    print("   ✅ Saved: plots/plot4_correlation_heatmap.png")
    
    # --- Plot 5: Monthly Charges Distribution by Churn ---
    print("📊 Plot 5: Monthly Charges Distribution by Churn")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for churn_val in sorted(df['Churn'].unique()):
        subset = df[df['Churn'] == churn_val]['MonthlyCharges']
        if churn_le is not None:
            label = churn_le.inverse_transform([churn_val])[0]
        else:
            label = f"Churn={churn_val}"
        color = '#e94560' if label in ['Yes', 1, '1'] else '#00b4d8'
        sns.kdeplot(subset, ax=ax, label=f'Churn: {label}', fill=True, alpha=0.4, 
                    color=color, linewidth=2)
    
    ax.set_title('Monthly Charges Distribution by Churn Status', fontsize=16, fontweight='bold')
    ax.set_xlabel('Monthly Charges ($)', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plot5_monthly_charges_by_churn.png'))
    plt.close()
    print("   ✅ Saved: plots/plot5_monthly_charges_by_churn.png")
    
    # --- Statistical Insights ---
    print("\n" + "─" * 50)
    print("📈 KEY STATISTICAL INSIGHTS")
    print("─" * 50)
    
    churn_rate = df['Churn'].mean() * 100 if df['Churn'].max() <= 1 else (df['Churn'].value_counts(normalize=True).max() * 100)
    print(f"\n  1. Overall Churn Rate: {churn_rate:.1f}%")
    
    print(f"\n  2. Tenure Statistics:")
    for churn_val in sorted(df['Churn'].unique()):
        subset = df[df['Churn'] == churn_val]['tenure']
        if churn_le is not None:
            label = churn_le.inverse_transform([churn_val])[0]
        else:
            label = f"Churn={churn_val}"
        print(f"     Churn={label}: Mean={subset.mean():.1f}, Median={subset.median():.1f}")
    
    print(f"\n  3. Monthly Charges Statistics:")
    for churn_val in sorted(df['Churn'].unique()):
        subset = df[df['Churn'] == churn_val]['MonthlyCharges']
        if churn_le is not None:
            label = churn_le.inverse_transform([churn_val])[0]
        else:
            label = f"Churn={churn_val}"
        print(f"     Churn={label}: Mean=${subset.mean():.2f}, Std=${subset.std():.2f}")
    
    top_corr = corr_matrix['Churn'].drop('Churn').abs().sort_values(ascending=False)
    print(f"\n  4. Top 5 Features Correlated with Churn:")
    for feat, corr_val in top_corr.head(5).items():
        direction = "+" if corr_matrix.loc[feat, 'Churn'] > 0 else "-"
        print(f"     {direction} {feat}: {corr_val:.4f}")
    
    print(f"\n  5. Dataset is {'imbalanced' if abs(churn_rate - 50) > 15 else 'relatively balanced'}")
    
    return corr_matrix


# ============================================================================
# STEP 4: FEATURE ENGINEERING
# ============================================================================
def step4_feature_engineering(df):
    """
    Engineer features and prepare data for modeling:
    - Scale numerical features
    - Create ChargesPerMonth feature
    - Train/test split (80/20)
    
    Args:
        df: Processed dataframe
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler, feature_names)
    """
    print_header(4, "FEATURE ENGINEERING")
    
    df = df.copy()
    
    # --- Create new feature: ChargesPerMonth ---
    print("--- Creating New Features ---")
    df['ChargesPerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)
    print(f"  ✅ Created 'ChargesPerMonth' = TotalCharges / (tenure + 1)")
    print(f"     Mean: {df['ChargesPerMonth'].mean():.2f}")
    print(f"     Std:  {df['ChargesPerMonth'].std():.2f}")
    
    # --- Separate features and target ---
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    feature_names = X.columns.tolist()
    
    print(f"\n  Features: {len(feature_names)}")
    print(f"  Target: Churn")
    print(f"  Target Distribution: {dict(y.value_counts())}")
    
    # --- Scale numerical features ---
    print("\n--- Scaling Numerical Features ---")
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'ChargesPerMonth']
    existing_num_cols = [c for c in numerical_cols if c in X.columns]
    
    scaler = StandardScaler()
    X[existing_num_cols] = scaler.fit_transform(X[existing_num_cols])
    
    for col in existing_num_cols:
        print(f"  ✅ Scaled '{col}': Mean={X[col].mean():.4f}, Std={X[col].std():.4f}")
    
    # --- Train/Test Split ---
    print("\n--- Train/Test Split ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"  Training Set: {X_train.shape[0]} samples ({(1-TEST_SIZE)*100:.0f}%)")
    print(f"  Test Set:     {X_test.shape[0]} samples ({TEST_SIZE*100:.0f}%)")
    print(f"  Train Target Distribution: {dict(y_train.value_counts())}")
    print(f"  Test Target Distribution:  {dict(y_test.value_counts())}")
    
    # --- Feature Importance Preview (using correlation) ---
    print("\n--- Feature Importance Preview (Correlation with Churn) ---")
    full_df = pd.concat([X, y], axis=1)
    importance = full_df.corr()['Churn'].drop('Churn').abs().sort_values(ascending=False)
    print("  Top 10 Most Important Features:")
    for i, (feat, imp) in enumerate(importance.head(10).items(), 1):
        bar = "█" * int(imp * 40)
        print(f"  {i:2d}. {feat:25s} | {imp:.4f} {bar}")
    
    return X_train, X_test, y_train, y_test, scaler, feature_names


# ============================================================================
# STEP 5: MODEL BUILDING
# ============================================================================
def step5_model_building(X_train, y_train):
    """
    Train multiple classification models with 5-fold cross-validation.
    Models: Logistic Regression, Random Forest, XGBoost, SVM
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        dict: {model_name: trained_model}
    """
    print_header(5, "MODEL BUILDING")
    
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, C=1.0, solver='lbfgs'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        'Support Vector Machine': SVC(
            kernel='rbf', probability=True, random_state=RANDOM_STATE,
            C=1.0, gamma='scale'
        )
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, use_label_encoder=False,
            eval_metric='logloss', n_jobs=-1
        )
    
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n🔧 Training: {name}")
        print(f"   Parameters: {model.get_params()}")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
        print(f"   📊 Cross-Validation F1 Scores: {[f'{s:.4f}' for s in cv_scores]}")
        print(f"   📊 Mean CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Train on full training set
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"   ✅ Model trained successfully!")
    
    return trained_models


# ============================================================================
# STEP 6: MODEL EVALUATION
# ============================================================================
def step6_model_evaluation(trained_models, X_test, y_test, feature_names):
    """
    Evaluate all trained models comprehensively:
    - Accuracy, Precision, Recall, F1
    - Confusion Matrix heatmaps
    - ROC curves (all models on one plot)
    - Comparison table
    
    Args:
        trained_models: Dict of trained models
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
    
    Returns:
        pd.DataFrame: Comparison table
    """
    print_header(6, "MODEL EVALUATION")
    
    results = {}
    roc_data = {}
    
    for name, model in trained_models.items():
        print(f"\n{'─' * 50}")
        print(f"📊 Evaluating: {name}")
        print(f"{'─' * 50}")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        }
        
        print(f"  Accuracy:  {acc:.4f}  ({acc*100:.2f}%)")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'], 
                                     zero_division=0))
        
        # --- Confusion Matrix Heatmap ---
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'],
                    linewidths=2, linecolor='white',
                    annot_kws={'size': 16, 'fontweight': 'bold'})
        ax.set_title(f'Confusion Matrix - {name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        
        # Add accuracy annotation
        ax.text(0.5, -0.15, f'Accuracy: {acc:.2%} | F1: {f1:.4f}',
                transform=ax.transAxes, ha='center', fontsize=11, style='italic',
                color='#333333')
        
        safe_name = name.replace(' ', '_').lower()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'confusion_matrix_{safe_name}.png'))
        plt.close()
        print(f"  ✅ Saved: plots/confusion_matrix_{safe_name}.png")
        
        # Store ROC data
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, roc_auc)
    
    # --- ROC Curve (All Models) ---
    print(f"\n{'─' * 50}")
    print(f"📊 Plotting ROC Curves (All Models)")
    print(f"{'─' * 50}")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#e94560', '#00b4d8', '#06d6a0', '#ffc93c', '#533483']
    
    for i, (name, (fpr, tpr, auc_score)) in enumerate(roc_data.items()):
        ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2.5,
                label=f'{name} (AUC = {auc_score:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color='grey')
    ax.set_title('ROC Curves - Model Comparison', fontsize=16, fontweight='bold')
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curves_comparison.png'))
    plt.close()
    print("  ✅ Saved: plots/roc_curves_comparison.png")
    
    # --- Final Comparison Table ---
    print(f"\n{'═' * 70}")
    print(f"{'MODEL COMPARISON TABLE':^70}")
    print(f"{'═' * 70}")
    
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.round(4)
    comparison_df.index.name = 'Model'
    print(comparison_df.to_string())
    
    print(f"\n{'═' * 70}")
    best_model_name = comparison_df['F1-Score'].idxmax()
    best_f1 = comparison_df.loc[best_model_name, 'F1-Score']
    print(f"🏆 Best Model by F1-Score: {best_model_name} (F1 = {best_f1:.4f})")
    print(f"{'═' * 70}")
    
    return comparison_df


# ============================================================================
# STEP 7: BEST MODEL SELECTION & SAVING
# ============================================================================
def step7_save_best_model(trained_models, comparison_df, scaler):
    """
    Select the best model by F1-Score and save it with joblib.
    
    Args:
        trained_models: Dict of trained models
        comparison_df: Comparison dataframe
        scaler: Fitted StandardScaler
    
    Returns:
        str: Name of the best model
    """
    print_header(7, "BEST MODEL SELECTION & SAVING")
    
    best_model_name = comparison_df['F1-Score'].idxmax()
    best_model = trained_models[best_model_name]
    
    print(f"🏆 Best Model: {best_model_name}")
    print(f"   F1-Score:  {comparison_df.loc[best_model_name, 'F1-Score']:.4f}")
    print(f"   Accuracy:  {comparison_df.loc[best_model_name, 'Accuracy']:.4f}")
    print(f"   ROC-AUC:   {comparison_df.loc[best_model_name, 'ROC-AUC']:.4f}")
    
    # Save model
    model_path = os.path.join(MODEL_DIR, 'best_churn_model.pkl')
    joblib.dump(best_model, model_path)
    print(f"\n  ✅ Model saved: {model_path}")
    print(f"     File size: {os.path.getsize(model_path) / 1024:.1f} KB")
    
    # Save scaler
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"  ✅ Scaler saved: {scaler_path}")
    print(f"     File size: {os.path.getsize(scaler_path) / 1024:.1f} KB")
    
    # Save label encoders info for deployment
    print(f"\n  ✅ Both artifacts saved successfully for deployment!")
    
    return best_model_name


# ============================================================================
# STEP 8: DEPLOYMENT SIMULATION
# ============================================================================
def predict_churn(customer_data):
    """
    Predict customer churn using the saved model and scaler.
    
    This function simulates a production deployment by:
    1. Loading the saved model and scaler from disk
    2. Preprocessing the input customer data
    3. Returning the churn probability and prediction
    
    Args:
        customer_data (dict): Dictionary with customer features:
            - gender, SeniorCitizen, Partner, Dependents, tenure
            - PhoneService, MultipleLines, InternetService
            - OnlineSecurity, OnlineBackup, DeviceProtection
            - TechSupport, StreamingTV, StreamingMovies
            - Contract, PaperlessBilling, PaymentMethod
            - MonthlyCharges, TotalCharges
    
    Returns:
        dict: {
            'prediction': 'Yes' or 'No',
            'churn_probability': float (0-1),
            'risk_level': 'Low', 'Medium', or 'High'
        }
    """
    # Load saved model and scaler
    model_path = os.path.join(MODEL_DIR, 'best_churn_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Encode categorical features
    categorical_mappings = {
        'gender': {'Female': 0, 'Male': 1},
        'Partner': {'No': 0, 'Yes': 1},
        'Dependents': {'No': 0, 'Yes': 1},
        'PhoneService': {'No': 0, 'Yes': 1},
        'MultipleLines': {'No': 0, 'No phone service': 1, 'Yes': 2},
        'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
        'OnlineSecurity': {'No': 0, 'No internet service': 1, 'Yes': 2},
        'OnlineBackup': {'No': 0, 'No internet service': 1, 'Yes': 2},
        'DeviceProtection': {'No': 0, 'No internet service': 1, 'Yes': 2},
        'TechSupport': {'No': 0, 'No internet service': 1, 'Yes': 2},
        'StreamingTV': {'No': 0, 'No internet service': 1, 'Yes': 2},
        'StreamingMovies': {'No': 0, 'No internet service': 1, 'Yes': 2},
        'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
        'PaperlessBilling': {'No': 0, 'Yes': 1},
        'PaymentMethod': {
            'Bank transfer (automatic)': 0,
            'Credit card (automatic)': 1,
            'Electronic check': 2,
            'Mailed check': 3
        }
    }
    
    # Build feature vector
    features = {}
    for col, mapping in categorical_mappings.items():
        val = customer_data.get(col, list(mapping.keys())[0])
        features[col] = mapping.get(val, 0)
    
    features['SeniorCitizen'] = int(customer_data.get('SeniorCitizen', 0))
    features['tenure'] = float(customer_data.get('tenure', 1))
    features['MonthlyCharges'] = float(customer_data.get('MonthlyCharges', 50))
    features['TotalCharges'] = float(customer_data.get('TotalCharges', 50))
    
    # Create ChargesPerMonth feature
    features['ChargesPerMonth'] = features['TotalCharges'] / (features['tenure'] + 1)
    
    # Create DataFrame with correct column order
    feature_order = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges', 'ChargesPerMonth'
    ]
    
    input_df = pd.DataFrame([features])[feature_order]
    
    # Scale numerical features
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'ChargesPerMonth']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    
    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    # Risk level
    if probability < 0.3:
        risk_level = 'Low'
    elif probability < 0.6:
        risk_level = 'Medium'
    else:
        risk_level = 'High'
    
    return {
        'prediction': 'Yes' if prediction == 1 else 'No',
        'churn_probability': round(float(probability), 4),
        'risk_level': risk_level
    }


def step8_deployment_simulation():
    """
    Test the predict_churn() function with 3 sample customers.
    Demonstrates production-ready prediction capability.
    """
    print_header(8, "DEPLOYMENT SIMULATION")
    
    sample_customers = [
        {
            'name': 'Customer A (Low Risk - Long Tenure)',
            'gender': 'Female',
            'SeniorCitizen': 0,
            'Partner': 'Yes',
            'Dependents': 'Yes',
            'tenure': 60,
            'PhoneService': 'Yes',
            'MultipleLines': 'Yes',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'Yes',
            'OnlineBackup': 'Yes',
            'DeviceProtection': 'Yes',
            'TechSupport': 'Yes',
            'StreamingTV': 'Yes',
            'StreamingMovies': 'Yes',
            'Contract': 'Two year',
            'PaperlessBilling': 'No',
            'PaymentMethod': 'Bank transfer (automatic)',
            'MonthlyCharges': 85.50,
            'TotalCharges': 5130.00
        },
        {
            'name': 'Customer B (High Risk - New Customer)',
            'gender': 'Male',
            'SeniorCitizen': 0,
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': 2,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 75.00,
            'TotalCharges': 150.00
        },
        {
            'name': 'Customer C (Medium Risk - Mid Tenure)',
            'gender': 'Female',
            'SeniorCitizen': 1,
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': 24,
            'PhoneService': 'Yes',
            'MultipleLines': 'Yes',
            'InternetService': 'DSL',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'Yes',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'Yes',
            'StreamingMovies': 'No',
            'Contract': 'One year',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Credit card (automatic)',
            'MonthlyCharges': 65.00,
            'TotalCharges': 1560.00
        }
    ]
    
    for customer in sample_customers:
        name = customer.pop('name')
        print(f"\n{'─' * 50}")
        print(f"👤 {name}")
        print(f"{'─' * 50}")
        
        result = predict_churn(customer)
        
        emoji = '🔴' if result['risk_level'] == 'High' else ('🟡' if result['risk_level'] == 'Medium' else '🟢')
        
        print(f"  Prediction:        {result['prediction']}")
        print(f"  Churn Probability: {result['churn_probability']:.2%}")
        print(f"  Risk Level:        {emoji} {result['risk_level']}")
        
        if result['risk_level'] == 'High':
            print(f"  💡 Recommendation: Immediate retention action needed!")
            print(f"     → Offer contract upgrade discount")
            print(f"     → Assign dedicated support agent")
        elif result['risk_level'] == 'Medium':
            print(f"  💡 Recommendation: Proactive engagement suggested")
            print(f"     → Send loyalty rewards")
            print(f"     → Schedule satisfaction survey")
        else:
            print(f"  💡 Recommendation: Continue regular engagement")
            print(f"     → Maintain service quality")
            print(f"     → Consider upselling opportunities")


# ============================================================================
# STEP 9: MONITORING STRATEGY
# ============================================================================
def step9_monitoring_strategy():
    """
    Print a comprehensive production monitoring plan for the churn model.
    """
    print_header(9, "MONITORING & MAINTENANCE STRATEGY")
    
    monitoring_plan = """
╔══════════════════════════════════════════════════════════════════════╗
║              PRODUCTION MONITORING PLAN                             ║
╚══════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. METRICS TO TRACK IN PRODUCTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   📊 Model Performance Metrics (Weekly):
      • Accuracy, Precision, Recall, F1-Score
      • ROC-AUC Score
      • Log Loss / Brier Score
      • Prediction Confidence Distribution
   
   📈 Business Metrics (Monthly):
      • Actual vs Predicted Churn Rate
      • Customer Retention Rate
      • Revenue Impact (saved customers × ARPU)
      • Cost of False Positives (unnecessary retention spend)
      • Cost of False Negatives (lost customers)
   
   ⚡ Operational Metrics (Real-time):
      • Prediction Latency (p50, p95, p99)
      • API Response Time
      • Model Endpoint Availability (uptime %)
      • Request Volume / Throughput

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. DATA DRIFT DETECTION APPROACH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   🔍 Feature Drift Monitoring:
      • Population Stability Index (PSI) for each feature
         - PSI < 0.1  → No drift (stable)
         - PSI 0.1-0.2 → Moderate drift (investigate)
         - PSI > 0.2   → Significant drift (retrain)
      
      • Kolmogorov-Smirnov (KS) Test for numerical features
      • Chi-Square Test for categorical features
      • Jensen-Shannon Divergence for distribution changes
   
   📊 Target Drift Monitoring:
      • Compare predicted vs actual churn distributions
      • Track prediction distribution shifts over time
      • Monitor label frequency changes monthly
   
   🛠 Tools: Evidently AI, WhyLabs, or custom dashboards

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. RETRAINING TRIGGER CONDITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   🔴 Immediate Retraining Triggers:
      • F1-Score drops > 10% from baseline
      • ROC-AUC drops below 0.70
      • PSI > 0.25 for any top-5 feature
      • Actual churn rate deviates > 15% from predictions
   
   🟡 Scheduled Retraining:
      • Monthly: Evaluate model on latest labeled data
      • Quarterly: Full model retraining with new data
      • Semi-annually: Feature engineering review
   
   🟢 Proactive Retraining:
      • New competitor enters market
      • Price plan changes
      • Significant infrastructure changes
      • Seasonal patterns detected

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. MODEL VERSIONING STRATEGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   📁 Version Control:
      • Use MLflow or DVC for model versioning
      • Naming: churn_model_v{major}.{minor}_{date}
      • Store: model artifacts, hyperparameters, training data hash
   
   🔄 Deployment Strategy:
      • Shadow Mode: New model runs alongside production (1 week)
      • A/B Testing: Split traffic 90/10, then 50/50
      • Canary Release: Gradual rollout with monitoring
      • Rollback Plan: Automated fallback to previous version
   
   📋 Model Registry:
      • Production, Staging, Archived states
      • Model lineage tracking
      • Automated performance comparison reports
      • Approval workflow for production promotion

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    print(monitoring_plan)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """
    Execute the complete Data Science lifecycle for Customer Churn Prediction.
    Runs all 9 steps sequentially.
    """
    print("\n" + "█" * 60)
    print("█" + " " * 58 + "█")
    print("█  CUSTOMER CHURN PREDICTION - COMPLETE LIFECYCLE" + " " * 9 + "█")
    print("█  IBM Telco Customer Churn Dataset" + " " * 24 + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60 + "\n")
    
    # Step 1: Data Collection
    df_raw = step1_data_collection()
    
    # Step 2: Data Preprocessing
    df_processed, label_encoders = step2_data_preprocessing(df_raw)
    
    # Step 3: EDA
    corr_matrix = step3_eda(df_processed, label_encoders)
    
    # Step 4: Feature Engineering
    X_train, X_test, y_train, y_test, scaler, feature_names = step4_feature_engineering(df_processed)
    
    # Step 5: Model Building
    trained_models = step5_model_building(X_train, y_train)
    
    # Step 6: Model Evaluation
    comparison_df = step6_model_evaluation(trained_models, X_test, y_test, feature_names)
    
    # Step 7: Save Best Model
    best_model_name = step7_save_best_model(trained_models, comparison_df, scaler)
    
    # Step 8: Deployment Simulation
    step8_deployment_simulation()
    
    # Step 9: Monitoring Strategy
    step9_monitoring_strategy()
    
    # Final Summary
    print("\n" + "█" * 60)
    print("█" + " " * 58 + "█")
    print("█  ✅ ALL 9 STEPS COMPLETED SUCCESSFULLY!" + " " * 19 + "█")
    print("█" + " " * 58 + "█")
    print("█  Generated Files:" + " " * 40 + "█")
    print("█    • plots/plot1_churn_distribution.png" + " " * 19 + "█")
    print("█    • plots/plot2_churn_vs_contract.png" + " " * 20 + "█")
    print("█    • plots/plot3_churn_vs_tenure.png" + " " * 22 + "█")
    print("█    • plots/plot4_correlation_heatmap.png" + " " * 18 + "█")
    print("█    • plots/plot5_monthly_charges_by_churn.png" + " " * 12 + "█")
    print("█    • plots/confusion_matrix_*.png" + " " * 25 + "█")
    print("█    • plots/roc_curves_comparison.png" + " " * 22 + "█")
    print("█    • best_churn_model.pkl" + " " * 32 + "█")
    print("█    • scaler.pkl" + " " * 41 + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60)


if __name__ == "__main__":
    main()
