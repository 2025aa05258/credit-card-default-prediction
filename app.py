import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Default Predictor",
    page_icon="💳",
    layout="wide"
)

# ─────────────────────────────────────────────
# Load saved models and scaler
# ─────────────────────────────────────────────
MODEL_DIR = "model"

@st.cache_resource
def load_models():
    """Load all trained models, scaler, and feature names."""
    models = {}
    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "K-Nearest Neighbors": "knn.pkl",
        "Naive Bayes (Gaussian)": "naive_bayes.pkl",
        "Random Forest (Ensemble)": "random_forest.pkl",
        "XGBoost (Ensemble)": "xgboost.pkl"
    }
    for name, filename in model_files.items():
        filepath = os.path.join(MODEL_DIR, filename)
        if os.path.exists(filepath):
            models[name] = joblib.load(filepath)

    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    return models, scaler, feature_names

models, scaler, feature_names = load_models()

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

# (b) Model selection dropdown
selected_model_name = st.sidebar.selectbox(
    "Select ML Model",
    list(models.keys()),
    index=5  # default to XGBoost
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**ML Assignment 2**  \n"
    "Credit Card Default Prediction  \n"
    "6 Classification Models"
)

# ─────────────────────────────────────────────
# Main Title
# ─────────────────────────────────────────────
st.title("💳 Credit Card Default Prediction")
st.markdown(
    "Predict whether a credit card client will **default** on their next month's payment "
    "using 6 different ML classification models."
)

# ─────────────────────────────────────────────
# (a) Dataset Upload Option (CSV)
# ─────────────────────────────────────────────
st.header("📂 Upload Test Data (CSV)")
st.info(
    "Upload a CSV file with the test data. The CSV should contain the **23 feature columns** "
    "and a **target** column. You can use the `test_data.csv` file from the `data/` folder."
)

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ File uploaded successfully! Shape: {df.shape}")

    # Show preview
    with st.expander("📋 Preview Uploaded Data", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    # Check if 'target' column exists
    if 'target' not in df.columns:
        st.error("❌ The uploaded CSV must contain a **'target'** column for evaluation.")
        st.stop()

    # Check if all feature columns are present
    missing_cols = [col for col in feature_names if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing feature columns: {missing_cols}")
        st.stop()

    # Separate features and target
    X_uploaded = df[feature_names]
    y_uploaded = df['target']

    # Scale features
    X_scaled = scaler.transform(X_uploaded)

    # Get selected model
    model = models[selected_model_name]

    # Make predictions
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    st.markdown("---")

    # ─────────────────────────────────────────────
    # (c) Display of Evaluation Metrics
    # ─────────────────────────────────────────────
    st.header(f"📊 Evaluation Metrics — {selected_model_name}")

    accuracy = accuracy_score(y_uploaded, y_pred)
    auc = roc_auc_score(y_uploaded, y_prob)
    precision = precision_score(y_uploaded, y_pred)
    recall = recall_score(y_uploaded, y_pred)
    f1 = f1_score(y_uploaded, y_pred)
    mcc = matthews_corrcoef(y_uploaded, y_pred)

    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 Accuracy", f"{accuracy:.4f}")
    col2.metric("📈 AUC Score", f"{auc:.4f}")
    col3.metric("🔍 Precision", f"{precision:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("📣 Recall", f"{recall:.4f}")
    col5.metric("⚖️ F1 Score", f"{f1:.4f}")
    col6.metric("📐 MCC Score", f"{mcc:.4f}")

    st.markdown("---")

    # ─────────────────────────────────────────────
    # (d) Confusion Matrix & Classification Report
    # ─────────────────────────────────────────────
    st.header(f"📋 Confusion Matrix & Classification Report — {selected_model_name}")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_uploaded, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default']
        )
        ax.set_title(f'{selected_model_name}', fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plt.tight_layout()
        st.pyplot(fig)

    with col_right:
        st.subheader("Classification Report")
        report = classification_report(
            y_uploaded, y_pred,
            target_names=['No Default', 'Default'],
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

    st.markdown("---")

    # ─────────────────────────────────────────────
    # Compare all models on uploaded data
    # ─────────────────────────────────────────────
    st.header("🏆 Compare All Models on Uploaded Data")

    comparison_results = []
    for name, mdl in models.items():
        yp = mdl.predict(X_scaled)
        ypr = mdl.predict_proba(X_scaled)[:, 1]
        comparison_results.append({
            'Model': name,
            'Accuracy': round(accuracy_score(y_uploaded, yp), 4),
            'AUC': round(roc_auc_score(y_uploaded, ypr), 4),
            'Precision': round(precision_score(y_uploaded, yp), 4),
            'Recall': round(recall_score(y_uploaded, yp), 4),
            'F1': round(f1_score(y_uploaded, yp), 4),
            'MCC': round(matthews_corrcoef(y_uploaded, yp), 4)
        })

    comp_df = pd.DataFrame(comparison_results).set_index('Model')
    st.dataframe(
        comp_df.style.highlight_max(axis=0, color='#90EE90').format("{:.4f}"),
        use_container_width=True
    )

else:
    st.warning("👆 Please upload a CSV file to get started.")
    st.markdown(
        "You can find a sample test file at `data/test_data.csv` in the project directory."
    )
