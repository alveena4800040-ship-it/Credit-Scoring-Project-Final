import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Credit Scoring AI 💳",
    page_icon="💳",
    layout="wide"
)

# ---------------------------
# THEME
# ---------------------------
theme = st.sidebar.radio("🎨 Theme", ["Light", "Dark"])

if theme == "Dark":
    st.markdown("""
        <style>
        body { background-color: #0E1117; color: white; }
        </style>
    """, unsafe_allow_html=True)

# ---------------------------
# TITLE
# ---------------------------
st.markdown("""
    <h1 style='text-align:center; color:#6C63FF;'>
        💳 Credit Scoring AI Dashboard
    </h1>
""", unsafe_allow_html=True)

st.write("")

# ---------------------------
# UPLOAD
# ---------------------------
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV", type=["csv"])

default_df = pd.read_csv("german_credit_data.csv")

if uploaded_file:
    new_df = pd.read_csv(uploaded_file)
    df = pd.concat([default_df, new_df], ignore_index=True)
else:
    df = default_df

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.title("Controls")
show_data = st.sidebar.checkbox("Show Data")
show_graphs = st.sidebar.checkbox("Show Graphs")

# ---------------------------
# LOADING
# ---------------------------
with st.spinner("Training AI Model... 🤖"):
    time.sleep(1)

# ---------------------------
# SHOW DATA
# ---------------------------
if show_data:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

# ---------------------------
# GRAPHS
# ---------------------------
if show_graphs:
    col1, col2 = st.columns(2)

    with col1:
        if "alter" in df.columns:
            st.subheader("Age Distribution")
            st.bar_chart(df["alter"])

    with col2:
        if "kredit" in df.columns:
            st.subheader("Credit Distribution")
            st.bar_chart(df["kredit"].value_counts())

# ---------------------------
# MODEL
# ---------------------------
if "kredit" in df.columns:

    df["kredit"] = df["kredit"].replace({2: 0})

    X = df.drop("kredit", axis=1)
    y = df["kredit"]

    # Convert categorical → numeric
    X = pd.get_dummies(X, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # ✅ FIXED MODEL (BALANCED)
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    # Predict
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    # ---------------------------
    # METRICS
    # ---------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.metric("📊 Accuracy", f"{acc*100:.2f}%")

    with col2:
        st.metric("📁 Records", df.shape[0])

    # ---------------------------
    # RISK LEVEL (UPDATED)
    # ---------------------------
    st.subheader("💳 Credit Risk Insight")

    if acc > 0.7:
        st.success("🟢 Model is Reliable")
    else:
        st.error("🔴 Model needs improvement")

    # ---------------------------
    # CONFUSION MATRIX
    # ---------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, pred)

    fig = plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm")
    st.pyplot(fig)

    st.write("This model predicts whether a customer is high or low credit risk.")

    # =====================================================
    # 🔥 MANUAL PREDICTION
    # =====================================================
    st.divider()
    st.subheader("🧾 Predict Credit Risk (Manual Entry)")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 100, 35)
        amount = st.number_input("Loan Amount", 0, 100000, 2000)
        duration = st.number_input("Duration (months)", 1, 60, 12)

    with col2:
        installment = st.number_input("Installment Rate", 1, 10, 2)
        residence = st.number_input("Residence Time", 1, 10, 3)
        employment = st.number_input("Employment (years category)", 0, 10, 2)

    if st.button("🔮 Predict"):

        try:
            input_df = pd.DataFrame({
                "alter": [age],
                "hoehe": [amount],
                "laufzeit": [duration],
                "rate": [installment],
                "wohnzeit": [residence],
                "beszeit": [employment]
            })

            input_df = pd.get_dummies(input_df)
            input_df = input_df.reindex(columns=X.columns, fill_value=0)

            prediction = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]

            st.write(f"📊 Probability of Good Credit: {prob*100:.2f}%")

            if prediction == 1:
                st.success("✅ Good Credit Risk")
            else:
                st.error("❌ Bad Credit Risk")

        except:
            st.error("⚠️ Prediction failed. Check input values.")

else:
    st.error("Target column 'kredit' not found!")