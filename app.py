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
# THEME TOGGLE
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
# SIDEBAR OPTIONS
# ---------------------------
st.sidebar.title("Controls")
show_data = st.sidebar.checkbox("Show Data")
show_graphs = st.sidebar.checkbox("Show Graphs")

# ---------------------------
# LOADING ANIMATION
# ---------------------------
with st.spinner("Training AI Model... 🤖"):
    time.sleep(1)

# ---------------------------
# DATA
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

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

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
    # CREDIT RISK METER
    # ---------------------------
    st.subheader("💳 Credit Risk Insight")

    score = acc * 100

    if score > 80:
        st.success("🟢 Low Risk Model (Highly Accurate)")
    elif score > 60:
        st.warning("🟡 Medium Risk Model")
    else:
        st.error("🔴 High Risk Model")

    # ---------------------------
    # CONFUSION MATRIX
    # ---------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, pred)

    fig = plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm")
    st.pyplot(fig)
  # Explanation
    st.write("This model predicts whether a customer is high or low credit risk.")
else:
    st.error("Target column not found!")