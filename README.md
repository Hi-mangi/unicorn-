import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Startups (1).csv")
    df.dropna(subset=["Valuation", "Num_Investors", "Year Joined", "Country", "Industry"], inplace=True)
    return df

df = load_data()

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Unicorn Valuation Predictor", layout="wide")

# -----------------------------
# Header
# -----------------------------
st.title("🦄 Unicorn Startup Valuation Predictor")
st.markdown("Predict a startup's valuation based on key attributes. Animated insights included!")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Input Features for Prediction")

num_investors = st.sidebar.slider("Number of Investors", 1, 50, 5)
year_joined = st.sidebar.slider("Year Joined Unicorn Club", 2010, 2025, 2020)
country = st.sidebar.selectbox("Country", sorted(df["Country"].dropna().unique()))
industry = st.sidebar.selectbox("Industry", sorted(df["Industry"].dropna().unique()))

# -----------------------------
# Preprocessing & Model Training
# -----------------------------
features = ["Num_Investors", "Year Joined", "Country", "Industry"]
target = "Valuation"

X = df[features]
y = df[target]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Country", "Industry"]),
    ],
    remainder="passthrough"
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

r2 = r2_score(y_test, pipeline.predict(X_test))

# -----------------------------
# Prediction
# -----------------------------
input_df = pd.DataFrame({
    "Num_Investors": [num_investors],
    "Year Joined": [year_joined],
    "Country": [country],
    "Industry": [industry]
})

prediction = pipeline.predict(input_df)[0]

st.subheader("💰 Predicted Valuation")
st.metric(label="Estimated Valuation (in billions)", value=f"${prediction:.2f}B")
st.caption(f"Model R² score on test data: `{r2:.2f}`")

# -----------------------------
# Animated Chart
# -----------------------------
st.subheader("📊 Animated Valuation Trend by Country")

fig = px.scatter(df,
    x="Year Joined",
    y="Valuation",
    size="Num_Investors",
    color="Country",
    animation_frame="Year Joined",
    hover_name="Company",
    size_max=45,
    log_y=True,
    title="Unicorn Valuations Over Time by Country"
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, Plotly & Sklearn")
your-repo/
├── streamlit_app.py
└── Startups (1).csv


