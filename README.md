import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(page_title="🦄 Unicorn Startup Dashboard", layout="wide")

# ----------------------------
# Load Dataset
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Startups (1).csv")
    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
    print(df.columns.tolist())  # Optional: See column names in terminal or logs
    df.dropna(subset=["Valuation", "Num_Investors", "Year Joined", "Country", "Industry"], inplace=True)
    return df

# ----------------------------
# Header
# ----------------------------
st.title("🦄 Unicorn Startups Analytics & Prediction App")
st.markdown("This interactive app explores unicorn startup trends and predicts valuations. Built with 💡 Streamlit + Plotly + Sklearn")

# ----------------------------
# Dataset Preview
# ----------------------------
with st.expander("📂 View Raw Dataset"):
    st.dataframe(df)

# ----------------------------
# Sidebar Filters
# ----------------------------
st.sidebar.header("🔎 Filter the Data")
selected_country = st.sidebar.multiselect("Select Country", options=df['Country'].unique(), default=None)
selected_industry = st.sidebar.multiselect("Select Industry", options=df['Industry'].unique(), default=None)

filtered_df = df.copy()
if selected_country:
    filtered_df = filtered_df[filtered_df['Country'].isin(selected_country)]
if selected_industry:
    filtered_df = filtered_df[filtered_df['Industry'].isin(selected_industry)]

# ----------------------------
# Summary Cards
# ----------------------------
col1, col2, col3 = st.columns(3)
col1.metric("🔢 Total Unicorns", f"{len(filtered_df)}")
col2.metric("🌍 Countries Covered", f"{filtered_df['Country'].nunique()}")
col3.metric("💼 Industries", f"{filtered_df['Industry'].nunique()}")

# ----------------------------
# Valuation Distribution Chart
# ----------------------------
st.subheader("💰 Valuation Distribution by Country")
fig1 = px.box(filtered_df, x="Country", y="Valuation", color="Country", points="all")
st.plotly_chart(fig1, use_container_width=True)

# ----------------------------
# Industry Pie Chart
# ----------------------------
st.subheader("🧠 Top Industries")
industry_count = filtered_df['Industry'].value_counts().nlargest(10)
fig2 = px.pie(industry_count, names=industry_count.index, values=industry_count.values, title="Top 10 Industries")
st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# Animated Chart
# ----------------------------
st.subheader("📊 Animated Valuation Growth Over Years")
fig3 = px.scatter(filtered_df,
                  x="Year Joined",
                  y="Valuation",
                  size="Num_Investors",
                  color="Country",
                  animation_frame="Year Joined",
                  hover_name="Company",
                  log_y=True,
                  title="Valuation Bubble Animation Over Years",
                  size_max=60)
st.plotly_chart(fig3, use_container_width=True)

# ----------------------------
# Prediction Section
# ----------------------------
st.subheader("🔮 Valuation Prediction Tool")

with st.form("prediction_form"):
    st.markdown("Enter the details below to predict a startup's valuation:")

  col1, col2 = st.columns(2)
 num_investors = col1.slider("Number of Investors", 1, 50, 5)
    year_joined = col2.slider("Year Joined", 2010, 2025, 2020)
    country = col1.selectbox("Country", sorted(df["Country"].unique()))
    industry = col2.selectbox("Industry", sorted(df["Industry"].unique()))
    submit = st.form_submit_button("Predict Valuation 💸")

  if submit:
        # Model
        model_df = df[["Num_Investors", "Year Joined", "Country", "Industry", "Valuation"]]
        X = model_df.drop("Valuation", axis=1)
        y = model_df["Valuation"]

preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), ["Country", "Industry"]),
            ],
            remainder="passthrough"
        )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
       import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(page_title="🦄 Unicorn Startup Dashboard", layout="wide")

# ----------------------------
# Load Dataset
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Startups (1).csv")
    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
    print(df.columns.tolist())  # Optional: See column names in terminal or logs
    df.dropna(subset=["Valuation", "Num_Investors", "Year Joined", "Country", "Industry"], inplace=True)
    return df

# ----------------------------
# Header
# ----------------------------
st.title("🦄 Unicorn Startups Analytics & Prediction App")
st.markdown("This interactive app explores unicorn startup trends and predicts valuations. Built with 💡 Streamlit + Plotly + Sklearn")

# ----------------------------
# Dataset Preview
# ----------------------------
with st.expander("📂 View Raw Dataset"):
    st.dataframe(df)

# ----------------------------
# Sidebar Filters
# ----------------------------
st.sidebar.header("🔎 Filter the Data")
selected_country = st.sidebar.multiselect("Select Country", options=df['Country'].unique(), default=None)
selected_industry = st.sidebar.multiselect("Select Industry", options=df['Industry'].unique(), default=None)

filtered_df = df.copy()
if selected_country:
    filtered_df = filtered_df[filtered_df['Country'].isin(selected_country)]
if selected_industry:
    filtered_df = filtered_df[filtered_df['Industry'].isin(selected_industry)]

# ----------------------------
# Summary Cards
# ----------------------------
col1, col2, col3 = st.columns(3)
col1.metric("🔢 Total Unicorns", f"{len(filtered_df)}")
col2.metric("🌍 Countries Covered", f"{filtered_df['Country'].nunique()}")
col3.metric("💼 Industries", f"{filtered_df['Industry'].nunique()}")

# ----------------------------
# Valuation Distribution Chart
# ----------------------------
st.subheader("💰 Valuation Distribution by Country")
fig1 = px.box(filtered_df, x="Country", y="Valuation", color="Country", points="all")
st.plotly_chart(fig1, use_container_width=True)

# ----------------------------
# Industry Pie Chart
# ----------------------------
st.subheader("🧠 Top Industries")
industry_count = filtered_df['Industry'].value_counts().nlargest(10)
fig2 = px.pie(industry_count, names=industry_count.index, values=industry_count.values, title="Top 10 Industries")
st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# Animated Chart
# ----------------------------
st.subheader("📊 Animated Valuation Growth Over Years")
fig3 = px.scatter(filtered_df,
                  x="Year Joined",
                  y="Valuation",
                  size="Num_Investors",
                  color="Country",
                  animation_frame="Year Joined",
                  hover_name="Company",
                  log_y=True,
                  title="Valuation Bubble Animation Over Years",
                  size_max=60)
st.plotly_chart(fig3, use_container_width=True)

# ----------------------------
# Prediction Section
# ----------------------------
st.subheader("🔮 Valuation Prediction Tool")

with st.form("prediction_form"):
    st.markdown("Enter the details below to predict a startup's valuation:")

 col1, col2 = st.columns(2)
    num_investors = col1.slider("Number of Investors", 1, 50, 5)
    year_joined = col2.slider("Year Joined", 2010, 2025, 2020)
    country = col1.selectbox("Country", sorted(df["Country"].unique()))
    industry = col2.selectbox("Industry", sorted(df["Industry"].unique()))
    submit = st.form_submit_button("Predict Valuation 💸")

   if submit:
       # Model
        model_df = df[["Num_Investors", "Year Joined", "Country", "Industry", "Valuation"]]
        X = model_df.drop("Valuation", axis=1)
        y = model_df["Valuation"]

   preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), ["Country", "Industry"]),
            ],
            remainder="passthrough"
        )

  pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=100, random_state=42))
        ])

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)
        r2 = r2_score(y_test, pipeline.predict(X_test))

   input_df = pd.DataFrame({
        "Num_Investors": [num_investors],
            "Year Joined": [year_joined],
            "Country": [country],
            "Industry": [industry]
        })

   predicted_valuation = pipeline.predict(input_df)[0]

 st.success(f"✅ Predicted Valuation: **${predicted_valuation:.2f} Billion**")
        st.caption(f"Model R² Score: `{r2:.2f}`")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("© 2025 Unicorn Analytics · Built with ❤️ by OpenAI & You")
     ("model", RandomForestRegressor(n_estimators=100, random_state=42))
        ])
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)
        r2 = r2_score(y_test, pipeline.predict(X_test))
        
   input_df = pd.DataFrame({
            "Num_Investors": [num_investors],
 "Year Joined": [year_joined],
            "Country": [country],
  "Industry": [industry]
})

predicted_valuation = pipeline.predict(input_df)[0]

st.success(f"✅ Predicted Valuation: **${predicted_valuation:.2f} Billion**")
 st.caption(f"Model R² Score: `{r2:.2f}`")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("© 2025 Unicorn Analytics · Built with ❤️ by OpenAI & You")
