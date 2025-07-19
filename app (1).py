import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# ----- Improved Custom Styling -----
st.set_page_config(page_title="Employee Salary Prediction", page_icon="💰", layout="wide")
st.markdown("""
    <style>
    body {
        background-color: #e1f5fe;
    }
    .main {
        background-color: #f4fcff!important;
    }
    .stButton>button {
        background-color: #1976D2;
        color: white;
        border-radius: 8px;
        font-size: 18px;
    }
    .css-1d391kg {
        background-color: #ffffff;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #e3f2fd;
    }
    </style>
    """, unsafe_allow_html=True)

# -- Sidebar with More Information --
st.sidebar.title("About The App")
st.sidebar.markdown("""
- *Elevate Your Salary Insights*
  - Gain next-level compensation intelligence with our Employee Salary Prediction platform—ideal for HR professionals, leaders, and job seekers across industries.

- *Instant Smart Salary Estimates*
  - Predict employee salaries instantly using advanced machine learning, tailored by experience, age, education, gender, and sector.

- *Accurate & Actionable*
  - Access salary figures in your local currency to power decision-making, negotiations, and workforce planning.

- *Interactive Data Analytics*
  - Visualize salary trends and patterns with intuitive charts and exploration tools.

- *Personalized Predictions*
  - Upload your own data to generate organization-specific, actionable insights.

- *Empowering for All*
  - Built for transparency and fairness, supporting teams and individuals in better compensation decisions.

- Move forward confidently—whether hiring, benchmarking, or planning your career—with evidence-based salary intelligence.
""")

st.sidebar.markdown("*Instructions:*\n1. Enter employee details\n2. Choose prediction model\n3. View predicted salary & similar profiles")
st.sidebar.markdown("---")
st.sidebar.info("")

# ---- Data Section (with Age and Gender for Demo) ----
data = {
    "YearsExperience": [0, 1, 2, 3, 4, 5, 7, 10, 12, 15, 18, 20, 22, 25, 30],
    "Degree": [
        "Bachelors", "Bachelors", "Bachelors", "Masters", "Masters", "Masters", "PhD", "PhD", "PhD",
        "Bachelors", "Masters", "PhD", "Bachelors", "Masters", "PhD"
    ],
    "Industry": [
        "Retail", "IT", "Finance", "Healthcare", "IT", "Finance", "Healthcare", "IT",
        "Finance", "Healthcare", "IT", "Finance", "Healthcare", "IT", "Finance"
    ],
    "Age": [21, 22, 24, 27, 29, 31, 35, 38, 41, 45, 48, 53, 56, 60, 65],
    "Gender": [
        "Male", "Female", "Male", "Female", "Other", "Male", "Other", "Female", "Male",
        "Other", "Female", "Other", "Male", "Female", "Other"
    ],
    "Salary": [
        9500, 23000, 34000, 52000, 65000, 105000, 215000, 395000, 575000,
        765000, 901000, 1050000, 1150000, 1235000, 1350000
    ]
}
df = pd.DataFrame(data)

# ---- Model Training (Include New Features) ----
def train_models(df):
    X = pd.get_dummies(df[["YearsExperience", "Degree", "Industry", "Age", "Gender"]])
    y = df["Salary"]
    lin_model = LinearRegression().fit(X, y)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    return lin_model, rf_model
lin_model, rf_model = train_models(df)

# ---- App Title & Introduction ----
st.title("💰 Employee Salary Prediction")
st.markdown("> Estimate employee salaries based on experience, education, age, gender, and sector.")

# ---- User Input Section ----
with st.form("prediction_form", clear_on_submit=False):
    st.markdown("#### 📋 Enter Employee Details Below")
    col1, col2, col3 = st.columns(3)
    with col1:
        years_exp = st.slider("Years of Experience", 0, 30, 2, 1)
        age = st.slider("Age", 20, 65, 25, 1)
    with col2:
        degree = st.selectbox("Degree Level", ["Bachelors", "Masters", "PhD"])
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    with col3:
        industry = st.selectbox("Industry", ["IT", "Finance", "Healthcare"])
        model_type = st.radio("Prediction Model",
            ["Linear Regression", "Random Forest"],
            index=1,
            help="Random Forest is robust for HR data."
        )
    submitted = st.form_submit_button("🔮 Predict Salary (in ₹)")

sim_df = pd.DataFrame()  # define sim_df with empty DataFrame as default

if submitted:
    # Prepare input
    input_df = pd.DataFrame({
        "YearsExperience": [years_exp],
        "Degree": [degree],
        "Industry": [industry],
        "Age": [age],
        "Gender": [gender],
    })
    input_encoded = pd.get_dummies(input_df)
    base_encoded = pd.get_dummies(df[["YearsExperience", "Degree", "Industry", "Age", "Gender"]])
    input_encoded = input_encoded.reindex(columns=base_encoded.columns, fill_value=0)

    # Predict salary (do not round)
    model = lin_model if model_type == "Linear Regression" else rf_model
    salary_pred = model.predict(input_encoded)[0]

    st.success(f"💸 Predicted Salary: ₹ {salary_pred:,.2f} per year")

    sim_df = df[
        (df["Degree"] == degree) &
        (df["Industry"] == industry) &
        (df["Gender"] == gender) &
        (abs(df["YearsExperience"] - years_exp) <= 1) &
        (abs(df["Age"] - age) <= 2)
    ]
    if not sim_df.empty:
        st.markdown("#### 👥 Similar Employees in Dataset")
        st.dataframe(sim_df)

# ---- Data Visualization Section ----
with st.expander("📊 See Salary Distribution Chart"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df, x="YearsExperience", y="Salary", hue="Degree", ax=ax)
    plt.title("Salary vs Years of Experience by Degree")
    plt.ylabel("Salary")
    plt.xlabel("Years of Experience")
    plt.tight_layout()
    st.pyplot(fig)

# ---- Footer ----
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#555;'>Crafted with Passion By Ayush Yele</div>",
    unsafe_allow_html=True)
