import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Styling & page config
st.set_page_config(page_title="Employee Salary Prediction", page_icon="💰", layout="wide")
st.markdown("""
    <style>
    body { background-color: #e1f5fe; }
    .main { background-color: #f4fcff!important; }
    .stButton>button { background-color: #1976D2; color: white; border-radius: 8px; font-size: 18px; }
    .css-1d391kg { background-color: #ffffff; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .sidebar .sidebar-content { background-color: #e3f2fd; }
    </style>
    """, unsafe_allow_html=True)

# Sidebar info
st.sidebar.title("About The App")
st.sidebar.markdown("""
- *Predict salaries based on experience, education, age, gender, and sector.*
- Built for HR professionals, managers, and job seekers.
- Upload your organization data for tailored predictions.
""")
st.sidebar.markdown("*Instructions:* Enter employee details, choose model, and get annual salary prediction.")

# --- Demo data generation ---
np.random.seed(42)
ages = np.arange(18, 66)
degrees = ['Bachelors', 'Masters', 'PhD']
industries = ['IT', 'Finance', 'Healthcare', 'Retail']
genders = ['Male', 'Female', 'Other']

data = {"Age": [], "YearsExperience": [], "Degree": [], "Industry": [], "Gender": [], "Salary": []}
for age in ages:
    for _ in range(np.random.randint(2, 4)):
        max_exp = max(0, age - 18)
        years_exp = np.random.randint(0, max_exp + 1)
        degree = np.random.choice(degrees, p=[0.6, 0.3, 0.1])
        industry = np.random.choice(industries)
        gender = np.random.choice(genders, p=[0.48,0.48,0.04])
        deg_base = {'Bachelors': 250000, 'Masters': 400000, 'PhD': 600000}[degree]
        ind_mult = {'IT':1.3, 'Finance':1.35, 'Healthcare':1.12, 'Retail':0.93}[industry]
        exp_factor = 1 + (years_exp ** 1.15) / 13
        age_bonus = 1 + max(0, (age - 25)) * 0.012
        noise = np.random.normal(0, 45000)
        salary = deg_base * ind_mult * exp_factor * age_bonus + noise
        salary = max(150000, float(np.round(salary, 2)))
        data["Age"].append(age)
        data["YearsExperience"].append(years_exp)
        data["Degree"].append(degree)
        data["Industry"].append(industry)
        data["Gender"].append(gender)
        data["Salary"].append(salary)
df = pd.DataFrame(data)

# --- Model training ---
def train_models(df):
    X = pd.get_dummies(df[["YearsExperience", "Degree", "Industry", "Age", "Gender"]])
    y = df["Salary"]
    lin_model = LinearRegression().fit(X, y)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    return lin_model, rf_model
lin_model, rf_model = train_models(df)

# --- App Layout ---
st.title("💰 Employee Salary Prediction")
st.markdown("> Estimate *annual* employee salaries based on several features.")

with st.form("prediction_form", clear_on_submit=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        years_exp = st.slider("Years of Experience", 0, 47, 2, 1)
        age = st.slider("Age", 18, 65, 25, 1)
    with col2:
        degree = st.selectbox("Degree Level", degrees)
        gender = st.selectbox("Gender", genders)
    with col3:
        industry = st.selectbox("Industry", industries)
        model_type = st.radio("Prediction Model", ["Linear Regression", "Random Forest"], index=1)
    submitted = st.form_submit_button("🔮 Predict Salary (in ₹, annual)")

sim_df = pd.DataFrame()
if submitted:
    # Prepare input
    input_df = pd.DataFrame({"YearsExperience": [years_exp], "Degree":[degree],
                             "Industry":[industry], "Age":[age], "Gender":[gender]})
    input_encoded = pd.get_dummies(input_df)
    base_encoded = pd.get_dummies(df[["YearsExperience", "Degree", "Industry", "Age", "Gender"]])
    input_encoded = input_encoded.reindex(columns=base_encoded.columns, fill_value=0)
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


st.markdown("<div style='text-align:center;color:#555;'>Crafted with Passion By Ayush Yele</div>", unsafe_allow_html=True)
