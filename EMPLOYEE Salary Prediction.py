import pandas as pd

data=pd.read_csv(r"/content/adult 3.csv")

data

data.shape

print(data.info())
print(data.isnull().sum())

print(data.describe())


print(data.select_dtypes(include='number').corr())

print(data.dtypes)

from sklearn.preprocessing import LabelEncoder



import matplotlib.pyplot as plt

plt.boxplot(data['age'])
plt.show()

for i, col in enumerate(data.columns):
    print(f"{i}: '{col}'")

data.columns = data.columns.str.strip()

print(data.columns.tolist())

from sklearn.preprocessing import LabelEncoder

# Initialize label encoders
le_education = LabelEncoder()
le_occupation = LabelEncoder()

# Convert columns to strings (optional but safe)
data['education'] = le_education.fit_transform(data['education'].astype(str))
data['occupation'] = le_occupation.fit_transform(data['occupation'].astype(str))

X = data.drop('income', axis=1)
y = data['income']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Identify categorical columns (excluding 'income' which is the target)
categorical_cols = data.select_dtypes(include='object').columns.tolist()
if 'income' in categorical_cols:
    categorical_cols.remove('income')

# Apply One-Hot Encoding to categorical columns
X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

# Align columns after one-hot encoding - this is crucial to ensure both train and test sets have the same columns
train_cols = X_train_encoded.columns
test_cols = X_test_encoded.columns

missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test_encoded[c] = 0

missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X_train_encoded[c] = 0

X_test_encoded = X_test_encoded[train_cols] # Ensure the order of columns is the same


# List of models to test
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier()
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train_encoded, y_train)
    y_pred = model.predict(X_test_encoded)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Plot
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['<=50K','>50K'], yticklabels=['<=50K','>50K'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
plt.grid()
plt.show()


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

# 1. Strip column names of extra spaces
data.columns = data.columns.str.strip()

# 2. Encode categorical columns
from sklearn.preprocessing import LabelEncoder

cat_cols = ['workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'race', 'gender', 'native-country', 'income']

le = LabelEncoder()
for col in cat_cols:
    data[col] = le.fit_transform(data[col].astype(str))

# 3. Prepare features and labels
X = data.drop('income', axis=1)
y = data['income']

# 4. Train/Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

importances = model.feature_importances_
features = X.columns

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=features)
plt.title('Feature Importances (Random Forest Classifier)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc * 100:.2f}%")

!pip install xgboost  # Run once in Colab or Jupyter

from xgboost import XGBClassifier

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {acc_xgb * 100:.2f}%")

!pip install catboost

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Drop rows with missing values
data.dropna(inplace=True)

# Split data
X = data.drop('income', axis=1)
y = data['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical columns
categorical_cols = X_train.select_dtypes(include='object').columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

# Create transformers for categorical and numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Create a pipeline with the preprocessor and the model
model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
cr = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(cr)

print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))

import pandas as pd

# Load the data
data = pd.read_csv(r"/content/adult 3.csv")

data=data.drop(columns=['education']) #redundant features removal

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load your data
df = pd.read_csv("/content/adult 3.csv")

# Replace '?' with 'others' and 'notlisted' in 'occupation' and 'workclass'
df['occupation'].replace({'?':'others'},inplace=True)
df['workclass'].replace({'?':'notlisted'},inplace=True)

# Remove rows with 'Without-pay' and 'Never-worked' in 'workclass'
df=df[df['workclass']!= 'Without-pay']
df=df[df['workclass']!= 'Never-worked']

# Remove rows with specific 'education' values
df=df[df['education']!= '5th-6th']
df=df[df['education']!= '1st-4th']
df=df[df['education']!= 'Preschool']

# Drop the 'education' column
df.drop(columns=['education'],inplace=True)

# Filter age outliers
df=df[(df['age']<=75)& (df['age']>=17)]


# Separate features and target
X = df.drop('income', axis=1)
y = df['income']

# Apply Label Encoding to categorical features
categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
encoder = LabelEncoder()
for col in categorical_cols:
    X[col] = encoder.fit_transform(X[col])

# Encode the target variable 'income'
le_income = LabelEncoder()
y_encoded = le_income.fit_transform(y)


# Split data using the encoded target variable
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


# Train
model = CatBoostClassifier(verbose=100, iterations=500)
model.fit(X_train, y_train_encoded)

# Predict
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test_encoded, y_pred)
print(f"Accuracy: {acc:.4f}")

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "LogisticRegression": LogisticRegression(),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "GradientBoosting": GradientBoostingClassifier()
}

results = {}

for name, model in models.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
plt.bar(results.keys(), results.values(), color='skyblue')
plt.ylabel('Accuracy Score')
plt.title('Model Comparison')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "GradientBoosting": GradientBoostingClassifier()
}

results = {}

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name}: {acc:.4f}")

# Get best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\n✅ Best model: {best_model_name} with accuracy {results[best_model_name]:.4f}")

# Save the best model
joblib.dump(best_model, "best_model.pkl")
print("✅ Saved best model as best_model.pkl")

import pandas as pd

# Load the data
data = pd.read_csv(r"/content/adult 3.csv")

# 1. Strip column names of extra spaces
data.columns = data.columns.str.strip()

# 2. Encode categorical columns
from sklearn.preprocessing import LabelEncoder

cat_cols = ['workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'race', 'gender', 'native-country', 'income']

le = LabelEncoder()
for col in cat_cols:
    data[col] = le.fit_transform(data[col].astype(str))

# 3. Prepare features and labels
X = data.drop('income', axis=1)
y = data['income']

# 4. Train/Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

!pip install streamlit scikit-learn pandas matplotlib seaborn

!pip install streamlit pyngrok

!ngrok authtoken 303Lt494bDD7ce1qUbyi4G8A224_2rkK7tVzqS8aDrWpyFdE3

!pip install -q streamlit
!npm install -g localtunnel

import os
import threading

def run_streamlit():
 os.system ('streamlit run app.py --server.port 8501')

 thread=threading.Thread(target=run_streamlit)
 thread.start()

!streamlit run app.py &>/dev/null &

!pip install catboost

%%writefile app.py
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

%%writefile app.py
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

!streamlit run app.py &>/dev/null &

!pip install pyngrok

from pyngrok import ngrok

# Optional: disconnect any previous tunnel
ngrok.kill()

# Replace "YOUR_NGROK_AUTH_TOKEN" with your actual ngrok authtoken
!ngrok authtoken "303Lt494bDD7ce1qUbyi4G8A224_2rkK7tVzqS8aDrWpyFdE3"


# Create new tunnel
public_url = ngrok.connect(8501)
print(" Streamlit app is live at:", public_url)

code = """
import streamlit as st
# Your Streamlit app code here
"""
with open("app.py", "w") as file:
    file.write(code)



%%writefile app.py
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

!pip freeze > requirements.txt

from google.colab import files
files.download('app.py')
files.download('requirements.txt')



