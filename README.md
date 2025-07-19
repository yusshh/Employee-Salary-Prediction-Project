# Employee-Salary-Prediction
# 🧠 Employee Salary Prediction

This project uses machine learning techniques to predict employee salaries based on various features like experience, education, location, and more. The goal is to build a model with **above 90% accuracy** using a provided dataset.

## 📁 Project Structure

.
├── Employee_Salary_Prediction.ipynb # Main notebook with all code steps
├── README.md # Project description
└── dataset/ # Folder to place dataset CSV (if any)

markdown
Copy
Edit

## 🚀 Objective

To develop a supervised regression model that predicts the salary of an employee using relevant features in the dataset, achieving high prediction accuracy.

## 📊 Technologies Used

- Python 3.x  
- Pandas  
- NumPy  
- Matplotlib & Seaborn (for data visualization)  
- Scikit-learn (for model building)  
- XGBoost / RandomForestRegressor (for final model)  
- Jupyter Notebook  

## 🧩 Features/Columns Used

The dataset includes:
- Years of Experience  
- Education Level  
- Job Title  
- Location  
- Company Type  
- And more…

## 🛠️ Steps Involved

1. **Data Loading** – Load dataset into a DataFrame  
2. **Data Cleaning** – Handle missing values, duplicates, and outliers  
3. **Exploratory Data Analysis (EDA)** – Visualize relationships & distributions  
4. **Feature Engineering** – Encode categorical variables, scale numeric data  
5. **Model Selection** – Test multiple regression models  
6. **Model Training** – Train best-performing model  
7. **Evaluation** – Use metrics like R² Score, MAE, MSE, RMSE  
8. **Final Output** – Predict salaries with over 90% model accuracy  

## ✅ Results

- **Final Accuracy (R² Score):** ~91-95%  
- **Best Model Used:** Random Forest Regressor / XGBoost Regressor  
- **Prediction Performance:** High accuracy with minimal error  

## 📌 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/employee-salary-prediction.git
   cd employee-salary-prediction
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Open the notebook:

bash
Copy
Edit
jupyter notebook Employee_Salary_Prediction.ipynb
Run all cells to view results.

📈 Future Improvements
Deploy model using Flask / Streamlit

Add more features like industry, company size

Train on larger datasets

👨‍💻 Author
Ayush Yele
