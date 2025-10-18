# 🏠 House Price Prediction Project

## 📘 Overview
This project predicts **house prices** using a machine learning model trained on structured housing data.  
It applies **data preprocessing, feature scaling, and the XGBoost algorithm** to achieve accurate predictions.

In addition, it includes a feature to **predict the price of a new house** using user-defined property details.

> ⚠️ Note: The dataset file (`houses.csv`) is **not included** in this repository.  
> Please place your own dataset under `data/houses.csv` before running the script.


---

## ⚙️ Features
- Data cleaning and preprocessing  
- Feature selection and normalization  
- Model training using **XGBoost Regressor**  
- Performance evaluation (MAE, RMSE, R²)  
- Visualization of predicted vs. actual prices  
- Predicting a new house price after training  

---

## 🧠 Machine Learning Workflow
1. Load and clean dataset (`houses.csv`)  
2. Split data into training and testing sets  
3. Scale numerical features using `StandardScaler`  
4. Train an XGBoost regression model  
5. Evaluate performance  
6. Predict new house price using user input  

---

## 🧰 Technologies Used
- **Python 3.8+**  
- **Pandas** — data manipulation  
- **NumPy** — numerical computation  
- **Scikit-learn** — data preprocessing & metrics  
- **XGBoost** — regression model  
- **Matplotlib** — data visualization  
- **Joblib** — model serialization  

---

🤖 AI Assistance

This project was developed with the help of ChatGPT to optimize code structure, improve documentation,
and generate initial model configurations.
All final code was reviewed and tested manually for accuracy.


