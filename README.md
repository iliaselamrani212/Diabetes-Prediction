  # 🩺 Diabetes Prediction – Kaggle Playground Series 2025   
 
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)    
![Kaggle](https://img.shields.io/badge/Kaggle-Playground-20BEFF)   
![License](https://img.shields.io/badge/License-MIT-green)  
![Status](https://img.shields.io/badge/Status-Completed-succ ess)       
    
This repository contains an end-to-end machine learning pipeline developed for the **Kaggle Playground Series – Season 5 Episode 12**.  
  
The objective is to predict the probability of a diabetes diagnosis using synthetic tabular medical data. This project focuses on **robust validation**, **advanced feature engineering**, and **ensemble learning**, designed as a professional portfolio project rather than a simple competition notebook .
      
---  
           
## 📌 Problem Statement     
          
| Attribute | Description |  
| :--- | :--- |    
| **Task** | Binary Classification (Predict diabetes probability) |     
| **Metric** | ROC-AUC |
| **Dataset** | Synthetic medical tabular data (Numerical + Categorical) |
| **Platform** | Kaggle Playground Series 2025 |

---

## 🧠 Project Highlights 

* ✔ **Complete Exploratory Data Analysis (EDA):** Deep dive into correlations and distributions.
* ✔ **High-order Interactions:** specialized engineering from 2-way up to 4-way feature interactions.
* ✔ **Leakage-safe Encoding:** Implemented robust Out-of-Fold (OOF) Target Encoding.
* ✔ **GPU Acceleration:** Utilized GPU-accelerated training for CatBoost, XGBoost, and LightGBM.
* ✔ **Stacking Ensemble:** Final optimization using Logistic Regression stacking on OOF predictions.
* ✔ **Reproducible:** Fully documented pipeline with seeded cross-validation.

---

## 🔍 Exploratory Data Analysis (EDA)

Before modeling, an extensive analysis was conducted .

**Key Findings:**
* **Data Quality:** No missing values in the synthetic dataset.
* **Demographics:** Diabetes prevalence shows a clear positive correlation with age.
* **Lifestyle:** Lower physical activity levels and diet scores are observed among diabetic patients.
* **Correlations:**
    * Strong link between `cholesterol_total` ↔ `ldl_cholesterol`.
    * Significant relationship between `bmi` ↔ `waist_to_hip_ratio`.

*These insights directly guided the interaction design in the feature engineering phase.*

---

## ⚙️ Feature Engineering

### 1️⃣ Categorical Handling
* **Native Support:** Used CatBoost's internal categorical processing.
* **Target Encoding:** Applied smooth mean encoding (OOF) specifically for high-cardinality interaction features.

### 2️⃣ Interaction Features
To capture complex non-linear relationships, I engineered specific feature combinations:
* **Depth:** Carefully selected 2-way, 3-way, and 4-way interactions.
* **Domain Knowledge:** Combined features like BMI, Cholesterol, Age, and Lifestyle.
* **Processing:** Converted interactions to categorical tokens and then applied target encoding.

### 3️⃣ Leakage Prevention
* Implemented a **Nested K-Fold strategy** for all target encoding steps to ensure no target information leaked into validation folds.

---

## 🤖 Models & Ensemble Strategy

### Base Models
* **CatBoostClassifier:** Leveraged for its Bayesian bootstrap and native handling of categorical features (GPU trained).
* **XGBoostClassifier:** Focused on the interaction-heavy feature set using the GPU histogram algorithm with custom regularization.
* **LightGBMClassifier:** Tuned extensively with Optuna to optimize large tree ensembles.

### Stacking Ensemble
* **Inputs:** Out-of-Fold (OOF) probability predictions from the three base models.
* **Meta-Model:** Logistic Regression.
* **Objective:** Maximize ROC-AUC via rank optimization, exploiting the low correlation between base learners.

---

## 🧪 Validation Strategy

* **Method:** Stratified K-Fold Cross-Validation (5 Folds).
* **Purpose:** Used for performance estimation, generating OOF predictions for stacking, and early stopping to prevent overfitting.

---

## 📈 Results

The stacking approach yielded significant improvements over single models due to the diversity of the base learners.

| Model | ROC-AUC (OOF) |
| :--- | :--- |
| **CatBoost** | Strong |
| **XGBoost** | Strong |
| **LightGBM** | Strong |
| **Stacked Ensemble** | **Best ✅** |

*Achieved a Top ~400 ranking on the Kaggle Leaderboard.*

---

## 💻 How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/iliaselamrani212/Diabetes-Prediction
    ```

2.  **Install dependencies** (ensure you have GPU support if running locally):
    ```bash
    pip install pandas numpy scikit-learn xgboost catboost lightgbm optuna matplotlib seaborn
    ```

