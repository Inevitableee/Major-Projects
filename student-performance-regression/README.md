# 🎓 Student Performance Predictor — Linear Regression

A machine learning regression project that predicts a student's **Performance Index** based on study habits, prior academic performance, sleep, and extracurricular involvement.

---

## 📌 Overview

This project builds a supervised regression model to answer a simple question: *can we predict how well a student will perform based on measurable daily habits?* Using a dataset of 10,000 student records, the notebook walks through the full ML workflow — from raw data to a saved, reusable model — achieving **98.89% accuracy (R² score)**.

---

## 📊 Dataset

| | |
|---|---|
| **Source** | [Student_Performance.csv](https://raw.githubusercontent.com/DataScience75/Cohort130/refs/heads/main/Student_Performance.csv) |
| **Records** | 10,000 |
| **Target Variable** | Performance Index (10–100) |

**Features used:**

| Feature | Description |
|---|---|
| Hours Studied | Total hours spent studying |
| Previous Scores | Scores from earlier tests |
| Extracurricular Activities | Participation (Yes/No) |
| Sleep Hours | Average daily sleep |
| Sample Question Papers Practiced | Number of practice papers completed |

---

## 🛠️ Tech Stack

- **Python 3.11**
- **pandas** — data loading & manipulation
- **seaborn / matplotlib** — exploratory data analysis & visualization
- **scikit-learn** — model training, splitting, evaluation (`LinearRegression`, `train_test_split`, `r2_score`)
- **joblib** — model persistence

---

## 🔍 Workflow

1. **Load & Explore the Data** — shape, missing values, data types, and correlation analysis
2. **Preprocessing** — encoded categorical feature (`Extracurricular Activities`) to numeric
3. **Exploratory Data Analysis** — correlation heatmap and distribution plots across all variables
4. **Train/Test Split** — 80% training (8,000 records) / 20% testing (2,000 records)
5. **Model Training** — fit a `LinearRegression` model on the training set
6. **Model Validation** — evaluated using R² score
7. **Model Persistence** — saved the trained model with `joblib` for reuse
8. **Inference** — loaded the saved model and generated predictions on new, unseen input

---

## 📈 Results

| Metric | Score |
|---|---|
| **R² Accuracy** | **98.89%** |

The model demonstrates a strong linear relationship between the selected features and student performance, with `Previous Scores` and `Hours Studied` emerging as key predictors during correlation analysis.

---

## 🚀 Getting Started

**1. Clone the repository**
```bash
git clone <your-repo-url>
cd <repo-folder>
```

**2. Install dependencies**
```bash
pip install pandas seaborn matplotlib scikit-learn joblib
```

**3. Run the notebook**
```bash
jupyter notebook ML_Regression.ipynb
```

---

## 📁 Repository Structure

```
├── ML_Regression.ipynb      # Full notebook: EDA, training, evaluation, inference
├── README.md                 # Project documentation
```

---

## 🔮 Future Improvements

- Experiment with regularized models (Ridge/Lasso) and compare performance
- Add cross-validation for more robust accuracy estimates
- Deploy the model behind a simple API (e.g., Flask/FastAPI) for live predictions

---

## 👤 Author

Feel free to connect or reach out with questions about this project.
