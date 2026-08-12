# Healthcare-Predictive-Analytics-Diabetes-Risk-Prediction-Using-Machine-Learning

# 🩺 Healthcare Predictive Analytics — Diabetes Risk Prediction

A machine learning project that predicts **diabetes status based on health and lifestyle indicators** using classification algorithms. The project compares **Logistic Regression** and **Random Forest** models and performs feature-importance analysis to understand which health indicators contribute most to the predictions.

> ⚠️ **Disclaimer:** This project is for educational and research purposes only. It is not a clinical diagnostic system and should not be used as a substitute for professional medical advice.

---

## 📌 Project Overview

Diabetes is a major health condition influenced by several demographic, lifestyle, and health-related factors. Machine learning can be used to identify patterns in these factors and estimate whether a person belongs to a diabetes-risk category.

This project uses the **Diabetes Health Indicators dataset from Kaggle** and develops a binary classification system.

### Project workflow

```text
Kaggle Healthcare Dataset
          ↓
     Data Cleaning
          ↓
   Remove Duplicates
          ↓
  Select 50,000 Records
          ↓
   Feature / Target Split
          ↓
    Train-Test Split
          ↓
     Normalization
          ↓
 ┌─────────────────────┐
 │ Logistic Regression │
 │ Random Forest       │
 └──────────┬──────────┘
            ↓
     Model Evaluation
            ↓
    Feature Importance
            ↓
      Best Model
            ↓
   Diabetes Prediction
```

---

# 🎯 Objectives

The main objectives of this project are:

* Develop a machine learning model for diabetes prediction.
* Use health and lifestyle indicators as predictive features.
* Preprocess healthcare data for machine learning.
* Normalize features where appropriate.
* Compare different classification algorithms.
* Evaluate models using multiple performance metrics.
* Identify important health indicators using feature importance analysis.
* Save the trained model for future deployment.
* Demonstrate responsible and ethical use of healthcare data.

---

# 📊 Dataset

The project uses the **Diabetes Health Indicators Dataset**, obtained from Kaggle.

The original dataset contains a large number of health-related records collected from the **Behavioral Risk Factor Surveillance System (BRFSS)**.

For this project, a **50,000-record subset** was used to make model training computationally manageable while retaining a substantial amount of data for machine learning.

### Target variable

```text
Diabetes_binary
```

The target represents:

| Value | Meaning     |
| ----: | ----------- |
|   `0` | No diabetes |
|   `1` | Diabetes    |

The selected dataset is a **balanced 50/50 version**, which helps reduce the impact of class imbalance during model training and evaluation.

---

# 🧾 Features

The dataset contains health, demographic, and lifestyle indicators.

Examples include:

* High blood pressure
* High cholesterol
* Cholesterol check
* Body Mass Index (BMI)
* Smoking
* Stroke history
* Heart disease or coronary heart disease
* Physical activity
* Fruit consumption
* Vegetable consumption
* Heavy alcohol consumption
* Healthcare coverage
* Difficulty accessing healthcare
* General health
* Mental health
* Physical health
* Walking difficulty
* Gender
* Age
* Education
* Income

These features are used to identify patterns associated with diabetes status.

---

# 🧹 Data Preprocessing

Several preprocessing steps were performed.

### 1. Dataset loading

The Kaggle CSV dataset was loaded using Pandas.

### 2. Missing-value checking

The dataset was checked for missing values before model training.

### 3. Duplicate removal

Duplicate records were identified and removed.

### 4. Dataset sampling

A subset of **50,000 records** was selected from the larger dataset.

Stratified sampling was used so that the target-class distribution remained consistent.

### 5. Feature and target separation

The target variable was separated from the health indicators:

```text
X = Health Features
y = Diabetes_binary
```

### 6. Train/Test Split

The dataset was divided into:

```text
80% → Training data
20% → Testing data
```

Stratified splitting was used to maintain the class distribution.

### 7. Feature normalization

`StandardScaler` was used for the Logistic Regression model.

Normalization places features on a comparable scale and helps Logistic Regression perform effectively.

The scaler was fitted only on the training data to avoid **data leakage**.

---

# 🤖 Machine Learning Models

Two classification algorithms were evaluated.

## 1. Logistic Regression

Logistic Regression was used as a baseline classification model.

It estimates the probability that a record belongs to the diabetes class.

Advantages:

* Simple
* Fast
* Interpretable
* Suitable for binary classification
* Provides probability estimates
* Coefficients can be analyzed for feature importance

---

## 2. Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees.

Advantages:

* Handles nonlinear relationships
* Captures feature interactions
* Does not require feature normalization
* Provides feature-importance scores
* Robust for structured healthcare datasets

---

# 📈 Model Performance

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

### Final results

| Model                   |   Accuracy |  Precision |     Recall |   F1 Score |    ROC-AUC |
| ----------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| **Logistic Regression** | **74.50%** | **0.7380** | **0.7726** | **0.7549** | **0.8174** |
| Random Forest           |     74.13% |     0.7257 |     0.7895 | **0.7562** |     0.8162 |

### Result interpretation

The two models achieved very similar performance.

**Logistic Regression** achieved:

* Highest Accuracy: **74.50%**
* Highest Precision: **0.7380**
* Highest ROC-AUC: **0.8174**

**Random Forest** achieved:

* Slightly higher Recall: **0.7895**
* Slightly higher F1 Score: **0.7562**

Therefore, the choice of the final model depends on the project's objective.

For a more interpretable healthcare analytics project, **Logistic Regression is a strong choice** because its coefficients can be examined to understand the influence of individual features.

---

# 🔬 Feature Importance Analysis

Feature importance analysis was performed to understand which health indicators contribute most strongly to model predictions.

### Random Forest

Random Forest provides feature-importance scores based on how useful each feature is for splitting the decision trees.

### Logistic Regression

For Logistic Regression, the model coefficients were analyzed.

The absolute magnitude of a coefficient indicates how strongly a feature contributes to the model.

This allows the project to answer questions such as:

> Which health indicators are most influential when predicting diabetes status?

A feature-importance visualization is generated as part of the project.

---

# 📊 Model Evaluation Visualizations

The project generates several visualizations:

### Confusion Matrix

Shows:

* True Positives
* True Negatives
* False Positives
* False Negatives

### ROC Curve

The ROC curve evaluates how effectively the model separates diabetes and non-diabetes cases.

The Logistic Regression model achieved:

```text
ROC-AUC = 0.8174
```

which indicates good discrimination between the two classes in this dataset.

### Feature Importance Graph

Displays the most influential health indicators used by the models.

---

# 💾 Saved Model Files

The trained model and supporting files are saved using Joblib.

```text
healthcare_diabetes_model.pkl
healthcare_feature_columns.pkl
healthcare_scaler.pkl
```

Additional evaluation files:

```text
model_performance.csv
random_forest_feature_importance.csv
logistic_regression_feature_importance.csv
```

These files can be used later to build a web-based prediction application.

---

# 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Joblib**
* **Google Colab**
* **Kaggle**

---

# 📁 Project Structure

```text
Healthcare-Predictive-Analytics/
│
├── dataset/
│   └── diabetes_binary_5050split_health_indicators_BRFSS2015.csv
│
├── notebooks/
│   └── diabetes_prediction.ipynb
│
├── models/
│   ├── healthcare_diabetes_model.pkl
│   ├── healthcare_feature_columns.pkl
│   └── healthcare_scaler.pkl
│
├── results/
│   ├── model_performance.csv
│   ├── random_forest_feature_importance.csv
│   └── logistic_regression_feature_importance.csv
│
├── visualizations/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
│
├── README.md
└── requirements.txt
```

---

# 🚀 How to Run

### 1. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

### 2. Open the Google Colab notebook

Upload the Kaggle dataset when prompted.

### 3. Run the preprocessing steps

The notebook will:

* Load the dataset
* Check data quality
* Remove duplicates
* Select 50,000 records
* Split the data
* Normalize features

### 4. Train the models

The notebook trains:

```text
Logistic Regression
Random Forest
```

### 5. Evaluate the models

The notebook calculates:

```text
Accuracy
Precision
Recall
F1 Score
ROC-AUC
```

### 6. Analyze feature importance

The most influential health indicators are visualized.

### 7. Save the trained model

The model can then be used for deployment.

---

# 🔐 Ethical Data Handling & Patient Privacy

Healthcare machine learning requires responsible handling of data.

This project follows these principles:

* Use a publicly available dataset for educational purposes.
* Do not collect personally identifiable information.
* Do not store patient names, addresses, phone numbers, or medical IDs.
* Avoid exposing individual records in a deployed application.
* Use anonymized/aggregated healthcare data where possible.
* Clearly communicate model limitations.
* Do not treat machine learning predictions as confirmed medical diagnoses.

### Privacy

The model should not be used to collect unnecessary personal information.

If the system is deployed, only the health indicators required for prediction should be collected, and they should be handled securely.

---

# ⚠️ Limitations

The model has several limitations:

1. It is trained on a specific dataset and may not generalize to every population.
2. The dataset represents survey-based health indicators rather than complete clinical records.
3. A prediction does not confirm that a person has diabetes.
4. Model performance may change when applied to different populations.
5. Important medical factors not included in the dataset cannot be considered by the model.
6. The system should not replace professional medical evaluation.

---

# 🔮 Future Improvements

Future versions could include:

### 1. Web Application

Develop a Streamlit interface where users enter health indicators and receive a prediction.

### 2. Explainable AI

Provide explanations for individual predictions using techniques such as SHAP.

### 3. Hyperparameter Optimization

Use Grid Search or Randomized Search to optimize the models.

### 4. Additional Models

Experiment with:

* XGBoost
* Gradient Boosting
* Support Vector Machine
* K-Nearest Neighbors

### 5. Better Risk Communication

Instead of simply displaying:

```text
Diabetes / No Diabetes
```

the application could provide a probability estimate with a clear disclaimer.

### 6. External Validation

Test the final model on an independent healthcare dataset to determine how well it generalizes.

---

# 🎓 Learning Outcomes

This project demonstrates practical knowledge of:

* Healthcare data analysis
* Data preprocessing
* Data normalization
* Binary classification
* Logistic Regression
* Random Forest
* Train/Test splitting
* Stratified sampling
* Model evaluation
* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion matrices
* ROC curves
* Feature importance
* Model serialization
* Healthcare data privacy
* Ethical AI

---

# 👩‍💻 Author

**Sneha Singh**

### Project

**Healthcare Predictive Analytics — Diabetes Risk Prediction Using Machine Learning**

---

## ⭐ Project Summary

> This project develops a machine learning-based healthcare predictive analytics system for diabetes status prediction using health and lifestyle indicators. A 50,000-record subset of a Kaggle Diabetes Health Indicators dataset was processed and used to train Logistic Regression and Random Forest classification models. Logistic Regression achieved 74.50% accuracy and an ROC-AUC of 0.8174, while Random Forest achieved 74.13% accuracy. Feature importance analysis was performed to identify influential health indicators. The project also emphasizes ethical data handling, patient privacy, and the limitations of using machine learning for healthcare prediction.
