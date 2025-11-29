# 🚀 Credit Card Fraud Detection (Machine Learning Project)
Detecting fraudulent credit card transactions using Logistic Regression, Random Forest, and XGBoost — with SMOTE for class imbalance handling.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-GradientBoosting-green)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

# 📌 Project Overview
Credit card fraud is a global challenge for banks and financial systems.  
This project builds a **Machine Learning pipeline** to detect fraudulent transactions using:

- **Logistic Regression**
- **Random Forest**
- **XGBoost**

The dataset is extremely imbalanced, so **SMOTE (Synthetic Minority Oversampling Technique)** is used for resampling.  
The best model achieved:

### ⭐ **99% ROC-AUC**

---

# 🔥 Key Features
- End-to-end ML pipeline  
- Handles class imbalance using SMOTE  
- Trains 3 ML models  
- Compares model performance  
- Saves the best model (`best_model.pkl`)  
- Visualizes confusion matrix & ROC curve  
- GitHub-friendly modular folder structure  

---

# 📂 Project Structure
```
fraud-detection-ml/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── creditcard.csv      # (not included — download from Kaggle)
│
├── notebooks/
│   └── fraud_detection.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
└── models/
    └── best_model.pkl
```

---

# 📊 Dataset
Dataset: **Credit Card Fraud Detection**  
📥 Download from Kaggle:  
https://www.kaggle.com/mlg-ulb/creditcardfraud  

- 284,807 transactions  
- 492 frauds (0.17%)  
- PCA-transformed features (V1–V28)  
- Strong class imbalance  

---

# 🛠️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/fraud-detection-ml.git
cd fraud-detection-ml
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Add Dataset
Place the dataset file inside:

```
data/creditcard.csv
```

---

# 🚀 Training the Model

Run:

```bash
python src/train.py
```

This script will:

✔ Load the dataset  
✔ Preprocess data  
✔ Apply SMOTE  
✔ Train Logistic Regression, Random Forest, XGBoost  
✔ Evaluate each model  
✔ Save the best model → `models/best_model.pkl`

---

# 🧠 Machine Learning Models

| Model | Strength | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | Fast baseline | ~0.97 |
| Random Forest | Good interpretability | ~0.98 |
| XGBoost | Best performance | ~0.99 |

---

# 📈 Evaluation Metrics
The project measures:

- ROC-AUC  
- Confusion Matrix  
- Classification Report  
- Precision  
- Recall  
- F1-score  

These metrics are critical due to **high class imbalance**.

---

# 🖼️ Example Results

### Confusion Matrix (Example)
```
[[56864     2]
 [   17    81]]
```

### ROC-AUC (Best Model)
```
0.987 – 0.99
```

---

# 🧪 Prediction Example
```python
import pickle
import pandas as pd

# Load model
model = pickle.load(open("models/best_model.pkl", "rb"))

sample = pd.DataFrame([{
    "V1": -1.29, "V2": 0.87, ..., "Amount": 45.90
}])

prediction = model.predict(sample)
print("Fraud" if prediction[0] == 1 else "Not Fraud")
```

