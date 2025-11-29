from data_loader import load_data
from preprocess import preprocess_data
from evaluate import evaluate_model
# import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import pickle

def train_models():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "Random Forest": RandomForestClassifier(n_estimators=200),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            # colsample_bytree is not directly supported in sklearn GBC, max_features is similar
            max_features=0.9
        )
    }

    best_auc = 0
    best_model = None

    for name, model in models.items():
        print(f"\nTraining: {name}")
        model.fit(X_train, y_train)
        auc = evaluate_model(model, X_test, y_test)

        if auc > best_auc:
            best_auc = auc
            best_model = model

    with open("../models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print("\nBest model saved → models/best_model.pkl")

if __name__ == "__main__":
    train_models()
