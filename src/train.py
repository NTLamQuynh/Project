import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib


from data_generator import generate_dataset


def preprocess_data(df):
    X = df.drop("Revenue", axis=1)
    y = df["Revenue"]

    num_cols = X.select_dtypes(include=["float64", "int64"]).columns
    cat_cols = X.select_dtypes(include=["object", "bool"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    return X, y, preprocessor


def train_best_model():
    df = generate_dataset()
    X, y, preprocessor = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
    eval_metric="logloss",
    scale_pos_weight=4,   # ← Cực quan trọng
    learning_rate=0.05,
    n_estimators=300,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42
)


    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, "../models/best_model.pkl")

    print("Model saved to models/best_model.pkl")


if __name__ == "__main__":
    train_best_model()
