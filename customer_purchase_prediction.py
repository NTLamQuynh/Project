
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def load_dataset():
    """
    Tự sinh ra một dataset mô phỏng hành vi khách hàng online.
    Cột nhãn: Revenue (1 = có mua, 0 = không mua).
    Không cần internet, không cần đọc CSV.
    """
    np.random.seed(42)
    n = 5000  # số khách hàng

    # Các cột số (numeric)
    administrative = np.random.randint(0, 10, size=n) #xem hành chính
    administrative_duration = np.random.exponential(scale=60, size=n)  # phút
    informational = np.random.randint(0, 5, size=n)
    informational_duration = np.random.exponential(scale=30, size=n)
    product_related = np.random.randint(1, 50, size=n)
    product_related_duration = np.random.exponential(scale=300, size=n)
    bounce_rates = np.random.uniform(0, 0.2, size=n)
    exit_rates = np.random.uniform(0, 0.3, size=n)
    page_values = np.random.exponential(scale=20, size=n)
    special_day = np.random.choice([0.0, 0.2, 0.4, 0.6, 0.8], size=n)

    # Các cột phân loại (categorical)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month = np.random.choice(months, size=n)

    operating_systems = np.random.randint(1, 5, size=n)
    browser = np.random.randint(1, 6, size=n)
    region = np.random.randint(1, 10, size=n)
    traffic_type = np.random.randint(1, 20, size=n)

    visitor_types = ["New_Visitor", "Returning_Visitor", "Other"]
    visitor_type = np.random.choice(visitor_types, size=n)

    weekend = np.random.choice([True, False], size=n)

    # Tạo xác suất mua hàng (probability) dựa trên vài đặc trưng
    # logic: khách xem nhiều sản phẩm, page_values cao, returning_visitor, cuối tuần, special_day cao → dễ mua hơn
    base_prob = 0.1
    #xác suất khách sẽ mua hàng
    prob = (
        base_prob
        + 0.003 * product_related
        + 0.01 * (page_values / (1 + page_values))
        + 0.05 * (visitor_type == "Returning_Visitor").astype(float)
        + 0.03 * weekend.astype(float)
        + 0.04 * special_day
        - 0.2 * bounce_rates
        - 0.1 * exit_rates
    )

    # ép về [0, 0.95]
    prob = np.clip(prob, 0, 0.95)

    # Revenue ~ Bernoulli(prob)
    revenue = np.random.binomial(1, prob, size=n)

    data = {
        "Administrative": administrative,
        "Administrative_Duration": administrative_duration,
        "Informational": informational,
        "Informational_Duration": informational_duration,
        "ProductRelated": product_related,
        "ProductRelated_Duration": product_related_duration,
        "BounceRates": bounce_rates,
        "ExitRates": exit_rates,
        "PageValues": page_values,
        "SpecialDay": special_day,
        "Month": month,
        "OperatingSystems": operating_systems,
        "Browser": browser,
        "Region": region,
        "TrafficType": traffic_type,
        "VisitorType": visitor_type,
        "Weekend": weekend,
        "Revenue": revenue,
    }

    df = pd.DataFrame(data)
    return df


def explore_data(df):
    print("\n=== 5 dòng đầu ===")
    print(df.head())

    print("\n=== Thông tin ===")
    print(df.info())

    print("\n=== Thống kê mô tả (các cột số) ===")
    print(df.describe())

    print("\n=== Tỷ lệ mua hàng (Revenue) ===")
    print(df["Revenue"].value_counts())

    # Vẽ tỷ lệ class
    df["Revenue"].value_counts().plot(kind="bar")
    plt.title("Tỷ lệ khách mua / không mua")
    plt.xticks(rotation=0)
    plt.show()

    # Heatmap tương quan
    plt.figure(figsize=(12, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=False)
    plt.title("Ma trận tương quan (các cột số)")
    plt.show()


# =======================
# 3. CHUẨN BỊ DỮ LIỆU
# =======================
def prepare_data(df):
    """
    - Tách X (features) và y (label)
    - Xác định cột số, cột phân loại
    - Tạo preprocessor: scale cột số, one-hot cột phân loại
    """
    # y là cột Revenue, convert True/False -> 1/0
    y = df["Revenue"].astype(int)
    X = df.drop("Revenue", axis=1)

    # Chọn cột số và cột phân loại
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object", "bool"]).columns

    print("\nCột số:", list(numerical_cols))
    print("Cột phân loại:", list(categorical_cols))

    # Preprocessor: chuẩn hóa số + one-hot cho phân loại
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    return X, y, preprocessor


# =======================
# 4. TRAIN NHIỀU MODEL
# =======================
def train_models(preprocessor, X_train, y_train):
    """
    Tạo 3 model:
    - Logistic Regression
    - Random Forest
    - XGBoost
    Tất cả đều đi qua bước tiền xử lý (preprocessor).
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
        "XGBoost": XGBClassifier(
            use_label_encoder=False, 
            eval_metric="logloss", 
            random_state=42
        ),
    }

    trained = {}

    for name, model in models.items():
        # Pipeline: preprocessor -> model
        clf = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )

        print(f"\nĐang train model: {name} ...")
        clf.fit(X_train, y_train)
        print(f"Hoàn tất train model: {name}")
        trained[name] = clf

    return trained


# =======================
# 5. ĐÁNH GIÁ
# =======================
def evaluate(trained, X_test, y_test):
    """
    In accuracy, classification_report và confusion matrix
    cho từng model trong dict trained.
    """
    for name, model in trained.items():
        print("\n==========================")
        print(f"ĐÁNH GIÁ MODEL: {name}")
        print("==========================")

        y_pred = model.predict(X_test)

        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification report:")
        print(classification_report(y_test, y_pred))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title(f"Confusion Matrix – {name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()


# =======================
# 6. DỰ ĐOÁN 1 KHÁCH HÀNG MỚI
# =======================
def predict_single(model):
    """
    Dự đoán cho 1 khách hàng giả định (hard-code sẵn).
    Bạn có thể sửa các giá trị trong dict sample.
    """

    sample = {
        "Administrative": 3,
        "Administrative_Duration": 60,
        "Informational": 0,
        "Informational_Duration": 0,
        "ProductRelated": 20,
        "ProductRelated_Duration": 500,
        "BounceRates": 0.02,
        "ExitRates": 0.04,
        "PageValues": 30,
        "SpecialDay": 0.5,
        "Month": "Dec",
        "OperatingSystems": 3,
        "Browser": 2,
        "Region": 1,
        "TrafficType": 3,
        "VisitorType": "Returning_Visitor",
        "Weekend": True,
    }

    # Đưa vào DataFrame 1 hàng cho đúng format
    df_sample = pd.DataFrame([sample])

    pred = model.predict(df_sample)[0]
    print("\n=== DỰ ĐOÁN MẪU MỚI ===")
    print("Input:", sample)
    print("Kết quả dự đoán:", "CÓ MUA" if pred == 1 else "KHÔNG MUA")


# =======================
# 7. MAIN
# =======================
def main():
   
    df = load_dataset()
    explore_data(df)

    X, y, preprocessor = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    trained = train_models(preprocessor, X_train, y_train)

    evaluate(trained, X_test, y_test)

    # Chọn model tốt nhất (ví dụ XGBoost)
    best_model = trained["XGBoost"]

    # 👉 TẠO THƯ MỤC models NẾU CHƯA CÓ
    os.makedirs("models", exist_ok=True)

    # 👉 LƯU MODEL VÀO FILE best_model.pkl
    model_path = os.path.join("models", "best_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"Đã lưu model vào: {model_path}")

    # Dự đoán thử 1 khách hàng (như cũ)
    predict_single(best_model)


if __name__ == "__main__":
    main()
