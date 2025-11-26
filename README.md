# Project
Dự án dự đoán *khách hàng có mua hàng hay không* dựa trên hành vi truy cập website.
Dữ liệu được mô phỏng theo dataset thực tế **Online Shoppers Purchasing Intention — UCI**.

Dự án gồm đầy đủ các bước của quy trình Machine Learning:

- Tạo dataset mô phỏng (synthetic dataset)
- Tiền xử lý dữ liệu (Scaling + One-Hot Encoding)
- Chia train/test theo chiến lược stratify
- Train 3 model (Logistic Regression, Random Forest, XGBoost)
- Fix imbalance bằng (sampling + scale_pos_weight)
- Đánh giá model (Accuracy, Precision/Recall/F1, AUC)
- Lưu model bằng joblib
- Build web app dự đoán bằng Streamlit


---

## 2. Dataset

Dataset được tự sinh bằng `data_generator.py` gồm 5000 dòng với các feature:

- Administrative, ProductRelated, Informational
- BounceRates, ExitRates, PageValues
- Month, VisitorType, Weekend
- OS, Browser, Region, TrafficType
- **Revenue (label)** = 1 nếu khách mua hàng

Dataset được sinh theo *logic giống thực tế*:

- Xem nhiều sản phẩm → tăng xác suất mua
- PageValues cao → tăng xác suất mua
- BounceRates cao → giảm xác suất mua
- Returning Visitor → tăng xác suất mua
- SpecialDay, Weekend → tăng xác suất mua

---

## 3. Preprocessing

Dùng `ColumnTransformer`:

- Numeric → StandardScaler  
- Categorical → OneHotEncoder  

Tất cả được gói trong **Pipeline**:
Pipeline(
preprocess → XGBoost
)
Giúp train/predict đồng nhất và dễ deploy.
## 4. Models

Train và so sánh 3 model:

- Logistic Regression  
- Random Forest  
- XGBoost (tuned: scale_pos_weight, n_estimators, learning_rate…)

Best Model: **XGBoost** (cao nhất AUC & Recall lớp 1)

---

## 5. Evaluation Metrics

- Accuracy  
- Precision / Recall / F1  
- Confusion Matrix  
- ROC Curve + AUC  

Mục tiêu chính: tăng Recall & Precision class 1 (người mua).

---

## 6. Saving Model

Model tốt nhất được lưu tại:
