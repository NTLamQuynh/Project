import pandas as pd
import joblib


def predict_customer(input_dict):
    model = joblib.load("../models/best_model.pkl")
    df = pd.DataFrame([input_dict])
    pred = model.predict(df)[0]
    return "BUY" if pred == 1 else "NO BUY"


if __name__ == "__main__":
    sample = {
        "Administrative": 3,
        "Administrative_Duration": 40,
        "Informational": 1,
        "Informational_Duration": 10,
        "ProductRelated": 20,
        "ProductRelated_Duration": 200,
        "BounceRates": 0.02,
        "ExitRates": 0.03,
        "PageValues": 20,
        "SpecialDay": 0.4,
        "Month": "Dec",
        "OperatingSystems": 3,
        "Browser": 2,
        "Region": 2,
        "TrafficType": 3,
        "VisitorType": "Returning_Visitor",
        "Weekend": True,
    }

    print(predict_customer(sample))
