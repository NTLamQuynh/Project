import numpy as np
import pandas as pd


def generate_dataset(n=5000, seed=42):
    np.random.seed(seed)

    # Numeric features
    administrative = np.random.randint(0, 10, size=n)
    administrative_duration = np.random.exponential(scale=60, size=n)
    informational = np.random.randint(0, 5, size=n)
    informational_duration = np.random.exponential(scale=30, size=n)
    product_related = np.random.randint(1, 50, size=n)
    product_related_duration = np.random.exponential(scale=300, size=n)
    bounce_rates = np.random.uniform(0, 0.2, size=n)
    exit_rates = np.random.uniform(0, 0.3, size=n)
    page_values = np.random.exponential(scale=20, size=n)
    special_day = np.random.choice([0.0, 0.2, 0.4, 0.6, 0.8], size=n)

    # Categorical
    months = ["Jan","Feb","Mar","Apr","May","June","Jul","Aug","Sep","Oct","Nov","Dec"]
    month = np.random.choice(months, size=n)

    operating_systems = np.random.randint(1, 5, size=n)
    browser = np.random.randint(1, 6, size=n)
    region = np.random.randint(1, 10, size=n)
    traffic_type = np.random.randint(1, 20, size=n)

    visitor_types = ["New_Visitor", "Returning_Visitor", "Other"]
    visitor_type = np.random.choice(visitor_types, size=n)
    weekend = np.random.choice([True, False], size=n)

    # Probability
    base_prob = 0.35
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
    prob = np.clip(prob, 0, 0.95)
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

    return pd.DataFrame(data)
