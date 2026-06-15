
import pandas as pd
import numpy as np
import dice_ml
from dice_ml import Dice
import warnings
warnings.filterwarnings("ignore")


# ── Feature configuration ─────────────────────────────────────────────────────
# These must exactly match the raw input columns BEFORE feature engineering.

CONTINUOUS_FEATURES = [
    "age", "tenure_months", "monthly_fee", "total_revenue",
    "monthly_logins", "weekly_active_days", "avg_session_time",
    "support_tickets", "escalations", "avg_resolution_time", "csat_score",
    "payment_failures", "last_login_days_ago", "usage_growth_rate",
    "nps_score", "email_open_rate", "marketing_click_rate",
    "features_used", "referral_count"
]

CATEGORICAL_FEATURES = [
    "gender", "country", "customer_segment", "contract_type",
    "payment_method", "complaint_type",
    "signup_channel", "discount_applied", "price_increase_last_3m", "survey_response"
]

# DiCE will NEVER suggest changing these.
# You can't tell a customer to change their age, country, or how long
# they've been with you — so we lock them out of the search.
IMMUTABLE_FEATURES = [
    "age", "gender", "country", "tenure_months", "total_revenue", "signup_channel"
]

# DiCE is ONLY allowed to vary these — things a retention team can actually act on.
ACTIONABLE_FEATURES = [
    "monthly_fee", "monthly_logins", "weekly_active_days", "avg_session_time",
    "support_tickets", "escalations", "avg_resolution_time", "csat_score",
    "payment_failures", "last_login_days_ago", "usage_growth_rate",
    "nps_score", "email_open_rate", "marketing_click_rate",
    "features_used", "referral_count",
    "customer_segment", "contract_type", "payment_method",
    "discount_applied", "price_increase_last_3m", "survey_response"
]
# Without these, DiCE might suggest csat_score=50 or payment_failures=-3.
# These bounds keep all suggestions realistic.
FEATURE_RANGES = {
    "monthly_fee":            [10, 500],
    "monthly_logins":         [0, 100],
    "weekly_active_days":     [0, 7],
    "avg_session_time":       [1, 300],
    "support_tickets":        [0, 20],
    "escalations":            [0, 10],
    "avg_resolution_time":    [1, 72],
    "csat_score":             [1.0, 5.0],
    "payment_failures":       [0, 5],
    "last_login_days_ago":    [0, 80],
    "usage_growth_rate":      [-0.6, 0.6],
    "nps_score":              [-100, 100],
    "email_open_rate":        [0.1, 0.9],
    "marketing_click_rate":   [0.01, 0.5],
    "features_used":          [1, 15],
    "referral_count":         [0, 7],
}
class CustomThresholdWrapper:
    """Wraps the pipeline to force DiCE to use a custom business threshold instead of 0.5"""
    def __init__(self, pipeline, threshold=0.15):
        self.pipeline = pipeline
        self.threshold = threshold
        self.classes_ = pipeline.classes_
        
    def predict(self, X):
        probs = self.pipeline.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)
        
    def predict_proba(self, X):
        raw_probs = self.pipeline.predict_proba(X)
        p1 = raw_probs[:, 1]
        
        # Scale probabilities so our threshold (0.15) becomes exactly 0.50 to DiCE
        scaled_p1 = np.where(
            p1 < self.threshold,
            (p1 / self.threshold) * 0.5,                                    # Stretch 0 -> 0.15 up to 0 -> 0.50
            0.5 + ((p1 - self.threshold) / (1.0 - self.threshold)) * 0.5    # Compress 0.15 -> 1.0 into 0.50 -> 1.0
        )
        
        return np.vstack([1.0 - scaled_p1, scaled_p1]).T
    


class CounterfactualExplainer:
    """
    Wraps DiCE to generate minimum-intervention counterfactuals.

    Parameters
    ----------
    pipeline : trained sklearn Pipeline (feature_eng → preprocessor → XGBClassifier)
    X_train  : raw training DataFrame (pre-feature-engineering).
               DiCE learns feature distributions and valid categories from this.
    """

    def __init__(self, pipeline, X_train: pd.DataFrame):
        self.pipeline = pipeline
        self._build_dice(X_train)

    def _build_dice(self, X_train: pd.DataFrame):
        """
        Build the DiCE Data and Model objects. Called once at startup.

        dice_ml.Data  : tells DiCE which features are continuous vs categorical
                        and what the outcome column is. DiCE uses the dataframe
                        to learn valid value ranges and category sets.

        dice_ml.Model : wraps our pipeline so DiCE can call predict() on any
                        candidate counterfactual it wants to evaluate.

        method="random" : DiCE generates counterfactuals by random perturbation.
                          Fast and good enough for development. Swap to
                          method="genetic" for production — it uses an
                          evolutionary algorithm to find higher quality results.
        """
        raw_cols = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES
        X_raw = X_train[raw_cols].copy()
        X_raw["churn"] = 0  # DiCE Data requires an outcome column for its schema

        dice_data = dice_ml.Data(
            dataframe=X_raw,
            continuous_features=CONTINUOUS_FEATURES,
            outcome_name="churn"
        )

        # We pass `self` as the model because DiCE calls model.predict().
        # We implement predict() below to forward calls to our full pipeline.
# Wrap our model to use a custom threshold of 15%
        wrapped_model = CustomThresholdWrapper(self.pipeline, threshold=0.15)

        dice_model = dice_ml.Model(
            model=wrapped_model,       # Pass the wrapper instead of self.pipeline
            backend="sklearn",
            model_type="classifier"
        )
        self.dice_exp = Dice(dice_data, dice_model, method="genetic")



    # ── Public API ────────────────────────────────────────────────────────────

    def get_interventions(
        self,
        customer_df: pd.DataFrame,
        num_cfs: int = 3,
        desired_class: int = 0,
        proximity_weight: float = 1.5,
        diversity_weight: float = 1.0,
    ) -> list[dict]:
        """
        Generate counterfactual interventions for one high-risk customer.

        Parameters
        ----------
        customer_df      : one-row DataFrame with raw input features
        num_cfs          : how many different intervention sets to return
        desired_class    : 0 = "no churn" — always keep this as 0
        proximity_weight : higher = prefer smaller/fewer changes from original.
                           Raise to 1.5 if suggestions feel too aggressive.
        diversity_weight : higher = make the 3 options more different from each other.

        Returns
        -------
        [
          {
            "counterfactual_id": 1,
            "predicted_churn_proba": 0.08,
            "interventions": [
              {
                "feature":   "payment_failures",
                "original":  3,
                "suggested": 0,
                "delta":     "decrease payment failures from 3 → 0"
              },
              ...
            ]
          },
          ...
        ]
        """
        raw_cols = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES
        query = customer_df[raw_cols].copy()

        query.columns = query.columns.str.replace(' ', '_')
        for col in query.columns:
            if query[col].dtype == object:
                query[col] = query[col].str.strip().str.replace(' ', '_', regex=True)

        try:
            cf_result = self.dice_exp.generate_counterfactuals(
                query_instances=query,
                total_CFs=num_cfs,
                desired_class=desired_class,
                permitted_range=FEATURE_RANGES,           
                features_to_vary=ACTIONABLE_FEATURES,
                proximity_weight=proximity_weight,       
                diversity_weight=diversity_weight,
                verbose=True,
            )
        except Exception as e:
            return [{
                "counterfactual_id": 1,
                "predicted_churn_proba": None,
                "interventions": [],
                "error": str(e)
            }]

        return self._parse_cf_result(cf_result, query)
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(df)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(df)

    def _parse_cf_result(self, cf_result, original_query: pd.DataFrame) -> list[dict]:
        results = []
        cf_examples = cf_result.cf_examples_list[0]
        cf_df = cf_examples.final_cfs_df

        if cf_df is None or cf_df.empty:
            return [{
                "counterfactual_id": 1,
                "interventions": [],
                "error": "No counterfactuals found."
            }]

        raw_feature_cols = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES
        cf_df_clean = cf_df[[c for c in raw_feature_cols if c in cf_df.columns]].copy()

        original_row = original_query[raw_feature_cols].iloc[0]

        for idx, (_, cf_row) in enumerate(cf_df_clean.iterrows()):
            
            cf_input = cf_df_clean.iloc[[idx]].copy()
            try:
                cf_proba = self.pipeline.predict_proba(cf_input)[0][1]
            except Exception:
                cf_proba = 0.0  # Fallback
                
            interventions = []

            for feature in ACTIONABLE_FEATURES:
                if feature not in cf_row.index or feature not in original_row.index:
                    continue

                orig_val = original_row[feature]
                new_val  = cf_row[feature]

                # Check if the feature actually changed
                try:
                    if abs(float(orig_val) - float(new_val)) < 0.01:
                        continue
                except (TypeError, ValueError):
                    if str(orig_val).strip() == str(new_val).strip():
                        continue


                cf_input_with_one_reverted = cf_input.copy()
                cf_input_with_one_reverted[feature] = orig_val

                try:
                    reverted_proba = self.pipeline.predict_proba(cf_input_with_one_reverted)[0][1]
                    impact_delta = reverted_proba - cf_proba
                except Exception:
                    impact_delta = 0.0

                interventions.append({
                    "feature":   feature,
                    "original":  _clean_value(orig_val),
                    "suggested": _clean_value(new_val),
                    "delta":     _format_delta(feature, orig_val, new_val),
                    "impact":    round(float(impact_delta) * 100, 2)
                })

            # 4. ── SORT BY IMPACT ──
            interventions = sorted(interventions, key=lambda x: x["impact"], reverse=True)

            results.append({
                "counterfactual_id":     idx + 1,
                "predicted_churn_proba": round(float(cf_proba), 4) if cf_proba is not None else None,
                "interventions":         interventions,
            })

        return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_value(val):
    """Round floats for display; leave strings as-is."""
    try:
        f = float(val)
        return round(f, 2) if f != int(f) else int(f)
    except (TypeError, ValueError):
        return str(val)


def _format_delta(feature: str, orig, new) -> str:
    """Return a short plain-English description of the change."""
    labels = {
        "csat_score":             "CSAT score",
        "monthly_fee":            "monthly fee ($)",
        "monthly_logins":         "monthly logins",
        "weekly_active_days":     "weekly active days",
        "avg_session_time":       "avg session time (min)",
        "support_tickets":        "support tickets",
        "escalations":            "escalations",
        "avg_resolution_time":    "avg resolution time (hrs)",
        "payment_failures":       "payment failures",
        "last_login_days_ago":    "days since last login",
        "usage_growth_rate":      "usage growth rate",
        "nps_score":              "NPS score",
        "email_open_rate":        "email open rate",
        "marketing_click_rate":   "marketing click rate",
        "features_used":          "features used",
        "referral_count":         "referrals made",
        "contract_type":          "contract type",
        "customer_segment":       "customer segment",
        "payment_method":         "payment method",
        "discount_applied":       "discount applied",
        "survey_response":        "survey response",
        "complaint_type":         "complaint type",
        "price_increase_last_3m": "price increase in last 3 months",
    }
    label = labels.get(feature, feature)
    try:
        diff = float(new) - float(orig)
        direction = "increase" if diff > 0 else "decrease"
        return f"{direction} {label} from {_clean_value(orig)} → {_clean_value(new)}"
    except (TypeError, ValueError):
        return f"change {label} from '{orig}' → '{new}'"


# ── Convenience loader ────────────────────────────────────────────────────────

def load_explainer(model_path: str, train_data_path: str) -> CounterfactualExplainer:
    import joblib

    pipeline = joblib.load(model_path)
    X_train  = pd.read_csv(train_data_path)

    X_train.columns = X_train.columns.str.replace(' ', '_')
    for col in X_train.columns:
        if X_train[col].dtype == object:
            X_train[col] = X_train[col].str.strip().str.replace(' ', '_', regex=True)

    # Drop extra columns like customer_id, city — keep only what DiCE needs
    valid_cols = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES
    X_train = X_train[[c for c in valid_cols if c in X_train.columns]]

    return CounterfactualExplainer(pipeline, X_train)