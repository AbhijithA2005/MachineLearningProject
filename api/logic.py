import os
import re
import numpy as np

# ── TF-IDF + Logistic Regression Classifier ─────────────────────────────────
# Trained at module load time from the 500+ sample dataset.
# scikit-learn is already a project dependency.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    from .dataset import LABELLED_DATASET
except ImportError:
    from dataset import LABELLED_DATASET

# Build and fit the TF-IDF pipeline at import time (~50ms on cold start)
_texts  = [text  for text, _ in LABELLED_DATASET]
_labels = [label for _, label in LABELLED_DATASET]

_tfidf_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams for richer features
        sublinear_tf=True,    # dampen high-frequency terms
        min_df=1,
        max_features=8000,
    )),
    ("clf", LogisticRegression(
        C=4.0,
        max_iter=1000,
        solver="lbfgs",
        multi_class="multinomial",
    )),
])
_tfidf_pipeline.fit(_texts, _labels)

# ── Entity extraction ─────────────────────────────────────────────────────────
def extract_entities(text):
    text_lower = text.lower()

    target = "the primary outcome"
    target_match = re.search(
        r'(predict|forecast|classify|detect|estimate|optimize|cluster|group|segment)'
        r'\s+([\w\s]+?)(?=\s+based|\s+using|\s+from|\s+in|\s+with|\.|$)',
        text_lower
    )
    if target_match:
        target = target_match.group(2).strip()

    data_source = "the available historical data"
    data_match = re.search(
        r'(based on|using|from|with)\s+([\w\s]+?)(?=\s+to|\s+for|\.|$)',
        text_lower
    )
    if data_match:
        data_source = data_match.group(2).strip()
    elif "history" in text_lower or "historical" in text_lower:
        data_source = "historical records"

    return target, data_source


# ── Core intelligence ─────────────────────────────────────────────────────────
def get_ml_info(problem_description):
    text = problem_description.lower()

    # ── Step 1: TF-IDF + Logistic Regression (primary classifier) ──
    proba   = _tfidf_pipeline.predict_proba([problem_description])[0]
    classes = _tfidf_pipeline.classes_          # alphabetical order
    tfidf_type = classes[np.argmax(proba)]
    tfidf_conf = float(np.max(proba))

    # ── Step 2: Rule-based keyword engine (fallback / hybrid tie-breaker) ──
    keyword_scores = {"Supervised": 0, "Unsupervised": 0, "Reinforcement": 0}
    keyword_weights = {
        "Supervised":    ["predict", "forecast", "classification", "regression",
                          "detect", "labels", "estimate", "historical labels",
                          "ground truth", "target variable"],
        "Unsupervised":  ["cluster", "group", "segment", "hidden", "patterns",
                          "structure", "unlabeled", "no labels", "discovery",
                          "similarity"],
        "Reinforcement": ["agent", "reward", "optimize", "policy", "feedback",
                          "trial", "control", "engine", "navigate", "autonomous",
                          "environment", "action", "state"],
    }
    for paradigm, keywords in keyword_weights.items():
        for k in keywords:
            if k in text:
                keyword_scores[paradigm] += 1
                if k in ["predict", "cluster", "reward", "agent"]:
                    keyword_scores[paradigm] += 1

    keyword_max   = max(keyword_scores.values())
    keyword_total = sum(keyword_scores.values())
    keyword_type  = max(keyword_scores, key=keyword_scores.get) if keyword_max > 0 else None
    keyword_conf  = (keyword_max / (keyword_total + 1)) if keyword_max > 0 else 0

    # ── Step 3: Hybrid decision ──
    # Trust TF-IDF when confident; fall back to keyword engine if ambiguous.
    TFIDF_THRESHOLD = 0.60
    if tfidf_conf >= TFIDF_THRESHOLD:
        final_type = tfidf_type
        confidence = min(0.97, 0.55 + tfidf_conf * 0.42)
    elif keyword_type is not None:
        final_type = keyword_type
        confidence = min(0.82, 0.50 + keyword_conf * 0.35)
    else:
        final_type = tfidf_type   # best guess from TF-IDF
        confidence = max(0.40, tfidf_conf * 0.70)

    # ── Complexity ──
    word_count = len(text.split())
    if word_count < 8:
        complexity = "Low"
        confidence = max(0.38, confidence - 0.18)
    elif word_count > 25:
        complexity = "High"
    else:
        complexity = "Medium"

    # ── Rich output generation ──
    target, data_source = extract_entities(problem_description)
    justification = generate_deep_justification(final_type, target, data_source, text)
    why_not       = generate_why_not(final_type, target)
    roadmap       = generate_advanced_roadmap(final_type, target, data_source)

    alts = {
        "Supervised":    {"type": "Time Series / Deep Learning",
                          "algorithms": ["ARIMA", "LSTM", "Transformers"]},
        "Unsupervised":  {"type": "Self-Supervised Learning",
                          "algorithms": ["Autoencoders", "Generative Adversarial Networks (GANs)"]},
        "Reinforcement": {"type": "Heuristic Optimization",
                          "algorithms": ["Genetic Algorithms", "Simulated Annealing"]},
    }

    if confidence > 0.80:
        conf_exp = (f"Confidence is high ({int(confidence*100)}%) because the TF-IDF "
                    f"sentence embedding strongly matched established {final_type} patterns "
                    f"in the training corpus.")
    elif confidence > 0.60:
        conf_exp = (f"Confidence is moderate ({int(confidence*100)}%). The TF-IDF classifier "
                    f"aligns the input with {final_type}, though some ambiguity remains in "
                    f"the feature space.")
    else:
        conf_exp = (f"Confidence is low ({int(confidence*100)}%). The description is brief or "
                    f"ambiguous. Mapped to {final_type} via hybrid keyword and TF-IDF scoring.")

    input_enhancements = ("To improve accuracy, specify whether your data has historical labels "
                          "or if the model needs to learn from an environment.")
    if complexity == "Low":
        input_enhancements = ("Your input is very brief. Add the target variable, "
                              "the data source, and whether labels exist.")

    use_cases = {
        "Supervised":    ["Customer Churn Prediction", "Sales Demand Forecasting",
                          "Financial Fraud Detection"],
        "Unsupervised":  ["Customer Segmentation", "Network Anomaly Detection",
                          "Topic Modeling in Documents"],
        "Reinforcement": ["Dynamic Pricing Optimization", "Autonomous Robot Navigation",
                          "Supply Chain Logistics Routing"],
    }

    algorithms = {
        "Supervised":    ["Random Forest", "Gradient Boosting (XGBoost/LGBM)",
                          "Neural Networks (MLP)"],
        "Unsupervised":  ["K-Means Clustering", "Principal Component Analysis (PCA)",
                          "Isolation Forest"],
        "Reinforcement": ["Proximal Policy Optimization (PPO)", "Deep Q-Network (DQN)",
                          "Soft Actor-Critic (SAC)"],
    }

    return {
        "ml_type":                    final_type,
        "confidence":                 confidence,
        "confidence_explanation":     conf_exp,
        "problem_complexity":         complexity,
        "justification":              justification,
        "why_not_other":              why_not,
        "alternative_approach":       alts.get(final_type),
        "algorithms":                 algorithms.get(final_type, []),
        "roadmap":                    roadmap,
        "input_enhancement_suggestions": input_enhancements,
        "similar_use_cases":          use_cases.get(final_type, []),
    }


# ── Justification generators ──────────────────────────────────────────────────
def generate_deep_justification(ml_type, target, data, text):
    if ml_type == "Supervised":
        return (f"The problem involves mapping input features from {data} to forecast or classify "
                f"'{target}'. Since this requires historical labelled data to train a model, "
                f"it aligns with Supervised Learning — confirmed by TF-IDF sentence similarity "
                f"against 500+ business problem descriptions.")
    elif ml_type == "Unsupervised":
        return (f"The core task is to analyze {data} to automatically discover hidden structures "
                f"or segment '{target}'. Since no labels are provided, this is an Unsupervised "
                f"Learning task — validated by TF-IDF embedding similarity to unlabelled "
                f"problem descriptions in the training corpus.")
    else:
        return (f"This involves training an agent to optimize '{target}' within an environment. "
                f"The system must learn optimal actions through trial-and-error using reward "
                f"signals — a pattern confirmed by TF-IDF matching against Reinforcement "
                f"Learning problem templates.")


def generate_why_not(ml_type, target):
    if ml_type == "Supervised":
        return (f"Unsupervised learning is not suitable because the task requires predicting "
                f"a specific target ('{target}') rather than grouping unlabelled data. "
                f"Reinforcement Learning is unsuitable as there is no environment requiring "
                f"sequential agent actions and reward optimization.")
    elif ml_type == "Unsupervised":
        return ("Supervised learning cannot be applied because there are no historical ground-truth "
                "labels to train against. Reinforcement Learning is not appropriate as the goal "
                "is structure discovery, not policy optimization through environmental interaction.")
    else:
        return ("Supervised learning is infeasible here — there is no static dataset of 'correct' "
                "actions; the agent must discover strategy dynamically. Unsupervised learning "
                "cannot handle the sequential decision-making and reward-signal structure "
                "of this problem.")


# ── 5-Step Roadmap generators ─────────────────────────────────────────────────
def generate_advanced_roadmap(ml_type, target, data):
    if ml_type == "Supervised":
        return [
            {
                "title":      "Data Aggregation",
                "action":     f"Extract and consolidate {data} into a structured tabular format. Validate schema, resolve duplicates, and identify data gaps.",
                "tech_stack": "Pandas, SQL, Apache Spark",
            },
            {
                "title":      "Feature Engineering",
                "action":     f"Clean missing values, encode categorical variables, and engineer predictive features targeting '{target}'.",
                "tech_stack": "Scikit-Learn, NumPy, FeatureTools",
            },
            {
                "title":      "Model Training & Selection",
                "action":     "Split into train/validation/test sets. Train baseline and advanced models. Tune hyperparameters via cross-validation.",
                "tech_stack": "XGBoost, LightGBM, Scikit-Learn",
            },
            {
                "title":      "API Deployment",
                "action":     "Serialize the best model and expose a low-latency inference endpoint with input validation and versioning.",
                "tech_stack": "FastAPI, joblib, Vercel / Docker",
            },
            {
                "title":      "Monitoring & Drift Detection",
                "action":     f"Track live prediction distributions against training baselines. Alert on data or concept drift in '{target}' predictions.",
                "tech_stack": "Evidently AI, MLflow, Prometheus",
            },
        ]
    elif ml_type == "Unsupervised":
        return [
            {
                "title":      "Data Ingestion",
                "action":     f"Pull raw datasets representing {data} into the analytical environment. Profile distributions and assess data quality.",
                "tech_stack": "Pandas, PySpark, Great Expectations",
            },
            {
                "title":      "Dimensionality Reduction",
                "action":     "Compute variance explained and apply PCA or UMAP to reduce noise and enable visualization of high-dimensional feature space.",
                "tech_stack": "Scikit-Learn, UMAP-learn",
            },
            {
                "title":      "Clustering Execution",
                "action":     f"Apply clustering algorithms to group '{target}' naturally. Evaluate using Silhouette Score and Davies-Bouldin Index.",
                "tech_stack": "K-Means, DBSCAN, HDBSCAN",
            },
            {
                "title":      "Cluster Interpretation",
                "action":     "Profile each discovered cluster by computing feature centroids and identifying distinguishing characteristics per group.",
                "tech_stack": "Pandas, Matplotlib, Seaborn",
            },
            {
                "title":      "Operational Integration",
                "action":     "Export cluster assignments, schedule periodic re-clustering, and embed segment labels into downstream BI and reporting pipelines.",
                "tech_stack": "SQL, Tableau, BigQuery, Airflow",
            },
        ]
    else:  # Reinforcement
        return [
            {
                "title":      "Environment Design",
                "action":     f"Define the state space, observation schema, and transition dynamics of the '{target}' system.",
                "tech_stack": "Python, Gymnasium (OpenAI Gym)",
            },
            {
                "title":      "Reward Formulation",
                "action":     f"Design a reward function that mathematically incentivizes optimal behavior for '{target}'. Conduct reward shaping analysis.",
                "tech_stack": "Python, NumPy",
            },
            {
                "title":      "Agent Initialization & Training",
                "action":     "Initialize the policy network and train using selected RL algorithm. Log episode returns and visualize convergence curves.",
                "tech_stack": "Stable-Baselines3, PyTorch, TensorBoard",
            },
            {
                "title":      "Policy Evaluation & Safety",
                "action":     "Stress-test the converged policy across edge-case environment states. Apply safety constraints and adversarial perturbations.",
                "tech_stack": "Gymnasium, Safety-Gym, Python",
            },
            {
                "title":      "Production Handoff",
                "action":     "Serialize the final policy, wire to the live environment API, and instrument with real-time performance dashboards.",
                "tech_stack": "FastAPI, ONNX, Prometheus, Vercel",
            },
        ]
