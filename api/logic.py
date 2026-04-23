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
    keyword_scores = {"Supervised": 0, "Unsupervised": 0, "Reinforcement": 0, "Semi-supervised": 0, "Self-supervised": 0}
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
        "Semi-supervised": ["semi-supervised", "partially labelled", "pseudo-label",
                            "small labelled", "large unlabelled", "few annotated",
                            "unannotated", "sparse ground-truth"],
        "Self-supervised": ["self-supervised", "pre-train", "mask", "reconstruct",
                            "predict the next word", "auto-regressive", "contrastive",
                            "foundation model"],
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
        "Unsupervised":  {"type": "Dimensionality Reduction / Density Estimation",
                          "algorithms": ["PCA", "t-SNE", "Gaussian Mixture Models"]},
        "Reinforcement": {"type": "Heuristic Optimization",
                          "algorithms": ["Genetic Algorithms", "Simulated Annealing"]},
        "Semi-supervised": {"type": "Active Learning",
                            "algorithms": ["Uncertainty Sampling", "Query by Committee"]},
        "Self-supervised": {"type": "Foundation Models",
                            "algorithms": ["Transformers", "Masked Autoencoders", "Contrastive Learning"]},
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
        "Semi-supervised": ["Web Page Classification", "Medical Image Analysis",
                            "Speech Recognition Enhancement"],
        "Self-supervised": ["Large Language Models", "Vision Foundation Models",
                            "General Purpose Embeddings"],
    }

    algorithms = {
        "Supervised":    ["Random Forest", "Gradient Boosting (XGBoost/LGBM)",
                          "Neural Networks (MLP)"],
        "Unsupervised":  ["K-Means Clustering", "Principal Component Analysis (PCA)",
                          "Isolation Forest"],
        "Reinforcement": ["Proximal Policy Optimization (PPO)", "Deep Q-Network (DQN)",
                          "Soft Actor-Critic (SAC)"],
        "Semi-supervised": ["Pseudo-Labelling", "Label Spreading / Label Propagation",
                            "Generative Models (e.g. Semi-supervised VAE)"],
        "Self-supervised": ["BERT / GPT architectures", "SimCLR / MoCo",
                            "Masked Autoencoders (MAE)"],
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
    elif ml_type == "Reinforcement":
        return (f"This involves training an agent to optimize '{target}' within an environment. "
                f"The system must learn optimal actions through trial-and-error using reward "
                f"signals — a pattern confirmed by TF-IDF matching against Reinforcement "
                f"Learning problem templates.")
    elif ml_type == "Self-supervised":
        return (f"The task involves learning representations from massive {data} by generating "
                f"labels directly from the data itself (e.g. masking and reconstructing '{target}'). "
                f"This matches the Self-Supervised Learning paradigm for building foundation models.")
    else:
        return (f"The task involves classifying or predicting '{target}' using {data}, but "
                f"leverages a massive amount of unlabelled data alongside a small set of "
                f"labelled examples due to high labelling costs. This aligns with Semi-supervised "
                f"Learning, validated by our classifier.")


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
    elif ml_type == "Reinforcement":
        return ("Supervised learning is infeasible here — there is no static dataset of 'correct' "
                "actions; the agent must discover strategy dynamically. Unsupervised learning "
                "cannot handle the sequential decision-making and reward-signal structure "
                "of this problem.")
    elif ml_type == "Self-supervised":
        return ("Supervised learning is not applicable since manual labelling of such massive data "
                "is impossible. Unsupervised learning is insufficient because the goal is to learn "
                "rich feature representations by predicting hidden targets, not just clustering.")
    else:
        return ("Pure supervised learning is not scalable here because acquiring enough "
                "high-quality labels would be prohibitively expensive or time-consuming. "
                "Pure unsupervised learning would waste the valuable signal present in the "
                "small amount of existing labelled data.")


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
    elif ml_type == "Reinforcement":
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
    elif ml_type == "Self-supervised":
        return [
            {
                "title":      "Data Curation at Scale",
                "action":     f"Ingest and preprocess massive unlabelled {data}. Clean artifacts and ensure diversity for foundation model training.",
                "tech_stack": "Apache Spark, Ray, HuggingFace Datasets",
            },
            {
                "title":      "Pretext Task Design",
                "action":     f"Define the self-supervised objective for '{target}' (e.g. masked language modeling, contrastive instance discrimination).",
                "tech_stack": "PyTorch, TensorFlow",
            },
            {
                "title":      "Distributed Pre-training",
                "action":     "Train the large neural network using the pretext task across multiple GPUs. Monitor representation collapse and loss scaling.",
                "tech_stack": "PyTorch Lightning, DeepSpeed, Megatron-LM",
            },
            {
                "title":      "Representation Evaluation",
                "action":     "Evaluate the learned embeddings using linear probing or zero-shot transfer on established downstream benchmarks.",
                "tech_stack": "Scikit-Learn, Weights & Biases",
            },
            {
                "title":      "Fine-Tuning & Deployment",
                "action":     "Fine-tune the pre-trained model on specific downstream tasks and deploy the foundation model via a scalable inference endpoint.",
                "tech_stack": "HuggingFace Transformers, TensorRT, vLLM",
            },
        ]
    else:
        return [
            {
                "title":      "Data Segmentation",
                "action":     f"Partition {data} into a small labelled set and a massive unlabelled set. Ensure the labelled set is representative of the whole.",
                "tech_stack": "Pandas, Scikit-Learn",
            },
            {
                "title":      "Baseline Supervised Model",
                "action":     f"Train a baseline model using only the limited labelled data targeting '{target}'. Establish a lower bound for performance.",
                "tech_stack": "Scikit-Learn, LightGBM",
            },
            {
                "title":      "Pseudo-Labelling / Self-Training",
                "action":     "Use the baseline model to predict labels for the unlabelled data. Filter high-confidence predictions and add them to the training pool.",
                "tech_stack": "Python, NumPy",
            },
            {
                "title":      "Iterative Retraining",
                "action":     "Retrain the model on the combined dataset (true labels + pseudo-labels). Repeat the process until performance plateaus.",
                "tech_stack": "Scikit-Learn, PyTorch",
            },
            {
                "title":      "Validation & Deployment",
                "action":     f"Evaluate the final model strictly on a hold-out set with true labels for '{target}'. Deploy to a scalable inference endpoint.",
                "tech_stack": "FastAPI, MLflow, Docker",
            },
        ]
