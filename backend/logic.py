import joblib
import os
from sentence_transformers import SentenceTransformer
import numpy as np

# Load models
base_dir = os.path.dirname(os.path.abspath(__file__))
classifier_path = os.path.join(base_dir, 'models', 'classifier.joblib')

classifier = None
if os.path.exists(classifier_path):
    try:
        classifier = joblib.load(classifier_path)
    except Exception as e:
        print(f"Error loading classifier: {e}")

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_entities(text):
    import re
    text_lower = text.lower()
    
    target = "the primary outcome"
    target_match = re.search(r'(predict|forecast|classify|detect|estimate|optimize|cluster|group|segment)\s+([\w\s]+?)(?=\s+based|\s+using|\s+from|\s+in|\s+with|\.|$)', text_lower)
    if target_match:
        target = target_match.group(2).strip()

    data_source = "the available historical data"
    data_match = re.search(r'(based on|using|from|with)\s+([\w\s]+?)(?=\s+to|\s+for|\.|$)', text_lower)
    if data_match:
        data_source = data_match.group(2).strip()
    elif "history" in text_lower or "historical" in text_lower:
        data_source = "historical records"
        
    return target, data_source

def get_ml_info(problem_description):
    text = problem_description.lower()
    rules_paradigm = None
    keywords_found = []
    
    supervised_keywords = ["predict", "forecast", "classification", "regression", "detect", "labels", "estimate"]
    unsupervised_keywords = ["cluster", "group", "segment", "hidden", "patterns", "structure", "unlabeled"]
    reinforcement_keywords = ["agent", "reward", "optimize", "policy", "feedback", "trial", "control", "engine", "navigate", "autonomous"]
    
    for k in supervised_keywords:
        if k in text:
            keywords_found.append(k)
            rules_paradigm = "Supervised"
    for k in unsupervised_keywords:
        if k in text:
            keywords_found.append(k)
            rules_paradigm = "Unsupervised"
    for k in reinforcement_keywords:
        if k in text:
            keywords_found.append(k)
            rules_paradigm = "Reinforcement"

    try:
        embedding = embedding_model.encode([problem_description])[0]
        probs = classifier.predict_proba([embedding])[0]
        classes = classifier.classes_
        prediction_idx = np.argmax(probs)
        ml_type_model = classes[prediction_idx]
        confidence = float(np.max(probs))
    except Exception:
        ml_type_model = rules_paradigm if rules_paradigm else "Supervised"
        confidence = 0.65

    final_type = ml_type_model
    if rules_paradigm and confidence < 0.6:
        final_type = rules_paradigm
    
    if rules_paradigm == ml_type_model:
        confidence = min(1.0, confidence + 0.15)
        
    # Analyze complexity
    word_count = len(text.split())
    if word_count < 8:
        complexity = "Low"
        confidence = max(0.4, confidence - 0.2)
    elif word_count > 25:
        complexity = "High"
    else:
        complexity = "Medium"

    # Deep Context Justification
    target, data_source = extract_entities(problem_description)
    justification = generate_deep_justification(final_type, target, data_source, text)
    why_not = generate_why_not(final_type, target)
    roadmap = generate_advanced_roadmap(final_type, target, data_source)
    
    # Alternatives
    alts = {
        "Supervised": {"type": "Time Series / Deep Learning", "algorithms": ["ARIMA", "LSTM", "Transformers"]},
        "Unsupervised": {"type": "Self-Supervised Learning", "algorithms": ["Autoencoders", "Generative Adversarial Networks (GANs)"]},
        "Reinforcement": {"type": "Heuristic Optimization", "algorithms": ["Genetic Algorithms", "Simulated Annealing"]}
    }
    
    # Confidence Explanation
    if confidence > 0.85:
        conf_exp = f"Confidence is high ({int(confidence*100)}%) because the problem description clearly matches established {final_type} patterns and contains specific operational keywords."
    elif confidence > 0.6:
        conf_exp = f"Confidence is moderate ({int(confidence*100)}%). While the task aligns with {final_type}, it lacks explicit data source definitions or target specifications."
    else:
        conf_exp = f"Confidence is low ({int(confidence*100)}%). The input is ambiguous. It has been mapped to {final_type} as a best effort, but more detail is required."

    input_enhancements = "To improve model selection, consider specifying the exact format of your data (e.g., CSV, images, text) and whether historical labels currently exist."
    if complexity == "Low":
        input_enhancements = "Your input is very brief. Please specify the target variable you wish to predict or optimize, and the exact data you have available."

    use_cases = {
        "Supervised": ["Customer Churn Prediction", "Sales Demand Forecasting", "Financial Fraud Detection"],
        "Unsupervised": ["Customer Segmentation", "Network Anomaly Detection", "Topic Modeling in Documents"],
        "Reinforcement": ["Dynamic Pricing Optimization", "Autonomous Robot Navigation", "Supply Chain Logistics Routing"]
    }

    algorithms = {
        "Supervised": ["Random Forest", "Gradient Boosting (XGBoost/LGBM)", "Neural Networks (MLP)"],
        "Unsupervised": ["K-Means Clustering", "Principal Component Analysis (PCA)", "Isolation Forest"],
        "Reinforcement": ["Proximal Policy Optimization (PPO)", "Deep Q-Network (DQN)", "Soft Actor-Critic (SAC)"]
    }

    return {
        "ml_type": final_type,
        "confidence": confidence,
        "confidence_explanation": conf_exp,
        "problem_complexity": complexity,
        "justification": justification,
        "why_not_other": why_not,
        "alternative_approach": alts.get(final_type),
        "algorithms": algorithms.get(final_type, []),
        "roadmap": roadmap,
        "input_enhancement_suggestions": input_enhancements,
        "similar_use_cases": use_cases.get(final_type, [])
    }

def generate_deep_justification(ml_type, target, data, text):
    if ml_type == "Supervised":
        return f"The problem involves mapping input features from {data} to forecast or classify '{target}'. Since this requires historical labeled data to train the model to output a known target, it aligns perfectly with Supervised Learning."
    elif ml_type == "Unsupervised":
        return f"The core task is to analyze {data} to automatically discover hidden structures or segment '{target}'. Because there are no pre-defined labeled outputs guiding the model, this is an Unsupervised Learning task."
    else:
        return f"This involves training an autonomous agent to optimize '{target}' within an interactive environment. Since the system must learn optimal actions through trial-and-error using reward signals rather than static data, it is a Reinforcement Learning problem."

def generate_why_not(ml_type, target):
    if ml_type == "Supervised":
        return f"Unsupervised learning is not suitable because the task requires predicting a specific target ('{target}') rather than simply grouping unlabeled data. Reinforcement learning is overkill as it doesn't require continuous environmental interaction."
    elif ml_type == "Unsupervised":
        return "Supervised learning cannot be used because there are no clear, historical ground-truth labels to train against. We must let the algorithm find the patterns inherently."
    else:
        return "Supervised learning isn't feasible here as we do not have a static dataset of 'correct' actions. The system must discover the optimal strategy dynamically through environmental interaction."

def generate_advanced_roadmap(ml_type, target, data):
    if ml_type == "Supervised":
        return [
            {"title": "Data Aggregation", "action": f"Extract and consolidate {data} into a structured format.", "tech_stack": "Pandas, SQL"},
            {"title": "Feature Engineering", "action": f"Clean missing values and engineer features highly correlated with '{target}'.", "tech_stack": "Scikit-Learn, NumPy"},
            {"title": "Model Training", "action": f"Split the dataset and train a baseline classification/regression model.", "tech_stack": "XGBoost, LightGBM"},
            {"title": "Hyperparameter Tuning", "action": "Optimize the model's parameters using cross-validation to minimize error.", "tech_stack": "Optuna, GridSearch"},
            {"title": "API Deployment", "action": f"Deploy the model to a REST endpoint to serve real-time '{target}' predictions.", "tech_stack": "FastAPI, Docker"}
        ]
    elif ml_type == "Unsupervised":
        return [
            {"title": "Data Ingestion", "action": f"Pull raw datasets representing {data} into the analytical environment.", "tech_stack": "Pandas, PySpark"},
            {"title": "Dimensionality Reduction", "action": "Scale the data and apply PCA to reduce noise and enhance pattern visibility.", "tech_stack": "Scikit-Learn"},
            {"title": "Clustering Execution", "action": f"Apply clustering algorithms to naturally group '{target}'.", "tech_stack": "K-Means, DBSCAN"},
            {"title": "Cluster Profiling", "action": "Analyze the distinct characteristics of each generated cluster for business relevance.", "tech_stack": "Matplotlib, Seaborn"},
            {"title": "Operational Integration", "action": "Export the cluster mappings back to the CRM or data warehouse.", "tech_stack": "Airflow, Snowflake"}
        ]
    else:
        return [
            {"title": "Environment Design", "action": "Define the state space, action space, and transition dynamics of the system.", "tech_stack": "OpenAI Gym, Ray RLlib"},
            {"title": "Reward Formulation", "action": f"Design a reward function that mathematically incentivizes '{target}'.", "tech_stack": "Python"},
            {"title": "Agent Initialization", "action": "Initialize a deep neural network to act as the policy agent.", "tech_stack": "PyTorch, TensorFlow"},
            {"title": "Simulation Training", "action": "Run thousands of simulated episodes allowing the agent to learn via exploration.", "tech_stack": "CUDA, AWS EC2"},
            {"title": "Production Handoff", "action": "Deploy the converged policy with strict safety constraints to prevent erratic real-world actions.", "tech_stack": "TorchServe"}
        ]
