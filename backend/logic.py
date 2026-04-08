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

def get_ml_info(problem_description):
    # Rule-based detection
    text = problem_description.lower()
    rules_paradigm = None
    keywords_found = []
    
    # Keyword mapping
    supervised_keywords = ["predict", "forecast", "classification", "regression", "detect", "labels", "estimate"]
    unsupervised_keywords = ["cluster", "group", "segment", "hidden", "patterns", "structure", "unlabeled"]
    reinforcement_keywords = ["agent", "reward", "optimize", "policy", "feedback", "trial", "control", "engine", "navigate"]
    
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

    # ML Model Prediction
    embedding = embedding_model.encode([problem_description])[0]
    probs = classifier.predict_proba([embedding])[0]
    classes = classifier.classes_
    prediction_idx = np.argmax(probs)
    ml_type_model = classes[prediction_idx]
    confidence = float(np.max(probs))

    # Hybrid decision
    final_type = ml_type_model
    if rules_paradigm:
        # If rules and model agree or if confidence is low, favor rules slightly or combine
        if confidence < 0.6:
            final_type = rules_paradigm
    
    # Confidence adjustment if rules match
    if rules_paradigm == ml_type_model:
        confidence = min(1.0, confidence + 0.05)

    # Algorithms
    algorithms = {
        "Supervised": ["Random Forest", "Gradient Boosting (XGBoost/LGBM)", "Support Vector Machines (SVM)", "Logistic/Linear Regression", "Neural Networks"],
        "Unsupervised": ["K-Means Clustering", "Principal Component Analysis (PCA)", "Isolation Forest (for anomaly detection)", "DBSCAN", "Hierarchical Clustering"],
        "Reinforcement": ["Deep Q-Network (DQN)", "Proximal Policy Optimization (PPO)", "A3C", "Q-Learning", "SARSA"]
    }

    # Custom Roadmap
    roadmap = generate_roadmap(problem_description, final_type, keywords_found)

    return {
        "ml_type": final_type,
        "confidence": confidence,
        "justification": generate_justification(problem_description, final_type, keywords_found),
        "algorithms": algorithms.get(final_type, []),
        "roadmap": roadmap
    }

def generate_justification(description, ml_type, keywords):
    reasons = []
    if keywords:
        reasons.append(f"The problem contains keywords like '{', '.join(keywords)}' which are indicative of {ml_type.lower()} learning tasks.")
    
    if ml_type == "Supervised":
        reasons.append("It involves mapping input variables to a known output target or label.")
    elif ml_type == "Unsupervised":
        reasons.append("The objective is to discover inherent groupings or structures within the data without pre-defined labels.")
    else:
        reasons.append("The task requires an agent to learn optimal behavior through trial-and-error and reward-based feedback loops.")
    
    return " ".join(reasons)

def generate_roadmap(description, ml_type, keywords):
    # Context-aware roadmap steps
    steps = []
    
    if ml_type == "Supervised":
        steps = [
            f"Define target variable based on '{description}' and identify historical labeled data sources.",
            f"Preprocess inputs (e.g., {', '.join(keywords[:1]) if keywords else 'features'}) and handle missing values in training set.",
            f"Select and train a benchmark model (e.g., Random Forest) to establish a performance baseline.",
            "Tune hyperparameters using cross-validation to optimize accuracy for the specific problem domain.",
            "Deploy the model as an API and monitor for feature drift against real-world data."
        ]
    elif ml_type == "Unsupervised":
        steps = [
            f"Gather raw data related to '{description}' and perform exploratory data analysis (EDA).",
            "Apply normalization and dimensionality reduction (like PCA) to make patterns more discernible.",
            "Execute clustering algorithms to identify hidden segments or structures in the dataset.",
            "Validate results using Silhouette scores or domain-expert review to ensure segments are meaningful.",
            "Integrate insights into business workflows (e.g., personalized marketing or system optimization)."
        ]
    else:
        steps = [
            f"Define the environment and state space for the '{description}' agent.",
            "Formulate a reward function that incentivizes positive outcomes (e.g., success in navigation or optimization).",
            "Initialize a learning agent (e.g., PPO) and simulate interactions in a controlled environment.",
            "Iteration phase: Allow the agent to learn from feedback and refine its policy over thousands of episodes.",
            "Transition the trained policy to a production environment with safety guardrails."
        ]
        
    return steps
