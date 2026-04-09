import os
import numpy as np

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
    
    # Lightweight weighted keyword classification
    # This replaces the heavy SentenceTransformers to fit on Vercel
    scores = {
        "Supervised": 0,
        "Unsupervised": 0,
        "Reinforcement": 0
    }
    
    weights = {
        "Supervised": ["predict", "forecast", "classification", "regression", "detect", "labels", "estimate", "historical labels", "ground truth", "target variable"],
        "Unsupervised": ["cluster", "group", "segment", "hidden", "patterns", "structure", "unlabeled", "no labels", "discovery", "similarity"],
        "Reinforcement": ["agent", "reward", "optimize", "policy", "feedback", "trial", "control", "engine", "navigate", "autonomous", "environment", "action", "state"]
    }
    
    # Calculate scores based on keyword occurrences
    for paradigm, keywords in weights.items():
        for k in keywords:
            if k in text:
                scores[paradigm] += 1
                # Give extra weight to strong indicators
                if k in ["predict", "cluster", "reward", "agent"]:
                    scores[paradigm] += 1

    # Determine final type
    max_score = max(scores.values())
    if max_score == 0:
        final_type = "Supervised" # Default fallback
        confidence = 0.45
    else:
        # Get the paradigm with the highest score
        final_type = max(scores, key=scores.get)
        # Calculate a pseudo-confidence score
        total_score = sum(scores.values())
        confidence = min(0.95, 0.6 + (max_score / (total_score + 1)) * 0.3)

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
    if confidence > 0.8:
        conf_exp = f"Confidence is high ({int(confidence*100)}%) because the problem description clearly matches established {final_type} patterns."
    elif confidence > 0.6:
        conf_exp = f"Confidence is moderate ({int(confidence*100)}%). While the task aligns with {final_type}, it lacks some explicit environment or target definitions."
    else:
        conf_exp = f"Confidence is low ({int(confidence*100)}%). The input is ambiguous. It has been mapped to {final_type} based on keyword frequency."

    input_enhancements = "To improve results, try specifying whether your data has historical labels or if the model needs to learn from an environment."
    if complexity == "Low":
        input_enhancements = "Your input is very brief. Please specify the target variable and the source of your data."

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
        return f"The problem involves mapping input features from {data} to forecast or classify '{target}'. Since this requires historical labeled data to train the model, it aligns with Supervised Learning."
    elif ml_type == "Unsupervised":
        return f"The core task is to analyze {data} to automatically discover hidden structures or segment '{target}'. Since no labels are provided, this is an Unsupervised Learning task."
    else:
        return f"This involves training an agent to optimize '{target}' within an environment. Since the system must learn optimal actions through trial-and-error using reward signals, it is a Reinforcement Learning problem."

def generate_why_not(ml_type, target):
    if ml_type == "Supervised":
        return f"Unsupervised learning is not suitable because the task requires predicting a specific target ('{target}') rather than simply grouping unlabeled data."
    elif ml_type == "Unsupervised":
        return "Supervised learning cannot be used because there are no clear, historical ground-truth labels to train against."
    else:
        return "Supervised learning isn't feasible here as we do not have a static dataset of 'correct' actions; the system must discover strategy dynamically."

def generate_advanced_roadmap(ml_type, target, data):
    if ml_type == "Supervised":
        return [
            {"title": "Data Aggregation", "action": f"Extract and consolidate {data} into a structured format.", "tech_stack": "Pandas, SQL"},
            {"title": "Feature Engineering", "action": f"Clean missing values and engineer features for '{target}'.", "tech_stack": "Scikit-Learn, NumPy"},
            {"title": "Model Training", "action": f"Split the dataset and train a baseline classification/regression model.", "tech_stack": "XGBoost, LightGBM"},
            {"title": "API Integration", "action": f"Deploy the model to a serverless endpoint.", "tech_stack": "FastAPI, Vercel"}
        ]
    elif ml_type == "Unsupervised":
        return [
            {"title": "Data Ingestion", "action": f"Pull raw datasets representing {data} into the analytical environment.", "tech_stack": "Pandas, PySpark"},
            {"title": "Dimensionality Reduction", "action": "Calculate variance and apply PCA to reduce noise.", "tech_stack": "Scikit-Learn"},
            {"title": "Clustering Execution", "action": f"Apply clustering algorithms to naturally group '{target}'.", "tech_stack": "K-Means, DBSCAN"},
            {"title": "Operational Integration", "action": "Export the cluster mappings to the data warehouse.", "tech_stack": "Vercel, SQL"}
        ]
    else:
        return [
            {"title": "Environment Design", "action": "Define the state space and transition dynamics of the system.", "tech_stack": "Python"},
            {"title": "Reward Formulation", "action": f"Design a reward function that mathematically incentivizes '{target}'.", "tech_stack": "Python"},
            {"title": "Agent Initialization", "action": "Initialize a neural network to act as the policy agent.", "tech_stack": "Keras, NumPy"},
            {"title": "Production Handoff", "action": "Deploy the converged policy with strict safety constraints.", "tech_stack": "Vercel"}
        ]
