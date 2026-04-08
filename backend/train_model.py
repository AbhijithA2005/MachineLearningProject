import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os

def train():
    # Load dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, '..', 'ml_problem_dataset_500.csv')
    df = pd.read_csv(csv_path)
    
    # Initialize model for embeddings
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = embedding_model.encode(df['problem_description'].tolist())
    
    # Prepare labels
    y = df['ml_type']
    
    # Train classifier
    print("Training classifier...")
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(embeddings, y)
    
    # Save models
    os.makedirs('models', exist_ok=True)
    joblib.dump(classifier, 'models/classifier.joblib')
    # We don't save the embedding_model because we can reload it from name
    
    print("Model trained and saved to models/classifier.joblib")

if __name__ == "__main__":
    train()
