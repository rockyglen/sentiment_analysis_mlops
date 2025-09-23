import pandas as pd
from dvc.api import get_url
import logging
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Set up logging for better script tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_log_model():
    """
    Loads preprocessed data, trains a Logistic Regression model, and
    logs the experiment details to MLflow.
    """
    try:
        logging.info("Starting model training and logging process...")

        # --- Load Processed Data (from DVC) ---
        logging.info("Loading preprocessed data...")
        data_url = get_url('data/processed_reviews.csv')
        df = pd.read_csv(data_url)
        
        # --- Prepare Data for Modeling ---
        X = df['review']
        y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        logging.info(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")

        # --- Vectorize the Text Data ---
        logging.info("Vectorizing text data with TF-IDF...")
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # --- MLflow Experiment Tracking ---
        mlflow.set_experiment("Sentiment Analysis Baseline")

        with mlflow.start_run():
            # Define and log parameters
            max_iter = 1000
            mlflow.log_param("model_name", "LogisticRegression")
            mlflow.log_param("max_features", vectorizer.max_features)
            mlflow.log_param("max_iter", max_iter)
            
            # --- Train the Model ---
            logging.info("Training the Logistic Regression model...")
            model = LogisticRegression(max_iter=max_iter)
            model.fit(X_train_vec, y_train)

            # --- Evaluate the Model ---
            logging.info("Evaluating the model...")
            y_pred = model.predict(X_test_vec)
            
            # Generate and log metrics
            report = classification_report(y_test, y_pred, output_dict=True)
            accuracy = accuracy_score(y_test, y_pred)
            
            mlflow.log_metrics({
                "accuracy": accuracy,
                "f1_score_positive": report['1']['f1-score'],
                "f1_score_negative": report['0']['f1-score'],
                "precision_positive": report['1']['precision'],
                "recall_positive": report['1']['recall']
            })
            
            logging.info(f"Model trained successfully with accuracy: {accuracy:.4f}")

            # --- Log Model Artifacts ---
            logging.info("Logging model artifact to MLflow...")
            # Use a wrapper to save both the model and vectorizer
            artifacts_path = "model_artifacts"
            mlflow.sklearn.log_model(sk_model=model, artifact_path=artifacts_path)
            
            # Note: For real production, you'd also save the vectorizer separately
            # as it's needed for inference. We'll handle this in a later step.
            
        logging.info("Model training and logging complete.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == '__main__':
    train_and_log_model()