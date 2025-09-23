import joblib
import uuid
import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
import logging
from time import time


# Define a data model for the input text
class TextInput(BaseModel):
    text: str


# Configure logging to write to a file
log_file_path = "/var/log/app/app.log"
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(),  # Also prints to console for debugging
    ],
)
logger = logging.getLogger("sentiment-api")
logger.info("Logger configured successfully.")


# Initialize the FastAPI application
app = FastAPI(
    title="Sentiment Analysis API",
    description="A simple API to predict the sentiment of movie reviews.",
)

# Load the trained model and vectorizer
try:
    logging.info("Loading model and vectorizer...")
    model = joblib.load("models/sentiment_model.joblib")
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
    logging.info("Model and vectorizer loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model artifacts: {e}")
    # Raise an exception to prevent the app from starting if artifacts are missing
    raise RuntimeError("Model artifacts not found. Please run the training script.")


# Preprocessing function (needs to be the same as in the training script)
def preprocess_text(text: str) -> str:
    # A simple example, should match the one used in train_model.py
    return text.lower().strip()


# Define the prediction endpoint
@app.post("/predict_sentiment/")
def predict_sentiment(item: TextInput, request: Request):
    """
    Predicts the sentiment of a given text review with detailed logging.
    """
    request_id = str(uuid.uuid4())
    start_time = time()

    try:
        # Preprocessing function (needs to be the same as in the training script)
        # Assuming you've already defined preprocess_text()
        preprocessed_text = preprocess_text(item.text)

        # Vectorize the preprocessed text
        vectorized_text = vectorizer.transform([preprocessed_text])

        # Make the prediction
        prediction = model.predict(vectorized_text)
        sentiment = "positive" if prediction[0] == 1 else "negative"

        # Calculate latency
        latency = (time() - start_time) * 1000  # in milliseconds

        # Log the prediction details
        logger.info(
            f"request_id={request_id}, status=success, latency={latency:.2f}ms, "
            f"input_text='{item.text}', prediction={sentiment}"
        )

        return {"sentiment": sentiment}

    except Exception as e:
        latency = (time() - start_time) * 1000
        logger.error(
            f"request_id={request_id}, status=failure, latency={latency:.2f}ms, "
            f"error_message={str(e)}"
        )
        return {"error": "An error occurred during prediction."}


# A root endpoint for a quick health check
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Sentiment Analysis API is running."}
