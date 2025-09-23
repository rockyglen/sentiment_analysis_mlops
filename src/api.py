import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Define a data model for the input text
class TextInput(BaseModel):
    text: str


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
def predict_sentiment(item: TextInput):
    """
    Predicts the sentiment of a given text review.
    """
    try:
        # 1. Preprocess the input text
        preprocessed_text = preprocess_text(item.text)

        # 2. Vectorize the preprocessed text
        # Note: The vectorizer.transform() method expects a list of strings
        vectorized_text = vectorizer.transform([preprocessed_text])

        # 3. Make the prediction
        prediction = model.predict(vectorized_text)

        # 4. Map the numerical prediction to a human-readable label
        sentiment = "positive" if prediction[0] == 1 else "negative"

        logging.info(f"Prediction for '{item.text}': {sentiment}")

        return {"sentiment": sentiment}

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return {"error": "An error occurred during prediction."}


# A root endpoint for a quick health check
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Sentiment Analysis API is running."}
