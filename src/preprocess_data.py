import pandas as pd
import re
import string
import logging
from dvc.api import get_url

# Set up logging for better tracking
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def preprocess_text(text):
    """
    Cleans and preprocesses a single text string.
    """
    # 1. Lowercase all text
    text = text.lower()

    # 2. Remove HTML tags like '<br />'
    text = re.sub("<br />", "", text)

    # 3. Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 4. Remove extra whitespace
    text = text.strip()

    return text


def preprocess_and_save_data():
    """
    Loads raw data, applies preprocessing, and saves the cleaned data.
    """
    try:
        logging.info("Starting data preprocessing...")

        # Load the raw dataset from DVC
        data_url = get_url("data/IMDB_Dataset.csv")
        df = pd.read_csv(data_url)
        logging.info("Raw dataset loaded successfully.")

        # --- Apply Preprocessing to the 'review' column ---
        logging.info("Applying text preprocessing to the review column...")
        df["review"] = df["review"].apply(preprocess_text)

        # --- Save the Cleaned Data ---
        output_path = "data/processed_reviews.csv"
        df.to_csv(output_path, index=False)
        logging.info(f"Processed data saved to {output_path}")

    except Exception as e:
        logging.error(f"An error occurred during preprocessing: {e}")


if __name__ == "__main__":
    preprocess_and_save_data()
