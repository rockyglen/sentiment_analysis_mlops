import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dvc.api import get_url
import logging

# Set up logging for better tracking of script execution
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def perform_eda():
    """
    Performs Exploratory Data Analysis on the IMDB movie review dataset.
    This includes checking data structure, class distribution, and review length.
    """
    try:
        logging.info("Starting EDA process...")

        # Get the URL of the DVC-tracked dataset to ensure reproducibility
        data_url = get_url("data/IMDB_Dataset.csv")

        # Load the dataset into a pandas DataFrame
        df = pd.read_csv(data_url)
        logging.info("Dataset loaded successfully.")

        # --- Initial Data Exploration ---
        logging.info("Displaying initial data info...")
        print("First 5 rows of the dataset:")
        print(df.head())
        print("\nDataset information:")
        print(df.info())
        print("\nMissing values:")
        print(df.isnull().sum())

        # --- Analysis of Sentiment Distribution ---
        logging.info("Analyzing sentiment distribution...")
        plt.figure(figsize=(8, 6))
        sns.countplot(x="sentiment", data=df)
        plt.title("Distribution of Sentiment Labels", fontsize=16)
        plt.xlabel("Sentiment", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.savefig("eda_sentiment_distribution.png")
        logging.info(
            "Saved sentiment distribution plot to 'eda_sentiment_distribution.png'"
        )
        plt.show()

        # --- Analysis of Review Length ---
        logging.info("Analyzing review length distribution...")
        df["review_length"] = df["review"].apply(lambda x: len(x.split()))

        plt.figure(figsize=(12, 7))
        sns.histplot(df["review_length"], bins=50, kde=True, color="skyblue")
        plt.title("Distribution of Review Lengths", fontsize=16)
        plt.xlabel("Number of Words", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.savefig("eda_review_length_distribution.png")
        logging.info(
            "Saved review length distribution plot to 'eda_review_length_distribution.png'"
        )
        plt.show()

        # Display summary statistics for review length
        logging.info("Review length summary statistics:")
        print(df["review_length"].describe())

        logging.info("EDA process completed.")

    except Exception as e:
        logging.error(f"An error occurred during EDA: {e}")


if __name__ == "__main__":
    perform_eda()
