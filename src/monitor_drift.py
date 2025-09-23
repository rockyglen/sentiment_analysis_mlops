import pandas as pd
import json
import logging
from scipy.stats import ks_2samp
from dvc.api import get_url

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("simple-drift-monitor")


def monitor_drift():
    """
    Compares word count distribution in production vs. training data using a K-S test.
    """
    try:
        logger.info("Starting simplified data drift monitoring.")

        # --- 1. Load Reference (Training) Data ---
        training_data_url = get_url("data/IMDB_Dataset.csv")
        reference_data = pd.read_csv(training_data_url)
        reference_data["review_length"] = reference_data["review"].apply(
            lambda x: len(str(x).split())
        )

        # --- 2. Load Current (Production) Data from logs ---
        with open("logs/sample_predictions.log", "r") as f:
            log_entries = [json.loads(line) for line in f if line.strip()]

        current_data = pd.DataFrame(log_entries)
        current_data["review_length"] = current_data["input_text"].apply(
            lambda x: len(str(x).split())
        )

        # --- 3. Perform Kolmogorov-Smirnov Test ---
        # A low p-value suggests the distributions are different (drift is detected)
        ks_statistic, p_value = ks_2samp(
            reference_data["review_length"], current_data["review_length"]
        )

        drift_detected = p_value < 0.05

        # --- 4. Log the result ---
        if drift_detected:
            logger.warning(
                f"🚨 Data Drift Detected! "
                f"K-S statistic: {ks_statistic:.4f}, p-value: {p_value:.4f}. "
                f"The distribution of review lengths has changed significantly."
            )
        else:
            logger.info(
                f"✅ No significant data drift detected. "
                f"K-S statistic: {ks_statistic:.4f}, p-value: {p_value:.4f}."
            )

    except Exception as e:
        logger.error(f"An error occurred during drift monitoring: {e}")


if __name__ == "__main__":
    monitor_drift()
