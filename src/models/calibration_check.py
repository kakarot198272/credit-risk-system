import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

LOG_PATH = "artifacts/metrics/validation_predictions_logistic.csv"
LGB_PATH = "artifacts/metrics/validation_predictions_lightgbm.csv"


def plot_calibration(true_labels, pred_probs, label):
    prob_true, prob_pred = calibration_curve(
        true_labels,
        pred_probs,
        n_bins=10
    )

    plt.plot(prob_pred, prob_true, marker='o', label=label)


def main():
    log_df = pd.read_csv(LOG_PATH)
    lgb_df = pd.read_csv(LGB_PATH)

    plt.figure(figsize=(6, 6))

    # Perfect calibration line
    plt.plot([0, 1], [0, 1], linestyle="--")

    plot_calibration(log_df["true_label"], log_df["pred_prob"], "Logistic")
    plot_calibration(lgb_df["true_label"], lgb_df["pred_prob"], "LightGBM")

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Default Frequency")
    plt.title("Calibration Curve")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
