"""Prediction-score monitoring entry point; raw feature drift is not collected."""

from src.mlops.drift_monitor import main

if __name__ == "__main__":
    main()
