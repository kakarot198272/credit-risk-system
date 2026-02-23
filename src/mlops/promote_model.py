import json
import os

REGISTRY_PATH = "models/registry.json"


def load_registry():
    if not os.path.exists(REGISTRY_PATH):
        print("Registry not found.")
        return None

    with open(REGISTRY_PATH, "r") as f:
        registry = json.load(f)

    return registry


def save_registry(registry):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=4)


def promote_model():
    registry = load_registry()

    if registry is None:
        return

    best_model = registry.get("best_profit_model")
    production_model = registry.get("production_model")

    print("\n===== MODEL PROMOTION =====")
    print(f"Best Profit Model: {best_model}")
    print(f"Current Production Model: {production_model}")

    if best_model is None:
        print("No best model available to promote.")
        return

    if best_model == production_model:
        print("Best model is already in production.")
        return

    confirm = input("\nPromote best model to production? (y/n): ")

    if confirm.lower() == "y":
        registry["production_model"] = best_model
        save_registry(registry)
        print(f"\nModel {best_model} promoted to production.")
    else:
        print("\nPromotion cancelled.")


if __name__ == "__main__":
    promote_model()
