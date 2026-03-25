from src.Pipeline.train_pipeline import run_train_pipeline


def main() -> None:
    score = run_train_pipeline("notebooks/stud.csv", "math_score")
    print(f"Training completed. R2 score: {score:.4f}")


if __name__ == "__main__":
    main()
