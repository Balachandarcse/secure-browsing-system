import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def main():
    print("Loading url_features.csv ...")

    df = pd.read_csv("/app/dataset/url_features.csv")
    df = df.dropna().reset_index(drop=True)

    y = df["label"].values
    X = df.drop(columns=["label"]).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training DEPLOYMENT-GRADE model...")

    # Pipeline = scaling + model (VERY IMPORTANT)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "/app/dataset/url_model.pkl")

    print("\n✅ Small, fast model saved to url_model.pkl")


if __name__ == "__main__":
    main()
