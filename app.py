import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# -------------------------------
# Title
# -------------------------------
st.title("📰 Fake News Detection App")

st.write("Paste a news article below and check if it's Fake or Real.")

# -------------------------------
# Load and Train Model (cached)
# -------------------------------
@st.cache_resource
def load_model():
    # Load datasets
    fake_df = pd.read_csv("Data/Fake.csv")
    real_df = pd.read_csv("Data/True.csv")

    # Add labels
    fake_df["label"] = "FAKE"
    real_df["label"] = "REAL"

    # Combine
    df = pd.concat([fake_df, real_df], ignore_index=True)

    # Shuffle data
    df = df.sample(frac=1).reset_index(drop=True)

    # Create numeric label
    df["fake"] = df["label"].apply(lambda x: 1 if x == "FAKE" else 0)

    X = df["text"]
    y = df["fake"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Vectorize
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_train_vectorized = vectorizer.fit_transform(X_train)

    # Train model
    model = LinearSVC()
    model.fit(X_train_vectorized, y_train)

    # Accuracy
    accuracy = model.score(vectorizer.transform(X_test), y_test)

    return model, vectorizer, accuracy

# Load model
model, vectorizer, accuracy = load_model()

# Show accuracy
st.write(f"Model Accuracy: {round(accuracy * 100, 2)}%")

# -------------------------------
# User Input
# -------------------------------
user_input = st.text_area("📝 Enter News Text:")

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        # Transform input
        input_vectorized = vectorizer.transform([user_input])

        # Predict
        prediction = model.predict(input_vectorized)

        # Output
        if prediction[0] == 1:
            st.error("🚨 Fake News")
        else:
            st.success("✅ Real News")