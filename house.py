import streamlit as st
import numpy as np
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.util import ngrams


# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

# Initialize NLP tools
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load models
scaler = joblib.load("scaler.pkl")
lr = joblib.load("linear_model.pkl")
ridge = joblib.load("ridge_model.pkl")
lasso = joblib.load("lasso_model.pkl")
corpus = joblib.load("corpus.pkl")

# App title and mode selection
st.title("🏠 House Application by Karishma")
mode = st.radio("Choose Mode:", ["💬 Chatbot", "📈 Price Prediction"])

# 📈 Price Prediction Mode
if mode == "📈 Price Prediction":
    st.subheader("📊 California Housing Price Prediction")

    MedInc = st.number_input("Median income (in 10k$)", min_value=0.0)
    HouseAge = st.number_input("House Age", min_value=1.0)
    AveRooms = st.number_input("Average Rooms", min_value=1.0)
    AveBedrms = st.number_input("Average Bedrooms", min_value=1.0)
    Population = st.number_input("Population", min_value=1.0)
    AveOccup = st.number_input("Average Occupancy", min_value=1.0)
    Latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0)
    Longitude = st.number_input("Longitude", min_value=-124.0, max_value=-114.0)

    features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
    features = scaler.transform(features)

    if st.button("Predict House Value"):
        pred_linear = lr.predict(features)[0]
        pred_lasso = lasso.predict(features)[0]
        pred_ridge = ridge.predict(features)[0]

        st.success(f"Linear Regression Prediction: ${pred_linear * 100000:.2f}")
        st.success(f"Ridge Regression Prediction: ${pred_ridge * 100000:.2f}")
        st.success(f"Lasso Regression Prediction: ${pred_lasso * 100000:.2f}")

# 💬 Chatbot Mode
elif mode == "💬 Chatbot":
    st.subheader("💬 Chat with California Housing Assistant")
    user_input = st.text_input("Ask something about the models or dataset:")

    def preprocess(text):
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
        stems = [stemmer.stem(t) for t in tokens]
        lemmas = [lemmatizer.lemmatize(t) for t in tokens]
        pos_tags = nltk.pos_tag(tokens)
        return {
            "tokens": tokens,
            "stems": stems,
            "lemmas": lemmas,
            "pos": pos_tags,
        }

    def match_response(user_text):
        user_data = preprocess(user_text)
        user_set = set(user_data["lemmas"] + user_data["stems"])

        for question, answer in corpus:
            q_data = preprocess(question)
            q_set = set(q_data["lemmas"] + q_data["stems"])
            if len(user_set & q_set) >= 2:
                return answer

        for q, a in corpus:
            if q.lower() == "default":
                return a
        return "Sorry, I didn't get that."

    def show_nlp_features(text):
        data = preprocess(text)

        st.markdown("### Tokens")
        st.text(", ".join(data["tokens"]))

        st.markdown("### Stems")
        st.text(", ".join(data["stems"]))

        st.markdown("### Lemmas")
        st.text(", ".join(data["lemmas"]))

        st.markdown("### POS Tags")
        for word, tag in data["pos"]:
            st.write(f"{word}: {tag}")


    if st.button("Ask"):
        response = match_response(user_input)
        st.write(":", response)
        st.markdown("---")
        st.subheader(" NLP Feature Analysis")
        show_nlp_features(user_input)