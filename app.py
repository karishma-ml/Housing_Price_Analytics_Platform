import streamlit as st
import numpy as np
import pandas as pd
import joblib
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px 


porter = PorterStemmer()
lemm= WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


scaler = joblib.load("scaler.pkl")
lr = joblib.load("linear_model.pkl")
ridge = joblib.load("ridge_model.pkl")
lasso = joblib.load("lasso_model.pkl")
corpus = joblib.load("corpus.pkl")

# background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://image2url.com/images/1761145615829-44f2734e-9dc6-4699-a47b-da34be3cb897.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

st.title("🏠 House Price Predication ")
#st.write("Prediction using the most suitable models")
#load Dataset
try:
    df = pd.DataFrame(housing.data , columns = housing.feature_names)
    st.success("☑️Data loaded Successfully")
except FileNotFoundError:
    st.error("❌Data Not Found")
    st.stop()

# sidebar navigation using radio buttons    
section = st.sidebar.radio("Select Dataset Section", ["Dataset Preview", "Dataset Information", "Numerical Summary"])

if section == "Dataset Preview":
    view_option = st.sidebar.radio("Select view", ["Hide", "Show"])
    if view_option == "Show":
        st.sidebar.subheader("✨ Dataset Preview")
        st.sidebar.dataframe(df.head())
elif section == "Dataset Information":
    st.sidebar.subheader("📅 Dataset Information")
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Number of Rows", df.shape[0])
    col2.metric("Number of Columns", df.shape[1])
elif section == "Numerical Summary":
    with st.sidebar.expander("📊 Summary of Numerical Columns", expanded=False):
        st.write(df.describe())


#chatbot 
st.subheader("🤖 HousingBot")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Chatbot function ---
user_input = st.text_input("Ask something about the dataset")
def chatbot_response(user_input):
    user_input = user_input.strip().lower()
    for entry in corpus:
        if entry[0].lower() == user_input:
            return f"Intent matched: {entry[1]}"
    return "Sorry, I didn't understand that. Try a different keyword."


if user_input:
    response = chatbot_response(user_input)
    st.session_state.chat_history.append(("You: " + user_input , "Bot: " + response))

for user_msg, bot_msg in st.session_state.chat_history:
    st.write(user_msg)
    st.write(bot_msg)

# authentication
credentials = {
    "karishma" : "kari123",
    "pawan" : "paw123",
    "kanika" :"kan123"
    }

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.subheader("🔒Login Requried")
    username = st.text_input("Username")
    password = st.text_input("Password")
    login_button = st.button("Login")
  
    if login_button:
        if username in credentials and credentials[username]==password:
            st.success("😊 Your Welcome")
            st.session_state.logged_in = True
        else:
            st.error("🔒Invalid Username and Password")

if st.session_state.logged_in:   
    st.subheader("📊 Data Analysis Visual FAQs")
    with st.expander("Q1: What percentage of patients have heart disease"):
        st.write("MMost houses are between 15 and 30 years old, with a peak around 20 years.")    
        fig = px.histogram(df, x='HouseAge',nbins=20, title='HouseAge Distribution')
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Q2: How many data samples and features does the dataset have?"):
        st.write("The dataset contains 20,640 samples and 8 numeric features.")
        data_count= {"Features": 8, "Target": 1}
        fig2 = px.pie(names=data_count.keys(), values=data_count.values(), title="Feature vs Target Ratio")
        fig2.update_layout(paper_bgcolor= 'rgba(0,0,0,0)', plot_bgcolor= 'rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width= True)

    with st.expander("Q3: What is the average median house value in California?"):
        st.write("The average median house value is around $206,000.")
        avg_values = df[['MedInc', 'AveRooms', 'AveOccup']].mean().reset_index()
        avg_values.columns = ['Feature', 'Average']
        fig3 = px.bar(avg_values, x='Feature', y='Average', title='Average Values of Selected Features')
        fig3.update_layout(paper_bgcolor= 'rgba(0,0,0,0)', plot_bgcolor= 'rgba(0,0,0,0)')
        st.plotly_chart(fig3, use_container_width= True)

    with st.expander("Q4: How are house values distributed across California?"):
        st.write("Most houses have a median value between $100,000 and $250,000.")
        fig4= px.histogram(df, x="MedInc", nbins= 20, title= "Distribution of median House Values")
        fig4.update_layout(paper_bgcolor= 'rgba(0,0,0,0)', plot_bgcolor= 'rgba(0,0,0,0)')
        st.plotly_chart(fig4, use_container_width= True)


# 📈 Price Prediction Mode
if st.session_state.logged_in:
# 🏠 Price Prediction
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


# NLP Function (10 Lines)
def preprocess(text, ngram_n = 2):
    words = word_tokenize(text.lower())
    stemmed = [porter.stem(w) for w in words]
    lemmatized = [lemm.lemmatize(w) for w in words]
    no_stopwords = " ".join([f"'{w}'" for w in words if w not in stop_words])
    pos_tags = pos_tag(words)
    pos_str = " ".join([f"[{w:5}->{t}]" for w, t in pos_tags])
    n_grams = list(ngrams(words, ngram_n))
    ngrams_str = " ".join(["'"+" ".join(gram) + "'" for gram in n_grams])
    tfidf = TfidfVectorizer().fit_transform([' '.join(lemmatized)]).toarray()
    tfidf = tfidf[0]
    tfidf_formatted = ", ".join(f"{val:.8f}" for val in tfidf)
    return {
        "no_stopwords": " ".join(no_stopwords),
        "stemmed": stemmed,
        "lemmatized": lemmatized,
        "pos": pos_str,
        "ngrams": ngrams_str,
        "tfidf": tfidf_formatted
    }

#  sidebar NLP- Techneques  -------------------------------
st.sidebar.title("🔍 NLP Techniques")
options = st.sidebar.multiselect(
    "Select technique to display:",
    ["Stemmed", "Word Tokenize", "Lemmatized", "Stopword", "Ngrams", "POS Tags"],
)
show_nlp = st.sidebar.button("Show")


def process_text(text):
    words = word_tokenize(text.lower())
    results = {
        "Word Tokenize": " ".join(words),
        "Stemmed": " ".join([f"'{porter.stem(w)}'" for w in words]),
        "Lemmatized": " ".join([f"'{lemm.lemmatize(w)}'" for w in words]),
        "Stopword": " ".join([f"'{w}'" for w in words if w not in stop_words]),
        "Ngrams":  ", ".join([f"({gram[0]}, {gram[1]})" for gram in ngrams(words, 2)]),
        "POS Tags": " ".join([f"[{w:5}->{t}]" for w, t in pos_tag(words)])
    }
    return results

if show_nlp:
    if 'last_response' in st.session_state:
        response = st.session_state['last_response']
        result = process_text(response)

        if options:
            st.sidebar.subheader("🧠 NLP Output")
            for technique in options:
                st.sidebar.info(f"**{technique}:** {result[technique]}")
        else:
            st.sidebar.warning("Please select at least one NLP technique.")
    else:
        st.sidebar.error("❗ Please ask something in chatbot first.")   


