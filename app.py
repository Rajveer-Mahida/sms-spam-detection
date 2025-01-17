import streamlit as st
import pickle

# Load model and vectorizer with error handling
try:
    model = pickle.load(open('./src/models/BernoulliNB.pkl', 'rb'))
    cv = pickle.load(open('./src/models/vectorizer.pkl', 'rb'))
except FileNotFoundError as e:
    st.error("Error loading model or vectorizer. Please check the file paths.")
    st.stop()

# App title and description
st.title("SMS SPAM CLASSIFICATION APPLICATION")
st.write("This is a Machine Learning application to classify SMS as spam or not.")
st.write("Enter an SMS below to check if it's spam.")

# Text input and button in the same layout
user_input = st.text_area("Enter an SMS to classify:", height=150)

if st.button("Classify"):
    if user_input:
        # Preprocessing and classification
        data = [user_input]
        vectorized_data = cv.transform(data).toarray()
        result = model.predict(vectorized_data)

        if result[0] == 0:
            st.success("The SMS is **not spam**.")
        else:
            st.warning("The SMS is **spam**.")

        # If predict_proba is available, include confidence score
        try:
            probabilities = model.predict_proba(vectorized_data)
            confidence = probabilities[0][1] if result[0] == 1 else probabilities[0][0]
            st.write(f"Confidence: **{confidence * 100:.2f}%**")
        except AttributeError:
            st.info("Confidence score not available for this model.")
    else:
        st.warning("Please type an SMS to classify.")

# Additional styling or footer
st.write("---")
st.write("Created with ❤️. Improve your SMS filtering with this app!")
