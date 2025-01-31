import streamlit as st
import pickle
import os
import warnings

# Suppress specific sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    
# Load model and vectorizer with error handling
model_path = './src/models/Random_Forest_Classifier.pkl'
# model_path = './src/models/Decision_Tree_Classifier.pkl'
vectorizer_path = './src/models/vectorizer.pkl'

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.error("Model or vectorizer files not found. Please check the file paths.")
    st.stop()

try:
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(vectorizer_path, 'rb') as vectorizer_file:
        cv = pickle.load(vectorizer_file)
except Exception as e:
    st.error(f"Error loading model or vectorizer: {str(e)}")
    st.stop()

# App title and description
st.title("SMS SPAM CLASSIFICATION APPLICATION")
st.write("This is a Machine Learning application to classify SMS as spam or not.")
st.write("Enter an SMS below to check if it's spam.")

# Text input and button in the same layout
user_input = st.text_area("Enter an SMS to classify:", height=150)

if st.button("Classify"):
    if not user_input.strip():
        st.warning("Please type an SMS to classify.")
    else:
        try:
            # Preprocessing and classification
            data = [user_input.lower()]  # Convert to lowercase before vectorization
            try:
                vectorized_data = cv.transform(data).toarray()
            except Exception as vec_error:
                st.error("Error during text vectorization. The model's vocabulary may not match the input text.")
                st.stop()
                
            result = model.predict(vectorized_data)

            # Display classification result
            if result[0] == 0:
                st.success("The SMS is **Not spam**.")
            else:
                st.warning("The SMS is **Spam**.")

            # If predict_proba is available, include confidence score
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(vectorized_data)
                confidence = probabilities[0][1] if result[0] == 1 else probabilities[0][0]
                st.write(f"Confidence: **{confidence * 100:.2f}%**")
            else:
                st.info("Confidence score not available for this model.")
        except Exception as e:
            st.error(f"Error during classification: {str(e)}")

# Additional styling or footer
st.write("---")
st.write("Created with ❤️ | Improve your SMS filtering with this app!")
