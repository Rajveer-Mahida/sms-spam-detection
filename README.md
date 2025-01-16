# SMS Spam Detection Streamlit App

This is a Machine Learning application built with Streamlit to classify SMS messages as spam or not spam. The app uses a trained model to predict the classification of the input SMS.

## Features

- Classify SMS messages as spam or not spam.
- Display confidence score for the classification.
- User-friendly interface built with Streamlit.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/sms-spam-detection.git
    cd sms-spam-detection/streamlit-app
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Enter an SMS message in the text area and click "Classify" to see if it's spam or not.

## File Structure

- `app.py`: Main application file for the Streamlit app.
- `requirements.txt`: List of required Python packages.
- `src/models/`: Trained model file.
- `src/data/sms-spam.csv`: Dataset file containing SMS messages and their labels.

## Error Handling

The app includes error handling for loading the model and vectorizer. If the files are not found, an error message will be displayed.


## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)
- [numpy](https://numpy.org/)
- [scipy](https://www.scipy.org/)
