import streamlit as st
import pandas as pd
import pickle
import nltk
import re
from urllib.parse import urlparse
from spacy import load
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.sparse import hstack

# Load the saved XGBoost model
with open('xgb_model.pickle', 'rb') as f:
    xgb_model = pickle.load(f)

# Load the saved label encoder
with open('label_encoder.pickle', 'rb') as f:
    le = pickle.load(f)

# Load the saved TF-IDF vectorizer
with open('tfidf_vectorizer.pickle', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
    
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def textProcess(sent):
    try:
        # Brackets replacing by space
        sent = re.sub(r'\[\{}]', ' ', sent)
        # URL removing
        sent = [word for word in sent.split() if not urlparse(word).scheme]
        sent = ' '.join(sent)
        # Removing escape characters
        sent = re.sub(r'\@\w+', '', sent)
        # Removing HTML tags
        sent = re.sub(r'<.*?>', '', sent)
        # Getting only characters and numbers from text
        sent = re.sub(r'[^A-Za-z0-9]', ' ', sent)
        # Lowercase all words
        sent = sent.lower()
        # Strip all words from sentences
        sent = ' '.join(word.strip() for word in sent.split())
        # Word tokenization
        tokens = word_tokenize(sent)
        # Removing words which are in stopwords
        tokens = [word for word in tokens if word not in stop_words]
        # Lemmatization
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        # Join tokens back to a single string
        sent = ' '.join(tokens)
        return sent

    except Exception as ex:
        print("Error:", ex)
        return sent
    

# Prediction code 
def predict_new_text(new_text):
    # Preprocess the new text
    processed_text = textProcess(new_text)

    # Convert the preprocessed text to TF-IDF features
    tfidf_new = tfidf_vectorizer.transform([processed_text])

    # Extract numerical features (you might need to adjust this based on your feature engineering)
    numerical_features_new = pd.DataFrame({'len_of_sentences': [len(new_text.split('.'))], 'len_characters': [len(new_text)]})

    # Combine the TF-IDF and numerical features
    new_text_combined = hstack((tfidf_new, numerical_features_new))

    # Make predictions using the loaded model
    prediction = xgb_model.predict(new_text_combined)

    # Inverse transform the prediction to get the original label
    predicted_label = le.inverse_transform(prediction)[0]

    return predicted_label

# Streamlit app
st.title('Stress Category Prediction')
st.write('Enter text to predict its stress category:')

user_input = st.text_area('Enter your text here')

if st.button('Predict'):
    if user_input:
        result = predict_new_text(user_input)
        st.write(f'The predicted stress category is: {result}')
    else:
        st.write('Please enter some text to get a prediction.')
