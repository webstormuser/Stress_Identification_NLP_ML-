import nltk
from nltk.corpus import stopwords
import pickle

# Download the stopwords corpus
nltk.download('stopwords')

# Get the stopwords
stop_words = set(stopwords.words('english'))

# Save the stopwords to a pickle file
with open('stopwords.pickle', 'wb') as f:
    pickle.dump(stop_words, f)
