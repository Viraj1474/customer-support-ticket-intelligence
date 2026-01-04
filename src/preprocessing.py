import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

_stop_words = set(stopwords.words('english'))
_lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """
    Clean and normalize ticket text for NLP tasks.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in _stop_words]
    words = [_lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)
