import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Ensure necessary NLTK data is downloaded
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")


# Define preprocessing functions
def case_folding(text):
    return text.lower()


def cleaning(text):
    # Remove URL
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenization(text):
    return word_tokenize(text)


def remove_stopwords(tokens):
    return [word for word in tokens if word not in stopwords.words("english")]


# Function to map NLTK POS tags to WordNet POS tags (used in lemmatization process)
def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to NOUN if no match


def lemmatization(tokens):
    lemmatizer = WordNetLemmatizer()
    pos_tagged = pos_tag(tokens)
    lemmatized_tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tagged
    ]
    return lemmatized_tokens


def single_text_preprocessing(text):
    # Apply case folding function
    text = case_folding(text)
    # Apply clearning function
    text = cleaning(text)
    # Apply tokenization function
    text = tokenization(text)
    # Apply remove stopwords function
    text = remove_stopwords(text)
    # Apply lemmatization function
    text = lemmatization(text)
    return text
