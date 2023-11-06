import nltk

from imblearn.over_sampling import RandomOverSampler
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import FrenchStemmer


# download nltk ressources
nltk.download('punkt')
nltk.download('stopwords')

stemmer = FrenchStemmer("french")
stopwords = list(stopwords.words('french'))

def preprocess_text(text):
    # Tokenization du texte + Retirer les stopwords
    return " ".join([stemmer.stem(word) for word in word_tokenize(text.lower(), language='french') if word.lower() not in stopwords])


def extract_features_is_name(df):
    X, y = [], []

    for _, row in df.iterrows():
        video_name_words = row['video_name'].split()
        for i, (word, label) in enumerate(zip(video_name_words, row['is_name'])):

            word_features = {
                'is_starting_word': i == 0,
                'is_final_word': i == len(video_name_words) - 1,
                'is_capitalized': word.istitle(),
                'prev_word': video_name_words[i - 1] if i > 0 else '',
                'next_word': video_name_words[i + 1] if i < len(video_name_words) - 1 else '',
            }

            X.append(word_features)
            y.append(label)

    return X, y


def extract_features_find_comic_name(df):
    all_features = []
    all_labels = []

    for _, row in df.iterrows():
        video_name = row['video_name']
        comic_name = row['comic_name']
        tokens = word_tokenize(video_name, language='french')
        video_length = len(tokens)
        num_capitalized = sum(1 for token in tokens if token[0].isupper())

        for _, token in enumerate(tokens):
            is_capitalized = token[0].isupper()
            word_features = [is_capitalized, video_length, num_capitalized]
            all_features.append(word_features)
            all_labels.append(1 if token in comic_name else 0)

    return all_features, all_labels


def make_features(df, task):
    if task == "is_comic_video":
        return df["video_name"].apply(preprocess_text), df["is_comic"]
    elif task == "is_name":
        return extract_features_is_name(df)
    elif task == "find_comic_name":
        X, y = extract_features_find_comic_name(df)
        ros = RandomOverSampler(random_state=42)

        return ros.fit_resample(X, y) 

    raise ValueError("Unknown task")
