from ast import literal_eval

# import nltk

from imblearn.over_sampling import RandomOverSampler
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import FrenchStemmer


# download nltk ressources
# nltk.download('punkt')
# nltk.download('stopwords')

stemmer = FrenchStemmer("french")
stopwords = list(stopwords.words('french'))


def preprocess_text(text):
    # Tokenization du texte + Retirer les stopwords
    return " ".join([stemmer.stem(word) for word in word_tokenize(text.lower(), language='french') if word.lower() not in stopwords])


def extract_features_is_name(df):
    sentence_end_punctuations = ['.', '!', '?', ';', '."', '!"', '?"', ';"']
    df['is_name'] = df['is_name'].apply(literal_eval)
    df['tokens'] = df['tokens'].apply(literal_eval)
    exploded_data = df.explode(['tokens', 'is_name']).reset_index(drop=True)

    # init columns
    exploded_data["is_final_word"]: bool = False
    exploded_data["is_starting_word"]: bool = False
    exploded_data["is_capitalized"]: bool = False
    exploded_data["previous_word"]: str = ""
    exploded_data["following_word"]: str = ""

    for i in range(len(exploded_data)):
        token: str = exploded_data.at[i, "tokens"]
        is_new_sentence: bool = (exploded_data.at[i - 1, "tokens"] in sentence_end_punctuations) if i != 0 else False

        if i == 0 or is_new_sentence:
            exploded_data.at[i, "is_starting_word"] = True

        if i != 0 and (token in sentence_end_punctuations or is_new_sentence):
            exploded_data.at[i - 1, "is_final_word"] = True
        if i == len(exploded_data) - 1:
            exploded_data.at[i, "is_final_word"] = True

        if token not in sentence_end_punctuations and token and token[0].isupper():
            exploded_data.at[i, "is_capitalized"] = True

        if i != 0 and not exploded_data.at[i, "is_starting_word"]:
            exploded_data.at[i, "previous_word"] = exploded_data.at[i - 1, "tokens"]

        if token not in sentence_end_punctuations and i != len(exploded_data) - 1:
            exploded_data.at[i, "following_word"] = exploded_data.at[i + 1, "tokens"]

    return (
        exploded_data[
            [
                "is_final_word",
                "is_starting_word",
                "is_capitalized",
                "previous_word",
                "following_word",
            ]
        ],
        exploded_data["is_name"].astype(int)
    )


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
