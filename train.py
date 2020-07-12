# from process import preprocessed_data
# from sklearn.compose import make_column_transformer
# from string import punctuation
# from sklearn.preprocessing import StandardScaler
import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
nltk.download('stopwords')


def tokenize(text):
    """ The function performs tghe underlisted tasks
    - Removes punctuations
    - Remove stopwords
    - Normalises the texts
    - Removes whitespaces

    Args :
    text: list of texts to be tokenized

    Returns:
    clean copy of the tokenized text
    """
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # tokenize text
    tokens = word_tokenize(text)
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(
        word, pos='v'
    ) for word in tokens if word not in stop_words
    ]
    return tokens


def split_data(df, col1, col2):
    """ split data into training and test sets
    Arg:
        df - preprocessed data
        col1- target
    Returns:
        train_features, test_features, train_target, test_target
    """
    print('splitting dataset into training and test sets')
    train_features, test_features, train_target,\
        test_target = train_test_split(
            df[col2], df[col1], test_size=0.2,
            random_state=42)
    print('train_features:  ', train_features.shape)
    print('test_features:  ', test_features.shape)
    print('train_target:  ', train_target.shape)
    print('test_target:  ', test_target.shape)
    return train_features, test_features, train_target, test_target


def build_model():
    """
    Build machine learning model using pipeline
    Args:
    cat_col-  categorical columns
    num_col- numerical columns
        Returns:
        model

    """
    print('Building preprocesseing pipeline')

    print('\n', 'Creating model pipeline')
    preprocessor_pipe = make_pipeline(
        (CountVectorizer(stop_words='english')),
        (TfidfTransformer())
    )
    return preprocessor_pipe


# define variables
col1 = 'label'
col2 = 'sms_message'
file_path = './data/SMSSpamCollection'
preprocessor_path = './models/preprocessor.pkl'
model_path = './models/model.pkl'
df = pd.read_table(file_path, sep='\t', names=[col1, col2])
# df = preprocessed_data()
# cat_col = 'sms_message'
# num_col = list(df.columns[2:])


def main(df, col1, col2, model_path):
    train_features, test_features, train_target,\
        test_target = split_data(df, col1, col2)

    # print(train_features.shape)
    # print('\n')
    # print(train_target.shape)
    # print('\n\n')
    print('fitting model')
    preprocessor_pipe = build_model()
    train_features = preprocessor_pipe.fit_transform(train_features)
    model = LogisticRegression()
    model.fit(train_features, train_target)

    # Save preprocessing pipeline to directory
    print("Saving preprocessing pipe line to {}".format(preprocessor_path))
    with open(preprocessor_path, 'wb') as file:
        joblib.dump(preprocessor_pipe, file)
        print("\npreprocessor pipeline saved\n")

# Save model to directory
    print("Saving model to {}".format(model_path))
    with open(model_path, 'wb') as file:
        joblib.dump(model, file)
    print("\nmodel saved\n")


if __name__ == '__main__':
    main(df, col1, col2, model_path)
