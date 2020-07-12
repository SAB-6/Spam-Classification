import pandas as pd
from string import punctuation


class Data:
    def create_numeric(x): return 1 if x == 'spam' else 0

    def __init__(self, file_path, col1, col2):
        self.file_path = file_path
        self.col1 = col1
        self.col2 = col2

    def read_data(self):
        """ Read csv file
        Args:
            file_path : file path for the spam and
            non-spam messages
        Returns:
            dataframe containing the spam/non-spam messages
            """
        print('Loading data')
        df = pd.read_table(self.file_path, sep='\t', names=[self.col1, self.col2])
        return df

    def preprocess_data(self):
        """ The function turns target column to numeric and added two
        additioanal features(columns)
        Additional features added are:
        - message length
        - no of punctuations
          Args:
            None

        Returns:
            data: preprocessed dataframe
            """
        df = self.read_data()

        print('Turning target column to numeric')
        dict_name = {'spam': 1, 'ham': 0}

        df[self.col1] = df[self.col1].map(dict_name)

        # print(df.head())
        # df.head()
        return df


# Define relevant variables
file_path = 'data/SMSSpamCollection'
col1 = 'label'
col2 = 'sms_message'


def count_punct(x):
    # create a column to count the number of punctuation
    # excluding comma and full stop
    punct = [i for i in punctuation if i not in (',', '.')]
    return sum([1 for i in x if i in punct])


def preprocessed_data():
    print('Loading data')
    df = Data(file_path, col1, col2)

    print('Preprocessing data')
    df = df.preprocess_data()

    print('Generating additional features')
    print(df[col2].head(), '\n\n')

    # print(df.head())
    df['msg_length'] = df[col2].apply(lambda x: len(x.split()))

    df['count_punct'] = (df[col2]).apply(count_punct)

    # print(df.columns)
    return df


if __name__ == "__main__":
    df = preprocessed_data()
    print(list(df.columns))
