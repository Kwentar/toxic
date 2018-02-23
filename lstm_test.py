import re
from nltk.corpus import stopwords

import numpy as np
import pandas as pd

stop_words = set(stopwords.words('english'))


def create_features(file_name):
    df = pd.read_csv(file_name)

    df['count_sent'] = df["comment_text"].apply(lambda x: len(re.findall("\n", str(x)))+1)

    # Unique word count
    df['count_unique_word'] = df["comment_text"].apply(lambda x: len(set(str(x).split())))

    # title case words count
    df["count_words_title"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    df["count_words_upper"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

    # Average length of the words
    df["mean_word_len"] = df["comment_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    df["mean_word_len"] = df["mean_word_len"].apply(lambda x: x if x < 30 else 30)
    df['count_word'] = df["comment_text"].apply(lambda x: len(str(x).split()))
    df['word_unique_percent'] = df['count_unique_word']/df['count_word']
    df['count_word'] = df["count_word"].apply(lambda x: x if x < 300 else 300)
    df['count_unique_word'] = df["count_unique_word"].apply(lambda x: x if x < 300 else 300)

    # derived features
    toxic_words = ['fuck', 'shit', 'suck', 'bitch', 'stupid']
    threat_words = ['kill', 'die', 'rape', 'death']
    identity_words = ['gay', 'nigger', 'jew']
    df['toxic_count'] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w in toxic_words]))
    df['threat_count'] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w in threat_words]))
    df['identity_count'] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w in identity_words]))

    list_classes = ['count_sent', 'count_unique_word', 'count_words_title', 'count_words_upper',
                    'mean_word_len', 'word_unique_percent', 'toxic_count', 'threat_count', 'identity_count']

    return df[list_classes].values
