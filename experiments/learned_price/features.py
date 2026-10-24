"""Source word/character TF-IDF features, fitted on the probing split only."""
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_pair():
    return (TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True,
                            token_pattern=r'(?u)\b\w+\b|[^\w\s]', dtype=np.float32),
            TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, max_features=100000, sublinear_tf=True, dtype=np.float32))

def fit_block(train_docs):
    wv, cv = tfidf_pair(); wv.fit(train_docs); cv.fit(train_docs)
    return lambda docs: (sparse.hstack([wv.transform(docs), cv.transform(docs)]).tocsr() / np.sqrt(2)).astype(np.float32)
