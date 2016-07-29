from __future__ import print_function
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from time import time
import os

n_samples = 2000
n_features = 1000
n_topics = 10
n_top_words = 20


def load_corpus(dir):
    print("Loading corpus...")
    corpus = []
    for doc_file in os.listdir(docs_dir):
        with open(docs_dir + doc_file, 'r') as f:
            corpus.append(f.read())
    return corpus


def get_tf_vectors(corpus):
    print("Extracting tf features for LDA...")
    return tf_vectorizer.fit_transform(corpus)


def fit_lda_model():
    print("Fitting LDA models with tf features, n_samples=%d and n_features=%d..."
          % (n_samples, n_features))
    lda.fit(tf)

def print_top_words():
    print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(lda.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([tf_feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


if __name__ == '__main__':
    docs_dir = '../data/today/parsed/'
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features,
                                        stop_words='english')
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                    learning_method='online', learning_offset=50.,
                                    random_state=0)
    corpus = load_corpus(docs_dir)
    tf = get_tf_vectors(corpus)
    fit_lda_model()
    print_top_words()
