from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.sparse import csr_matrix
import scipy

# a = sparse.rand(5,5)
# b = sparse.rand(5,5)

def convert_to_windowed(in_matrix, window_size):
    out_matrix = csr_matrix(in_matrix.get_shape(), dtype=int)
    out_matrix = out_matrix.tolil()
    n_vectors = out_matrix.get_shape()[0]
    # Relative to current utt vector, window starts one position forward and back
    window_range = range(1, window_size + 1)
    print out_matrix.get_shape()
    for i in range(n_vectors):
        out_matrix[i] += in_matrix[i]
        for j in window_range:
            # Window left if possible
            if i - j >= 0:
                out_matrix[i] += (in_matrix[i - j])
            # Window right if possible
            if i + j < n_vectors:
                out_matrix[i] +=  in_matrix[i + j]
    return out_matrix

def make_cosine_list(matrix):
    # Returns list: nth index in list is cosine sim of nth and nth + 1 utt in matrix
    n_vectors = matrix.shape[0]
    cosines = []
    # Iterate over each utt in doc_matrix, score with subsequent utt (final utt cannot have cosine score)
    for i in range(n_vectors - 1):
        if switch == "tfidf":
            cosines.append(distance.cosine(matrix[i].todense(), matrix[i+1].todense()))
        elif switch == "lda":
            cosines.append(distance.cosine(matrix[i], matrix[i+1]))
    return cosines

a = ["A hand-painted flag of so-called Islamic State has been found in the room of an Afghan refugee", "A hand-painted banner of so-called Islamic Country has been found in the room of an Iran refugee"]
count_vectorizer = CountVectorizer(min_df=0, max_features=100)
tf_matrix = count_vectorizer.fit_transform(a)
print convert_to_windowed(tf_matrix, 3)


# print tf_matrix