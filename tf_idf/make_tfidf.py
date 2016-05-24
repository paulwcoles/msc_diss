# -*- coding: UTF-8 -*-
##
# Author: Paul W. Coles
##
# Make and print TF-IDF matrix from the MGB corpus
##

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import os
import string

def make_corpus(path):
    for subdir, dirs, files in os.walk(path):
        for f in files:
            file_path = subdir + os.path.sep + f
            mgb = open(file_path, 'r')
            text = mgb.read()
            text_decoded = text.decode("utf-8", "replace")
            lower_case = text_decoded.lower()
            corpus[f] = lower_case

if __name__ == "__main__":
    rootdir = "../data/docs_mgbtrain_small/"
    corpus = {}
    make_corpus(rootdir)
    vectorizer = TfidfVectorizer(min_df=1)
    tf_idf_matrix = vectorizer.fit_transform(corpus.values(), y=None)
    print tf_idf_matrix
