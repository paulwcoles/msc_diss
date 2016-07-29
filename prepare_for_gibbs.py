import os
import time
import re
import sys
from nltk.corpus import stopwords
from nltk.stem.porter import *

mode = sys.argv[1]
if mode == 'train':
    src_dir = './data/today/parsed/'
    out_dir = './GibbsLDA++-0.2/models/today_train/'
elif mode == 'test':
    src_dir = './data/today/test/'
    out_dir = './GibbsLDA++-0.2/models/today_test/'

def stem(word_list):
	stemmed = []

	return stemmed

stemmer = PorterStemmer()
# Prepare outfile for gibbs LDA

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
with open(out_dir + (time.strftime("%d_%m_%Y")) + '_' + mode + '.dat', 'w') as outfile:
## Write to outfile
    for root, dirs, single_docs in os.walk(src_dir):
        m = str(len(single_docs))
        outfile.write(m + '\n')
        english_stops = stopwords.words('english')
        for single_doc in single_docs:
            clean = []
            with open((os.path.join(root, single_doc)), 'r') as doc:
                text = doc.read()
                word_list = text.split()
                for word in word_list:
                    word = stemmer.stem(word)
                    if word.lower() not in english_stops:
                        clean.append(word)
                for clean_word in clean:
                    outfile.write(clean_word + ' ')
                outfile.write('\n')
